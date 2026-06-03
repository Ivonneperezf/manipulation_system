#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import cv2
import rospkg
import tempfile
import os
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from ultralytics.models.sam import SAM3SemanticPredictor
import sensor_msgs.point_cloud2 as pc2
from typing import List, Tuple, Optional


class MaskFilter:
    """
    Filtra y limpia un conjunto de máscaras binarias crudas provenientes de SAM3.

    PARAMETROS
    min_area          : píxeles mínimos para no ser considerado ruido
    max_area_ratio    : fracción máxima del área de imagen permitida
                        (bajar este valor descarta la máscara "fruta entera")
    min_compactness   : relación mínima área_máscara / área_bbox
    containment_thresh: proporción mínima de i dentro de j para eliminar j
                        (subir este valor evita eliminar cubitos adyacentes)
    size_ratio_min    : j debe ser al menos esta vez más grande que i para
                        ser candidato a "contenedora"
    """

    def __init__(self,
                 min_area=100,
                 max_area_ratio=0.06,
                 min_compactness=0.10,
                 containment_thresh=0.92,
                 size_ratio_min=3.0):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_compactness = min_compactness
        self.containment_thresh = containment_thresh
        self.size_ratio_min = size_ratio_min

    def _basic_filter(self, masks, img_area):
        kept_masks, kept_indices = [], []
        for idx, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            mask_area = mask_bool.sum()

            if mask_area < self.min_area:
                continue
            if mask_area / img_area > self.max_area_ratio:
                continue

            ys, xs = np.where(mask_bool)
            bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            if mask_area / (bbox_area + 1e-6) < self.min_compactness:
                continue

            kept_masks.append(mask_bool)
            kept_indices.append(idx)
        return kept_masks, kept_indices

    def _remove_containing_masks(self, masks, original_indices):
        n = len(masks)
        if n <= 1:
            return masks, original_indices

        flat = np.stack([m.ravel() for m in masks])
        areas = flat.sum(axis=1).astype(np.float32)
        inter = (flat @ flat.T).astype(np.float32)
        containment = inter / (areas[:, None] + 1e-6)
        size_ratio = areas[None, :] / (areas[:, None] + 1e-6)
        np.fill_diagonal(containment, 0.0)
        np.fill_diagonal(size_ratio, 0.0)

        is_container = (size_ratio >= self.size_ratio_min) & (containment >= self.containment_thresh)
        to_remove = is_container.any(axis=0)

        return (
            [m for m, r in zip(masks, to_remove) if not r],
            [idx for idx, r in zip(original_indices, to_remove) if not r],
        )

    def clean(self, raw_masks, img_area):
        masks, indices = self._basic_filter(raw_masks, img_area)
        return self._remove_containing_masks(masks, indices)


class KinovaVisionSAM3:
    def __init__(self):
        rospy.init_node("vision_d415_sam3")
        self.pub = rospy.Publisher("object_centroid", PointStamped, queue_size=10)

        self.TOPIC_RGB    = rospy.get_param("~topics/rgb_topic",         "/d415/color/image_raw")
        self.TOPIC_INFO   = rospy.get_param("~topics/camera_info_topic", "/d415/color/camera_info")
        self.TOPIC_POINTS = rospy.get_param("~topics/points_topic",      "/d415/depth/points")

        self.prompts = ["a small cube of fruit"]

        # Filtros ajustados para cubitos individuales:
        #   max_area_ratio  bajo  → descarta la máscara "fruta entera"
        #   containment_thresh alto → no elimina cubitos que se tocan
        #   size_ratio_min  alto  → solo elimina contenedoras muy grandes
        self.mask_filter = MaskFilter(
            min_area=100,
            max_area_ratio=0.06,
            min_compactness=0.10,
            containment_thresh=0.92,
            size_ratio_min=3.0,
        )

        # Fracción del grupo usada como candidatos centrales (0.30 = top 30%)
        self.CENTER_FRACTION = 0.30

        self.is_processing = False

        rospy.loginfo("Cargando SAM3...")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(rospy.get_param("~paths/pack", "statemachine"))

        overrides = dict(
            conf=0.2,
            task="segment",
            mode="predict",
            model=os.path.join(package_path, "weights", "sam3.pt"),
            iou=0.3,
            half=True,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        try:
            info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]
            self.cam_frame   = info.header.frame_id
        except rospy.ROSException:
            rospy.logerr("No se detectó la cámara.")
            return

        self.last_cloud: Optional[np.ndarray] = None
        self._tmp_dir = tempfile.mkdtemp(prefix="sam3_ros_")
        self._tmp_frame_path = os.path.join(self._tmp_dir, "current_frame.jpg")

        rospy.Subscriber(self.TOPIC_RGB,    Image,        self.rgb_cb,   queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.TOPIC_POINTS, PointCloud2,  self.cloud_cb, queue_size=1)
        rospy.loginfo("Nodo SAM3 Listo.")

    def cloud_cb(self, msg: PointCloud2) -> None:
        puntos = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if puntos:
            self.last_cloud = np.array(puntos, dtype=np.float32)

    def rgb_cb(self, msg: Image) -> None:
        if self.is_processing:
            return
        try:
            self.is_processing = True
            self._process_frame(msg)
        except Exception as exc:
            rospy.logerr(f"Error en rgb_cb: {exc}")
        finally:
            self.is_processing = False

    def _normalize_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        CLAHE en el canal L de LAB: mejora el contraste local entre cubitos
        sin importar si la luz es fuerte, tenue o lateral.
        El frame_bgr original NO se toca; se usa solo para visualización.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _select_target(self,
                       all_clean_masks: List[np.ndarray],
                       z_values: List[float],
                       img_w: int,
                       img_h: int) -> int:
        """
        Criterio de selección del cubito objetivo:
          1. Calcular el centroide en píxeles de cada máscara.
          2. Calcular el centro de masa del grupo completo.
          3. Ordenar todos los cubitos por distancia a ese centro.
          4. Tomar el top CENTER_FRACTION (mínimo 1) más centrados.
          5. De esos candidatos, elegir el de menor Z (más cercano a la cámara).

        Robusto ante:
          - Un solo cubito          → n_candidatos=1, se toma directamente.
          - Cubitos muy separados   → no hay radio fijo, siempre hay candidatos.
          - Cubitos en la orilla    → quedan fuera del top central.
        """
        n = len(all_clean_masks)

        # ── Centroides en píxeles ──────────────────────────────────────
        centroids_uv = []
        for mask_bool in all_clean_masks:
            mask_uint8 = mask_bool.astype(np.uint8)
            mask_h, mask_w = mask_uint8.shape
            M = cv2.moments(mask_uint8)
            if M["m00"] == 0:
                centroids_uv.append((img_w / 2.0, img_h / 2.0))
                continue
            u = M["m10"] / M["m00"] * img_w / mask_w
            v = M["m01"] / M["m00"] * img_h / mask_h
            centroids_uv.append((u, v))

        centroids_arr = np.array(centroids_uv)          # (N, 2)
        group_center  = centroids_arr.mean(axis=0)      # (u_mean, v_mean)
        dists         = np.linalg.norm(centroids_arr - group_center, axis=1)

        # ── Caso trivial: un solo cubito ───────────────────────────────
        if n == 1:
            return 0

        # ── Top CENTER_FRACTION más centrados ──────────────────────────
        n_candidatos = max(1, int(n * self.CENTER_FRACTION))
        candidatos   = np.argsort(dists)[:n_candidatos].tolist()

        # ── De esos, el menor Z ────────────────────────────────────────
        z_candidatos = [z_values[i] for i in candidatos]
        return candidatos[int(np.argmin(z_candidatos))]

    def _process_frame(self, msg: Image) -> None:
        frame_rgb = ros_numpy.numpify(msg)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w = frame_rgb.shape[:2]
        img_area = img_h * img_w

        # SAM3 recibe la imagen normalizada (más robusto a cambios de luz)
        # La visualización sigue usando frame_bgr original
        frame_bgr_norm = self._normalize_image(frame_bgr)
        cv2.imwrite(self._tmp_frame_path, frame_bgr_norm)
        self.predictor.set_image(self._tmp_frame_path)
        results = self.predictor(text=self.prompts)

        all_clean_masks: List[np.ndarray] = []
        all_original_indices: List[int]   = []
        target_result = None

        # ── 1. Recopilar y filtrar todas las máscaras ──────────────────
        for result in results:
            if result.masks is None:
                continue

            raw_masks = result.masks.data.cpu().numpy()

            after_basic, after_idx = self.mask_filter._basic_filter(raw_masks, img_area)
            after_clean, after_clean_idx = self.mask_filter._remove_containing_masks(
                after_basic, after_idx
            )
            rospy.loginfo(
                f"Máscaras → crudas: {len(raw_masks)} | "
                f"tras basic: {len(after_basic)} | "
                f"tras containment: {len(after_clean)}"
            )

            for m, idx in zip(after_clean, after_clean_idx):
                all_clean_masks.append(m)
                all_original_indices.append(idx)
                target_result = result

        # ── 2. Verificar que hay máscaras ──────────────────────────────
        if not all_clean_masks:
            rospy.logwarn_throttle(3, "No se detectaron cubitos en este frame.")
            cv2.imshow("SAM3 Segmentation", frame_bgr)
            cv2.waitKey(1)
            return

        rospy.loginfo(f"Cubitos candidatos tras filtro: {len(all_clean_masks)}")

        # ── 3. Calcular Z de cada máscara ──────────────────────────────
        z_values: List[float] = []
        for mask_bool in all_clean_masks:
            mask_uint8 = mask_bool.astype(np.uint8)
            z = self.get_depth_from_mask(mask_uint8, frame_bgr.shape)
            z_values.append(z if z > 0.0 else float("inf"))

        # ── 4. Seleccionar objetivo: más centrado + menor Z ────────────
        target_idx = self._select_target(all_clean_masks, z_values, img_w, img_h)

        if z_values[target_idx] == float("inf"):
            rospy.logwarn("El cubito objetivo no tiene profundidad válida.")
            cv2.imshow("SAM3 Segmentation", frame_bgr)
            cv2.waitKey(1)
            return

        # ── 5. Extraer datos del cubito ganador ────────────────────────
        mask_bool      = all_clean_masks[target_idx]
        orig_idx       = all_original_indices[target_idx]
        z_m            = z_values[target_idx]
        mask_uint8     = mask_bool.astype(np.uint8)
        mask_h, mask_w = mask_uint8.shape

        # Nombre de clase
        if (
            target_result
            and target_result.boxes is not None
            and orig_idx < len(target_result.boxes.cls)
        ):
            cls_id   = int(target_result.boxes.cls[orig_idx])
            obj_name = target_result.names[cls_id]
        else:
            obj_name = "fruit"

        # Centroide en imagen
        M = cv2.moments(mask_uint8)
        if M["m00"] == 0:
            rospy.logwarn("Centroide inválido para la máscara objetivo.")
            cv2.imshow("SAM3 Segmentation", frame_bgr)
            cv2.waitKey(1)
            return

        u_mask = M["m10"] / M["m00"]
        v_mask = M["m01"] / M["m00"]
        u = int(u_mask * img_w / mask_w)
        v = int(v_mask * img_h / mask_h)

        # Coordenadas 3D en frame de cámara
        x_c = (u - self.cx) * z_m / self.fx
        y_c = (v - self.cy) * z_m / self.fy

        # ── 6. Visualización ───────────────────────────────────────────
        mask_resized = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        contours, _  = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 3)
        cv2.circle(frame_bgr, (u, v), 8, (0, 0, 255), -1)
        cv2.putText(
            frame_bgr,
            f"TARGET: {obj_name} ({x_c:.3f}, {y_c:.3f}, {z_m:.3f})",
            (u + 10, v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

        # ── 7. Publicar y mostrar ──────────────────────────────────────
        self._publish_centroid(x_c, y_c, z_m)

        cv2.imshow("SAM3 Segmentation", frame_bgr)
        cv2.waitKey(1)

    def get_depth_from_mask(self, mask: np.ndarray, frame_shape: Tuple[int, int, int]) -> float:
        if self.last_cloud is None:
            return 0.0

        mask_h, mask_w = mask.shape
        img_h, img_w   = frame_shape[:2]
        points_3d      = self.last_cloud.copy()

        valid     = points_3d[:, 2] > 0
        points_3d = points_3d[valid]
        if len(points_3d) == 0:
            return 0.0

        u_arr = (points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx).astype(np.int32)
        v_arr = (points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy).astype(np.int32)

        in_bounds = (u_arr >= 0) & (u_arr < img_w) & (v_arr >= 0) & (v_arr < img_h)
        u_arr, v_arr = u_arr[in_bounds], v_arr[in_bounds]
        z_arr = points_3d[in_bounds, 2]

        u_mask_idx = (u_arr * mask_w / img_w).astype(np.int32)
        v_mask_idx = (v_arr * mask_h / img_h).astype(np.int32)

        in_bounds2 = (
            (u_mask_idx >= 0) & (u_mask_idx < mask_w) &
            (v_mask_idx >= 0) & (v_mask_idx < mask_h)
        )
        u_mask_idx = u_mask_idx[in_bounds2]
        v_mask_idx = v_mask_idx[in_bounds2]
        z_arr      = z_arr[in_bounds2]

        in_mask  = mask[v_mask_idx, u_mask_idx] > 0
        z_masked = z_arr[in_mask]

        return float(np.mean(z_masked)) if len(z_masked) > 0 else 0.0

    def _publish_centroid(self, x: float, y: float, z: float) -> None:
        msg = PointStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = self.cam_frame
        msg.point.x, msg.point.y, msg.point.z = x, y, z
        self.pub.publish(msg)
        rospy.loginfo(f"[PICK] TARGET → X={x:.3f} Y={y:.3f} Z={z:.3f}")

if __name__ == "__main__":
    try:
        KinovaVisionSAM3()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()