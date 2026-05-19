#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import cv2
import rospkg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from ultralytics.models.sam import SAM3SemanticPredictor
import sensor_msgs.point_cloud2 as pc2

#Llimpieza de mascaras
class MaskFilter:
    """
    Filtra y limpia un conjunto de máscaras binarias crudas provenientes de SAM3.

    PARAMETROS
    min_area          : píxeles mínimos para no ser considerado ruido
    max_area_ratio    : fracción máxima del área de imagen permitida
    min_compactness   : relación mínima área_máscara / área_bbox
    containment_thresh: proporción mínima de i dentro de j para eliminar j
    size_ratio_min    : j debe ser al menos esta vez más grande que i para
                        ser candidato a "contenedora"
    """

    def __init__(
        self,
        min_area: int = 200,
        max_area_ratio: float = 0.15,
        min_compactness: float = 0.10,
        containment_thresh: float = 0.85,
        size_ratio_min: float = 2.5,
    ):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_compactness = min_compactness
        self.containment_thresh = containment_thresh
        self.size_ratio_min = size_ratio_min

    #FILTRO 1: descarta ruido, objetos gigantes y formas raras

    def _basic_filter(
        self, masks: np.ndarray, img_area: int
    ) -> tuple[list[np.ndarray], list[int]]:
        """ Devuelve (máscaras_válidas, índices_originales).
        Trabajamos con bool arrays para eficiencia."""
        kept_masks: list[np.ndarray] = []
        kept_indices: list[int] = []

        for idx, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            mask_area = mask_bool.sum()

            #ruido
            if mask_area < self.min_area:
                continue

            #demasiado grande
            if mask_area / img_area > self.max_area_ratio:
                continue

            #compacidad: descarta L-shapes y formas degeneradas
            ys, xs = np.where(mask_bool)
            bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
            if mask_area / (bbox_area + 1e-6) < self.min_compactness:
                continue

            kept_masks.append(mask_bool)
            kept_indices.append(idx)

        return kept_masks, kept_indices

    #FILTRO 2: elimina máscaras contenedoras
    def _remove_containing_masks(
        self, masks: list[np.ndarray], original_indices: list[int]
    ) -> tuple[list[np.ndarray], list[int]]:
        """ Elimina máscaras j que son claramente contenedoras de i.
        Condición para marcar j como contenedora:
          • areas[j] >= areas[i] * size_ratio_min   (j es bastante más grande)
          • intersection(i,j) / areas[i] > containment_thresh (i casi cabe en j)"""
        n = len(masks)
        if n <= 1:
            return masks, original_indices

        # Aplanamos las máscaras a vectores 1-D para operaciones matriciales
        flat = np.stack([m.ravel() for m in masks])          # (n, H*W)
        areas = flat.sum(axis=1).astype(np.float32)           # (n,)
        # Matriz de intersecciones: inter[i,j] = |mask_i AND mask_j|
        inter = (flat @ flat.T).astype(np.float32)            # (n, n)
        # containment[i,j] = inter[i,j] / areas[i]  →  ¿qué fracción de i está en j?
        containment = inter / (areas[:, None] + 1e-6)         # (n, n)
        # size_ratio[i,j] = areas[j] / areas[i]  →  ¿j es mucho más grande que i?
        size_ratio = areas[None, :] / (areas[:, None] + 1e-6) # (n, n)
        # j es contenedora de i si: j es grande Y contiene casi todo i
        # Excluimos la diagonal (i==j)
        np.fill_diagonal(containment, 0.0)
        np.fill_diagonal(size_ratio, 0.0)

        is_container = (
            (size_ratio >= self.size_ratio_min) &
            (containment >= self.containment_thresh)
        )                                                       # (n, n) bool

        # j se elimina si es contenedora de AL MENOS un i
        to_remove = is_container.any(axis=0)                   # (n,) bool
        kept_masks = [m for m, r in zip(masks, to_remove) if not r]
        kept_indices = [idx for idx, r in zip(original_indices, to_remove) if not r]
        return kept_masks, kept_indices

    def clean(
        self, raw_masks: np.ndarray, img_area: int
    ) -> tuple[list[np.ndarray], list[int]]:
        """ Devuelve (máscaras_limpias_bool, índices_en_raw_masks).
        Los índices permiten recuperar metadatos (cls, conf) del resultado SAM3. """
        masks, indices = self._basic_filter(raw_masks, img_area)
        masks, indices = self._remove_containing_masks(masks, indices)
        return masks, indices


#Nodo ROS principal

class KinovaVisionSAM3:

    def __init__(self):
        rospy.init_node("vision_d415_sam3")
        self.pub = rospy.Publisher("object_centroid", PointStamped, queue_size=10)
        self.TOPIC_RGB    = rospy.get_param("~topics/rgb_topic",         "/d415/color/image_raw")
        self.TOPIC_INFO   = rospy.get_param("~topics/camera_info_topic", "/d415/color/camera_info")
        self.TOPIC_POINTS = rospy.get_param("~topics/points_topic",      "/d415/depth/points")
        #Prompt
        self.objects_to_find = ["a single piece of fruit"]
        #Filtrado de máscaras
        self.mask_filter = MaskFilter(
            min_area=200,
            max_area_ratio=0.15,
            min_compactness=0.10,
            containment_thresh=0.85,
            size_ratio_min=2.5,
        )
        #Control de concurrencia
        self.is_processing = False
        rospy.loginfo("Cargando SAM3...")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(rospy.get_param("~paths/pack", "statemachine"))

        overrides = dict(
            conf=0.2,
            task="segment",
            mode="predict",
            model=package_path + "/weights/sam3.pt",
            iou=0.3,
            half=True,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        #Intrínsecos de cámara
        try:
            info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]
            self.cam_frame   = info.header.frame_id
        except rospy.ROSException:
            rospy.logerr("No se detectó la cámara D415.")
            return
        #Estado compartido
        self.last_cloud: np.ndarray | None = None

        rospy.Subscriber(self.TOPIC_RGB,    Image,       self.rgb_cb,   queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.TOPIC_POINTS, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.loginfo("Nodo SAM3 listo.")

    #Callback de la nube de puntos
    def cloud_cb(self, msg: PointCloud2) -> None:
        puntos = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if puntos:
            self.last_cloud = np.array(puntos, dtype=np.float32)

    def rgb_cb(self, msg: Image) -> None:
        #Descartamos frame si aún procesamos el anterior
        if self.is_processing:
            return
        try:
            self.is_processing = True
            self._process_frame(msg)
        except Exception as exc:
            rospy.logerr(f"Error en rgb_cb: {exc}")
        finally:
            self.is_processing = False

    def _process_frame(self, msg: Image) -> None:
        frame_rgb = ros_numpy.numpify(msg)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w = frame_rgb.shape[:2]
        img_area = img_h * img_w

        #Inferencia SAM3
        self.predictor.set_image(frame_rgb)
        results = self.predictor(text=self.objects_to_find)
        centroids: list[tuple] = []
        for result in results:
            if result.masks is None:
                continue
            raw_masks = result.masks.data.cpu().numpy()  # (N, H_mask, W_mask)
            #Limpieza de máscaras (mantiene índices para obj_name)
            clean_masks, original_indices = self.mask_filter.clean(raw_masks, img_area)
            rospy.loginfo_throttle(2, f"Máscaras: {len(raw_masks)} crudas → {len(clean_masks)} limpias")

            for mask_bool, orig_idx in zip(clean_masks, original_indices):
                mask_uint8 = mask_bool.astype(np.uint8)  # valores 0/1
                mask_h, mask_w = mask_uint8.shape
                # Nombre del objeto conservado a partir del índice original
                if result.boxes is not None and orig_idx < len(result.boxes.cls):
                    cls_id   = int(result.boxes.cls[orig_idx])
                    obj_name = result.names[cls_id]
                else:
                    obj_name = "fruit"
                #Centroide en espacio de máscara → espacio de imagen
                M = cv2.moments(mask_uint8)
                if M["m00"] == 0:
                    continue
                u_mask = M["m10"] / M["m00"]
                v_mask = M["m01"] / M["m00"]
                #Escala al frame original (intrínsecos definidos sobre este)
                u = int(u_mask * img_w / mask_w)
                v = int(v_mask * img_h / mask_h)

                #Profundidad desde nube de puntos
                z_m = 0.0
                if self.last_cloud is not None:
                    z_m = self.get_depth_from_mask(mask_uint8, frame_bgr.shape)
                if z_m == 0.0:
                    rospy.logwarn_throttle(5, "Profundidad inválida, saltando máscara")
                    continue

                #Proyección pinhole → coordenadas 3D
                x_c = (u - self.cx) * z_m / self.fx
                y_c = (v - self.cy) * z_m / self.fy

                # Visualización
                mask_resized = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 2)
                centroids.append((u, v, x_c, y_c, z_m, obj_name))

        #Centroide promedio y publicación
        if centroids:
            u_med = int(np.mean([c[0] for c in centroids]))
            v_med = int(np.mean([c[1] for c in centroids]))
            x_med = float(np.mean([c[2] for c in centroids]))
            y_med = float(np.mean([c[3] for c in centroids]))
            z_med = float(np.mean([c[4] for c in centroids]))
            name  = centroids[0][5]
            cv2.circle(frame_bgr, (u_med, v_med), 8, (0, 0, 255), -1)
            cv2.putText(
                frame_bgr,
                f"{name} ({x_med:.3f}, {y_med:.3f}, {z_med:.3f})",
                (u_med + 10, v_med),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
            self._publish_centroid(x_med, y_med, z_med)
        cv2.imshow("SAM3 Segmentation", frame_bgr)
        cv2.waitKey(1)

    #Profundidad promedio dentro de máscara
    def get_depth_from_mask(self, mask: np.ndarray, frame_shape: tuple) -> float:
        if self.last_cloud is None:
            return 0.0
        mask_h, mask_w = mask.shape
        img_h, img_w   = frame_shape[:2]
        points_3d      = self.last_cloud
        valid     = points_3d[:, 2] > 0
        points_3d = points_3d[valid]
        if len(points_3d) == 0:
            return 0.0
        u_arr = (points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx).astype(np.int32)
        v_arr = (points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy).astype(np.int32)
        rospy.loginfo_throttle(
            2,
            f"DEBUG: u=[{u_arr.min()},{u_arr.max()}] v=[{v_arr.min()},{v_arr.max()}] "
            f"| img w={img_w} h={img_h}",
        )
        in_bounds = (u_arr >= 0) & (u_arr < img_w) & (v_arr >= 0) & (v_arr < img_h)
        u_arr     = u_arr[in_bounds]
        v_arr     = v_arr[in_bounds]
        z_arr     = points_3d[in_bounds, 2]
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

        if len(z_masked) == 0:
            return 0.0

        return float(np.mean(z_masked))

    #Publicación del centroide
    def _publish_centroid(self, x: float, y: float, z: float) -> None:
        msg = PointStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = self.cam_frame
        msg.point.x = x
        msg.point.y = y
        msg.point.z = z
        self.pub.publish(msg)
        rospy.loginfo_throttle(1, f"Publicando: X={x:.3f}  Y={y:.3f}  Z={z:.3f}")

if __name__ == "__main__":
    try:
        KinovaVisionSAM3()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()