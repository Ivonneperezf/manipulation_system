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
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import std_msgs.msg
import struct

"""Clase para filtrar la mascara y obtener mejor precision en la seleccion de cubos"""
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
    # Constructor con parámetros ajustados para cubitos individuales
    def __init__(self, min_area=100, max_area_ratio=0.06, min_compactness=0.10, containment_thresh=0.92, size_ratio_min=3.0):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_compactness = min_compactness
        self.containment_thresh = containment_thresh
        self.size_ratio_min = size_ratio_min

    # Filtros básicos: área, proporción de área respecto a bbox, y relación con el área total de la imagen
    def _basic_filter(self, masks, img_area):
        kept_masks, kept_indices = [], []
        # Recorremos cada máscara y aplicamos los filtros 
        for idx, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            mask_area = mask_bool.sum()
            # Filtro de area mínima
            if mask_area < self.min_area:
                continue
            # Filtro de área máxima relativa a la imagen
            if mask_area / img_area > self.max_area_ratio:
                continue
            # Filtro de compactness: define la bounding box y calcula su área, luego compara con el área de la máscara
            ys, xs = np.where(mask_bool)
            bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            if mask_area / (bbox_area + 1e-6) < self.min_compactness:
                continue

            kept_masks.append(mask_bool)
            kept_indices.append(idx)
        # Devolvemos solo las máscaras que pasaron los filtros básicos y sus índices originales
        return kept_masks, kept_indices

    # Filtro de contención: elimina máscaras que contienen a otras con alta proporción y son suficientemente más grandes
    def _remove_containing_masks(self, masks, original_indices):
        # Si solo hay una o ninguna máscara, no hay contención posible, devolvemos tal cual
        n = len(masks)
        if n <= 1:
            return masks, original_indices

        # Para cada par de máscaras, calculamos el área de intersección y la proporción de contención, 
        # filtrando aquellas que contienen a otras con alta proporción y son suficientemente más grandes
        flat = np.stack([m.ravel() for m in masks])
        areas = flat.sum(axis=1).astype(np.float32)
        inter = (flat @ flat.T).astype(np.float32)
        containment = inter / (areas[:, None] + 1e-6)
        size_ratio = areas[None, :] / (areas[:, None] + 1e-6)
        np.fill_diagonal(containment, 0.0)
        np.fill_diagonal(size_ratio, 0.0)

        is_container = (size_ratio >= self.size_ratio_min) & (containment >= self.containment_thresh)
        to_remove = is_container.any(axis=0)

        # Devolvemos solo las máscaras que no fueron marcadas para eliminación y sus índices originales correspondientes
        return (
            [m for m, r in zip(masks, to_remove) if not r],
            [idx for idx, r in zip(original_indices, to_remove) if not r],
        )

    # Funcion principal que aplica los filtros basicos y luego el filtro de contención, 
    # devolviendo las máscaras limpias y sus índices originales
    def clean(self, raw_masks, img_area):
        masks, indices = self._basic_filter(raw_masks, img_area)
        return self._remove_containing_masks(masks, indices)

"""Clase principal del nodo ROS que integra SAM3 para segmentar cubitos de fruta individuales,"""
class KinovaVisionSAM3:
    def __init__(self):
        # Inica el nodo
        rospy.init_node("vision_d415_sam3")
        # Publicadores
        self.pub = rospy.Publisher("object_centroid", PointStamped, queue_size=10)
        self.pc_pub = rospy.Publisher('/object_pointcloud', PointCloud2, queue_size=10)
        #Configuración de tópicos con valores por defecto
        self.TOPIC_RGB    = rospy.get_param("~topics/rgb_topic",         "/d415/color/image_raw")
        self.TOPIC_INFO   = rospy.get_param("~topics/camera_info_topic", "/d415/color/camera_info")
        self.TOPIC_POINTS = rospy.get_param("~topics/points_topic",      "/d415/depth/points")
        # Definición de prompts para SAM3
        self.prompts = ["a small cube of fruit"]

        # Filtros ajustados para cubitos individuales de la clase definida anteriormente
        self.mask_filter = MaskFilter(
            min_area=100,
            max_area_ratio=0.06,
            min_compactness=0.10,
            containment_thresh=0.92,
            size_ratio_min=3.0,
        )

        # Parámetros de selección de cubitos de frutas
        # Fracción del grupo usada como candidatos centrales
        self.CENTER_FRACTION = 0.30
        # Si hay <= FEW_THRESHOLD cubitos, ignorar centro y usar solo menor Z
        self.FEW_THRESHOLD = 4
        # Flag para evitar procesamiento concurrente de frames
        self.is_processing = False

        # Cargar modelo SAM3 con configuración específica para segmentación de cubitos
        rospy.loginfo("Cargando SAM3...")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(rospy.get_param("~paths/pack", "statemachine"))
        # Carga el modelo con parámetros ajustados para segmentación de cubitos individuales
        overrides = dict(
            conf=0.2,
            task="segment",
            mode="predict",
            model=os.path.join(package_path, "weights", "sam3.pt"),
            iou=0.3,
            half=True,
        )
        # Cragamos parametros con el preset de segmentación y el modelo específico para cubitos
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        # Obtener parámetros intrínsecos de la cámara desde el tópico de CameraInfo
        try:
            info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]
            self.cam_frame   = info.header.frame_id
        except rospy.ROSException:
            rospy.logerr("No se detectó la cámara.")
            return

        # Variable para almacenar la última nube de puntos recibida
        self.last_cloud: Optional[np.ndarray] = None
        # Crear un directorio temporal para almacenar imágenes normalizadas para SAM3
        self._tmp_dir = tempfile.mkdtemp(prefix="sam3_ros_")
        self._tmp_frame_path = os.path.join(self._tmp_dir, "current_frame.jpg")

        # Suscriptores para RGB y nube de puntos
        rospy.Subscriber(self.TOPIC_RGB,    Image,        self.rgb_cb,   queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.TOPIC_POINTS, PointCloud2,  self.cloud_cb, queue_size=1)
        rospy.loginfo("Nodo SAM3 Listo.")

    # Callback para recibir la nube de puntos y almacenarla en self.last_cloud
    def cloud_cb(self, msg: PointCloud2) -> None:
        puntos = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if puntos:
            self.last_cloud = np.array(puntos, dtype=np.float32)

    # Callback para recibir la imagen RGB y procesarla
    def rgb_cb(self, msg: Image) -> None:
        # Si ya estamos procesando un frame, ignoramos este nuevo mensaje para evitar solapamientos
        if self.is_processing:
            return
        # Filtramos errores e indicamos el procesamiento en caso de que no existe un procesamiento concurrente
        try:
            self.is_processing = True
            self._process_frame(msg)
        except Exception as exc:
            rospy.logerr(f"Error en rgb_cb: {exc}")
        finally:
            self.is_processing = False

    # Función para normalizar la imagen usando CLAHE en el canal L de LAB, mejorando el contraste local entre cubitos
    def _normalize_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        CLAHE en el canal L de LAB: mejora el contraste local entre cubitos
        sin importar si la luz es fuerte, tenue o lateral.
        El frame_bgr original NO se toca; se usa solo para visualización.
        """
        # Convertir a LAB y aplicar CLAHE solo al canal L
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        # Convertir de vuelta a BGR para SAM3, pero el frame_bgr original se mantiene sin cambios para visualización
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _select_target(self, all_clean_masks: List[np.ndarray], z_values: List[float], img_w: int, img_h: int) -> int:
        """
        Criterio de selección adaptativo según cuántos cubitos hay
        MUCHOS (> FEW_THRESHOLD):
          1. Calcular centroide de cada máscara en píxeles
          2. Calcular centro de masa del grupo
          3. Tomar el top CENTER_FRACTION más cercanos al centro
          4. De esos candidatos, elegir el de menor Z
          Evita orillas cuando hay masa central disponible

        POCOS (<= FEW_THRESHOLD):
          - Ignorar criterio de centro (ya no hay "grupo central" real)
          - Tomar directamente el de menor Z con profundidad válida
          Al final de la tarea cualquier cubito accesible es válido

        Casos extremos cubiertos:
          - 1 cubito          -> se toma directamente
          - Cubitos separados -> top 30% sin radio fijo, siempre hay candidatos
          - Pocos al final    -> menor Z sin restricción de centro
          - Todos sin Z valid -> fallback al índice 0
        """
        n = len(all_clean_masks)
        centroids_uv = []
        # Si solo hay una mascara
        if n == 1:
            u, v = centroids_uv[0]
            return 0, int(u), int(v)
        # Centroides en píxeles
        centroids_uv = []
        for mask_bool in all_clean_masks:
            mask_uint8 = mask_bool.astype(np.uint8)
            mask_h, mask_w = mask_uint8.shape
            # Extraemos los momentos de la mascara para calcular su centroide en coordenadas de imagen
            M = cv2.moments(mask_uint8)
            # Si el área es cero, asignamos el centro de la imagen como fallback para evitar errores de división
            if M["m00"] == 0:
                centroids_uv.append((img_w / 2.0, img_h / 2.0))
                continue
            u = M["m10"] / M["m00"] * img_w / mask_w
            v = M["m01"] / M["m00"] * img_h / mask_h
            centroids_uv.append((u, v))
        # Calculamos el centro del grupo de cubitos y la distancia de cada centroide a ese centro para la selección adaptativa
        centroids_arr = np.array(centroids_uv)      # (N, 2)
        group_center  = centroids_arr.mean(axis=0)  # (u_mean, v_mean)
        dists         = np.linalg.norm(centroids_arr - group_center, axis=1)

        # Si hay pocos cubitos solo tomamos el menor Z
        if n <= self.FEW_THRESHOLD:
            rospy.loginfo(f"Modo FINAL ({n} cubitos): seleccionando por menor Z.")
            valid = [(i, z_values[i]) for i in range(n) if z_values[i] < float("inf")]
            if not valid:
                u, v = centroids_uv[0]
                return 0, int(u), int(v)
            idx = min(valid, key=lambda x: x[1])[0]
            u, v = centroids_uv[idx]
            return idx, int(u), int(v)

        # Si hay muchos cubitos buscamos el top 30% centrados + menor Z
        rospy.loginfo(f"Modo NORMAL ({n} cubitos): seleccionando por centro + Z.")
        n_candidatos = max(1, int(n * self.CENTER_FRACTION))
        candidatos   = np.argsort(dists)[:n_candidatos].tolist()
        z_candidatos = [z_values[i] for i in candidatos]
        # De los candidatos centrados, elegimos el de menor Z (con fallback al primero si todos son inf)
        idx = candidatos[int(np.argmin(z_candidatos))]
        u, v = centroids_uv[idx]
        return idx, int(u), int(v)

    # Funcion para procesar cada frame RGB
    def _process_frame(self, msg: Image) -> None:
        # Convertir el mensaje ROS Image a formato OpenCV (BGR) y obtener dimensiones
        frame_rgb = ros_numpy.numpify(msg)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w = frame_rgb.shape[:2]
        img_area = img_h * img_w

        # SAM3 recibe la imagen normalizada en un archivo remporal, no el frame original
        frame_bgr_norm = self._normalize_image(frame_bgr)
        cv2.imwrite(self._tmp_frame_path, frame_bgr_norm)
        # Ejecutar SAM3 para obtener máscaras y resultados de detección
        self.predictor.set_image(self._tmp_frame_path)
        results = self.predictor(text=self.prompts)

        all_clean_masks: List[np.ndarray] = []
        all_original_indices: List[int]   = []
        target_result = None

        # Recopilar y filtrar todas las máscaras
        for result in results:
            # Si hay mascaras, procesarlas
            if result.masks is None:
                continue
            
            # Convertir las mascaras sin procesar a numpy booleanas para el filtrado
            raw_masks = result.masks.data.cpu().numpy()

            # Aplicamos los filtros básicos y de contención para obtener las máscaras limpias y sus índices originales
            after_basic, after_idx = self.mask_filter._basic_filter(raw_masks, img_area)
            after_clean, after_clean_idx = self.mask_filter._remove_containing_masks(after_basic, after_idx)
            # Logueamos la informacion de las mascaras por cada filtro
            rospy.loginfo(
                f"Máscaras -> crudas: {len(raw_masks)} | "
                f"tras basic: {len(after_basic)} | "
                f"tras containment: {len(after_clean)}"
            )

            # Agregamos las máscaras limpias y sus índices originales a la lista general, y guardamos el resultado asociado al target
            for m, idx in zip(after_clean, after_clean_idx):
                all_clean_masks.append(m)
                all_original_indices.append(idx)
                target_result = result

        # Verificar que hay máscaras
        if not all_clean_masks:
            rospy.logwarn_throttle(3, "No se detectaron cubitos en este frame.")
            cv2.imshow("SAM3 Segmentation", frame_bgr)
            cv2.waitKey(1)
            return
        
        # Logueamos la cantidad de máscaras candidatas después de todo el proceso de filtrado
        rospy.loginfo(f"Cubitos candidatos tras filtro: {len(all_clean_masks)}")

        #Calcular Z de cada máscara
        z_values: List[float] = []
        z_masked_list = []
        u_mask_list = []
        v_mask_list = []
        # Recorremos las mascaras booleanas limpias y filtradas
        for mask_bool in all_clean_masks:
            mask_uint8 = mask_bool.astype(np.uint8)
            # Obtenemos la profundidad por mascara 
            z_masked_func, u_valid_func, v_valid_func = self.get_depth_from_mask(mask_uint8, frame_bgr.shape)
            # Guardamos los valores de profundidad y las coordenadas 2D válidas dentro de la máscara para cada mascara
            z_masked_list.append(z_masked_func)
            u_mask_list.append(u_valid_func)
            v_mask_list.append(v_valid_func)
            # Calculamos la profundidad representativa de la mascara usando la media de los valores de z
            z = float(np.mean(z_masked_func)) if len(z_masked_func) > 0 else 0.0
            z_values.append(z if z > 0.0 else float("inf"))

        # Selección adaptativa
        target_idx, u, v = self._select_target(all_clean_masks, z_values, img_w, img_h)

        # Verificar que el cubito seleccionado tenga una profundidad válida antes de continuar con la extracción de datos y visualización
        if z_values[target_idx] == float("inf"):
            rospy.logwarn("El cubito objetivo no tiene profundidad válida.")
            cv2.imshow("SAM3 Segmentation", frame_bgr)
            cv2.waitKey(1)
            return

        #Extraer datos del cubito ganador
        mask_bool      = all_clean_masks[target_idx]
        orig_idx       = all_original_indices[target_idx]
        z_m            = z_values[target_idx]
        z_masked   = z_masked_list[target_idx]
        u_mask_arr = u_mask_list[target_idx]
        v_mask_arr = v_mask_list[target_idx]
        mask_uint8     = mask_bool.astype(np.uint8)
        mask_h, mask_w = mask_uint8.shape

        # Nombre de clase si está disponible, o "fruit" como fallback genérico
        if (target_result and target_result.boxes is not None and orig_idx < len(target_result.boxes.cls)):
            cls_id   = int(target_result.boxes.cls[orig_idx])
            obj_name = target_result.names[cls_id]
        else:
            obj_name = "fruit"

        # Coordenadas 3D en frame de cámara
        x_c = (u - self.cx) * z_m / self.fx
        y_c = (v - self.cy) * z_m / self.fy

        # Visualización
        mask_resized = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        contours, _  = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Color del contorno según modo
        # verde = normal (aun hay cubitos)
        # amarillo = final (casi no hay cubitos)
        color = (0, 255, 255) if len(all_clean_masks) <= self.FEW_THRESHOLD else (0, 255, 0)
        mode_label = "FINAL" if len(all_clean_masks) <= self.FEW_THRESHOLD else "TARGET"

        cv2.drawContours(frame_bgr, contours, -1, color, 3)
        cv2.circle(frame_bgr, (u, v), 8, (0, 0, 255), -1)
        cv2.putText(
            frame_bgr,
            f"{mode_label}: {obj_name} ({x_c:.3f}, {y_c:.3f}, {z_m:.3f})",
            (u + 10, v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

        # Publicar centroide y mostrar
        self._publish_centroid(x_c, y_c, z_m)

        # Publicacion de nube de puntos de la mascara segmentada seleccionada en color magenta
        r, g, b = 255, 0, 255
        rgb_packed = struct.unpack('f', struct.pack('BBBB', b, g, r, 0))[0]
        if len(z_masked) > 0:
            pts = []
            for u_p, v_p, z_p in zip(u_mask_arr, v_mask_arr, z_masked):
                X = (u_p * img_w / mask_w - self.cx) * z_p / self.fx
                Y = (v_p * img_h / mask_h - self.cy) * z_p / self.fy
                pts.append([X, Y, z_p, rgb_packed])

            fields = [
                PointField('x',   0,  PointField.FLOAT32, 1),
                PointField('y',   4,  PointField.FLOAT32, 1),
                PointField('z',   8,  PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.FLOAT32, 1),
            ]
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.cam_frame
            cloud_msg = pc2.create_cloud(header, fields, pts)
            self.pc_pub.publish(cloud_msg)
        cv2.imshow("SAM3 Segmentation", frame_bgr)
        cv2.waitKey(1)

    # Funcion para obtener la profundidad de cada mascara filtrada
    def get_depth_from_mask(self, mask: np.ndarray, frame_shape: Tuple[int, int, int]) -> float:
        # Si no hay nube de puntos, devolvemos arreglos vacíos
        if self.last_cloud is None:
            return np.array([]), np.array([]), np.array([])
        # Obtenemos las dimensiones de la mascara
        mask_h, mask_w = mask.shape
        # Obtenemos las dimensiones de la imagen sin procesar
        img_h, img_w   = frame_shape[:2]
        # Filtramos la nube de puntos para obtener solo los valores positivos de z
        points_3d = self.last_cloud
        valid = points_3d[:, 2] > 0
        points_3d = points_3d[valid]
        rospy.loginfo_throttle(2, f"DEBUG: Puntos 3D válidos: {len(points_3d)}")
        # Si no hay puntos 3D válidos, devolvemos arreglos vacíos
        if len(points_3d) == 0:
            return np.array([]), np.array([]), np.array([])
        # Obtenemos las coordenadas 2D proyectadas de los puntos 3D usando los parámetros intrínsecos de la cámara
        u_arr = (points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx).astype(np.int32)
        v_arr = (points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy).astype(np.int32)
        # Filtramos los puntos proyectados que caen dentro de los límites de la imagen de la camara
        in_bounds = (u_arr >= 0) & (u_arr < img_w) & (v_arr >= 0) & (v_arr < img_h)
        u_arr = u_arr[in_bounds]
        v_arr = v_arr[in_bounds]
        z_arr = points_3d[in_bounds, 2]
        # Mapeamos las coordenadas de la imagen a las coordenadas de la máscara y filtramos los puntos que caen dentro de la máscara
        u_mask = (u_arr * mask_w / img_w).astype(np.int32)
        v_mask = (v_arr * mask_h / img_h).astype(np.int32)
        in_bounds2 = (u_mask >= 0) & (u_mask < mask_w) & (v_mask >= 0) & (v_mask < mask_h)
        u_mask = u_mask[in_bounds2]
        v_mask = v_mask[in_bounds2]
        z_arr  = z_arr[in_bounds2]
        # Filtramos los puntos que caen dentro de la máscara binaria
        in_mask = mask[v_mask, u_mask] > 0
        z_masked = z_arr[in_mask]
        rospy.loginfo_throttle(2, f"DEBUG: Puntos 3D en la máscara: {len(z_masked)}")
        # Si no hay puntos 3D dentro de la máscara, devolvemos arreglos vacíos
        if len(z_masked) == 0:
            return np.array([]), np.array([]), np.array([])
        # Devolvemos los valores de profundidad y las coordenadas 2D válidas dentro de la máscara
        u_valid = u_mask[in_mask]
        v_valid = v_mask[in_mask]
        return z_masked, u_valid, v_valid

    # Función para publicar el centroide seleccionado como un mensaje PointStamped en ROS
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