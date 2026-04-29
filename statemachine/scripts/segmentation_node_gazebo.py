#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import cv2
import rospkg as rp
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from ultralytics import YOLO, SAM
import sensor_msgs.point_cloud2 as pc2

class KinovaVisionD415:
    def __init__(self):
        rospy.init_node('vision_simulation')
        self.pub = rospy.Publisher('/object_centroid', PointStamped, queue_size=10)

        self.TOPIC_RGB    = "/d415/color/image_raw"
        self.TOPIC_DEPTH  = "/d415/depth/image_raw"
        self.TOPIC_INFO   = "/d415/color/camera_info"
        # NUBE DE PUNTOS
        self.TOPIC_POINTS = "/d415/depth/points"

        rospy.loginfo("Cargando modelos de YOLOv8 y MobileSAM")
        rospack      = rp.RosPack()
        package_path = rospack.get_path('statemachine')
        self.yolo    = YOLO(package_path + '/weights/yolov8s.pt')
        self.sam     = SAM(package_path  + '/weights/sam2_t.pt')

        rospy.loginfo("Sincronizando con cámara Gazebo D415...")
        try:
            info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]
            self.cam_frame   = info.header.frame_id
        except rospy.ROSException:
            rospy.logerr("No se detectó la cámara. Revisa que Gazebo esté corriendo.")
            return

        self.last_depth  = None
        self.last_cloud  = None

        rospy.Subscriber(self.TOPIC_DEPTH,  Image,        self.depth_cb)
        
        # ============================================================
        rospy.Subscriber(self.TOPIC_POINTS, PointCloud2,  self.cloud_cb, queue_size=1)
        rospy.Subscriber(self.TOPIC_RGB,    Image,        self.rgb_cb, queue_size=1, buff_size=2**24)

    def depth_cb(self, msg):
        # Sigue activo como puente para proyectar x_c e y_c
        self.last_depth = ros_numpy.numpify(msg)  # float32, metros (Gazebo)

    # ================================================================
    def cloud_cb(self, msg):
        puntos = list(pc2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True   # descarta puntos sin retorno (NaN)
        ))
        if len(puntos) > 0:
            self.last_cloud = np.array(puntos, dtype=np.float32)  # shape (N, 3)

    # ================================================================
    # MÓDULO DE PROFUNDIDAD — Promedio de Z mediante proyección con máscara
    #
    # Flujo:
    #   1. Reprojectamos cada punto 3D de la nube al plano imagen (u, v)
    #   2. Verificamos que el píxel (u, v) esté dentro de la máscara SAM
    #   3. Promediamos el Z de todos los puntos aceptados → profundidad suavizada
    #
    # Ventajas frente al enfoque anterior (KD-Tree):
    #   - No requiere Open3D ni construcción de KD-Tree por frame
    #   - Opera directamente sobre los puntos reales del objeto segmentado
    #   - Más ligero computacionalmente al filtrar con la máscara binaria
    #
    # MÓDULO ANTERIOR (KD-Tree + K vecinos) — conservado como referencia
    # ================================================================
    # def get_depth_from_cloud(self, u, v, z_seed, k=10):
    #     """Obtiene Z confiable usando KD-Tree sobre la nube 3D."""
    #     if self.last_cloud is None:
    #         return 0.0
    #
    #     x_seed = (u - self.cx) * z_seed / self.fx
    #     y_seed = (v - self.cy) * z_seed / self.fy
    #     seed   = np.array([[x_seed, y_seed, z_seed]])
    #
    #     nube = o3d.geometry.PointCloud()
    #     nube.points = o3d.utility.Vector3dVector(self.last_cloud)
    #     kd_tree = o3d.geometry.KDTreeFlann(nube)
    #
    #     _, idx, _ = kd_tree.search_knn_vector_3d(seed[0], k)
    #
    #     vecinos  = self.last_cloud[idx]        # shape (K, 3)
    #     z_result = float(np.mean(vecinos[:, 2]))  # promedio del eje Z
    #     return z_result
    # ================================================================

    def get_depth_from_mask(self, mask):
    
        # Proyecta la nube 3D al plano imagen y promedia el Z de los puntos
        # que caen dentro de la máscara SAM
        if self.last_cloud is None:
            return 0.0

        mask_h, mask_w = mask.shape
        points_3d = self.last_cloud  # (N, 3)

        # --- Filtrar puntos detrás de la cámara ---
        valid = points_3d[:, 2] > 0
        points_3d = points_3d[valid]

        if len(points_3d) == 0:
            return 0.0

        # --- Proyección vectorizada 3D → 2D ---
        u_arr = (points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx).astype(np.int32)
        v_arr = (points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy).astype(np.int32)

        # --- Conservar solo los que caen dentro de la imagen ---
        in_bounds = (u_arr >= 0) & (u_arr < mask_w) & (v_arr >= 0) & (v_arr < mask_h)
        u_arr     = u_arr[in_bounds]
        v_arr     = v_arr[in_bounds]
        z_arr     = points_3d[in_bounds, 2]

        # --- Filtrar por máscara (multiplicación equivalente) ---
        in_mask = mask[v_arr, u_arr] > 0
        z_masked = z_arr[in_mask]

        if len(z_masked) == 0:
            return 0.0

        return float(np.mean(z_masked))

    def rgb_cb(self, msg):
        if self.last_depth is None:
            return

        frame_rgb = ros_numpy.numpify(msg)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        clases_interes = [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

        input_size     = 320
        img_small      = cv2.resize(frame_bgr, (input_size, input_size))
        h_orig, w_orig = frame_bgr.shape[:2]
        sw, sh         = w_orig / input_size, h_orig / input_size

        results = self.yolo(img_small, verbose=False, conf=0.5, classes=clases_interes)[0]

        for box in results.boxes:
            class_id   = int(box.cls[0])
            class_name = self.yolo.names[class_id]

            b      = box.xyxy[0].cpu().numpy()
            x1, y1 = int(b[0] * sw), int(b[1] * sh)
            x2, y2 = int(b[2] * sw), int(b[3] * sh)
            cx_box = (x1 + x2) / 2
            cy_box = (y1 + y2) / 2

            sam_results = self.sam.predict(
                frame_bgr,
                bboxes=[[x1, y1, x2, y2]],
                points=[[cx_box, cy_box]],
                labels=[1],
                verbose=False,
                imgsz=320
            )[0]

            if sam_results.masks is not None:
                mask = sam_results.masks.data[0].cpu().numpy().astype(np.uint8)
                M    = cv2.moments(mask)
                if M["m00"] == 0:
                    continue

                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])

                # Paso previo: z_seed desde imagen de profundidad (semilla de validación)
                z_seed = self.get_filtered_depth(u, v)

                if 0.2 < z_seed < 1.2:

                    # ---- NUEVO módulo de profundidad ----
                    # Proyectamos la nube con la máscara SAM y promediamos Z
                    if self.last_cloud is not None:
                        z_m = self.get_depth_from_mask(mask)
                        if z_m == 0.0:        # fallback si la proyección falla
                            z_m = z_seed
                    else:
                        z_m = z_seed

                    x_c = (u - self.cx) * z_m / self.fx
                    y_c = (v - self.cy) * z_m / self.fy

                    self.publish_msg(x_c, y_c, z_m)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 2)
                    label     = f"{class_name.upper()} | X:{x_c:.2f} Y:{y_c:.2f} Z:{z_m:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame_bgr, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(frame_bgr, label, (x1, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.circle(frame_bgr, (u, v), 5, (0, 0, 255), -1)

        cv2.imshow("D415 Gazebo - YOLO+SAM", frame_bgr)
        cv2.waitKey(1)

    def get_filtered_depth(self, u, v):
        """Ventana 5x5 — actúa como validación de rango para z_seed."""
        try:
            roi          = self.last_depth[v-2:v+3, u-2:u+3]
            valid_depths = roi[np.isfinite(roi) & (roi > 0)]
            if len(valid_depths) > 0:
                return float(np.median(valid_depths))
            return 0.0
        except IndexError:
            return 0.0

    def publish_msg(self, x, y, z):
        """Publica X, Y, Z del centroide. Z ahora viene del promedio sobre la máscara."""
        target_msg             = PointStamped()
        target_msg.header.stamp    = rospy.Time.now()
        target_msg.header.frame_id = self.cam_frame
        target_msg.point.x = x
        target_msg.point.y = y
        target_msg.point.z = z
        self.pub.publish(target_msg)
        rospy.loginfo_throttle(1, f"Publicando: X={x:.3f}  Y={y:.3f}  Z={z:.3f}")

if __name__ == '__main__':
    try:
        KinovaVisionD415()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
