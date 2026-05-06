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

class KinovaVisionSAM3:
    def __init__(self):
        rospy.init_node('vision_d415_sam3')
        
        # Publicador del centroide
        self.pub = rospy.Publisher('object_centroid', PointStamped, queue_size=10)

        #Tópicos fijos de la RealSense D415
        self.TOPIC_RGB    = rospy.get_param("~topics/rgb_topic", "/d415/color/image_raw")
        self.TOPIC_INFO   = rospy.get_param("~topics/camera_info_topic", "/d415/color/camera_info")
        self.TOPIC_POINTS = rospy.get_param("~topics/points_topic", "/d415/depth/points")
        
        # Flag para evitar que los callbacks se acumulen (Control de Concurrencia)
        self.is_processing = False 
        
        rospy.loginfo("Iniciando nodo Kinova-SAM3...") 
        
        # Obtener ruta del modelo dinámicamente
        rp = rospkg.RosPack()
        try:
            package_path = rp.get_path(rospy.get_param("~paths/pack", 'statemachine'))
            model_path = package_path + "/weights/sam3.pt"
        except Exception as e:
            rospy.logwarn(f"No se encontró el paquete: {e}. Usando ruta local.")
            model_path = "sam3.pt"

        rospy.loginfo(f"Cargando pesos desde: {model_path}")
        
        # Configuración de SAM3 (Promptable Concept Segmentation)
        overrides = dict(conf=0.35, task="segment", mode="predict", model=model_path, half=True,)
        
        # Inicialización del predictor
        try:
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            rospy.logerr(f"Error al cargar SAM3: {e}")
            return

        # Definimos qué conceptos queremos que SAM3 busque por texto
        self.objects_to_find = ["chopped fruit", "mango piece", "apple slice", "apple"]

        # Configuración de Cámara (Intrínsecos)
        try:
            rospy.loginfo("Esperando CameraInfo...")
            info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]
            self.cam_frame = info.header.frame_id 
        except rospy.ROSException:
            rospy.logerr("No se detectó la cámara D415. Revisa los tópicos.")
            return

        # Variable para la nube de puntos
        self.last_cloud = None

        rospy.Subscriber(self.TOPIC_POINTS, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber(self.TOPIC_RGB, Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("Nodo SAM3 listo y procesando...")

    # Callback de nube de puntos
    def cloud_cb(self, msg):
        # Lee los puntos x, y y z, ignorando los valores nulos
        puntos = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        # Si hay puntos validos los guarda en un array de atributo de clase
        if len(puntos) > 0:
            self.last_cloud = np.array(puntos, dtype=np.float32)

    # Funcion para obtener la profundidad promedio dentro de la mascara de SAM
    def get_depth_from_mask(self, mask):
        # Si no hay nube de puntos, retornamos 0.0 como indicador de fallo
        if self.last_cloud is None:
            return 0.0

        mask_h, mask_w = mask.shape # Obtenemos dimensiones de la mascara
        points_3d = self.last_cloud # Obtenemos la nube de puntos actual

        valid = points_3d[:, 2] > 0 # Filtramos puntos con Z > 0
        points_3d = points_3d[valid] # Aplicamos el filtro

        # Si no hay puntos validos retornamos 0.0 a manera de indicador de fallo
        if len(points_3d) == 0:
            return 0.0

        # Proyectamos los puntos 3D a coordenadas de imagen (u, v)
        u_arr = (points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx).astype(np.int32)
        v_arr = (points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy).astype(np.int32)

        # Filtramos los puntos que caen dentro de los límites de la máscara
        in_bounds = (u_arr >= 0) & (u_arr < mask_w) & (v_arr >= 0) & (v_arr < mask_h)
        u_arr = u_arr[in_bounds]
        v_arr = v_arr[in_bounds]
        z_arr = points_3d[in_bounds, 2]

        # Conservamos solo los puntos que caen dentro de la mascara
        in_mask = mask[v_arr, u_arr] > 0
        z_masked = z_arr[in_mask]

        # En caso de no haber puntos retornamos fallo
        if len(z_masked) == 0:
            return 0.0
        # Retornamos la profundidad promedio de los puntos dentro de la mascara
        return float(np.mean(z_masked))

    def rgb_cb(self, msg):
        if self.is_processing:
            return

        try:
            self.is_processing = True
            
            # 1. Preparación de imagen
            frame_rgb = ros_numpy.numpify(msg)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 2. Inferencia SAM3
            self.predictor.set_image(frame_rgb)
            results = self.predictor(text=self.objects_to_find)

            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    
                    for i in range(len(masks)):
                        mask_uint8 = (masks[i] * 255).astype(np.uint8)
                        
                        if result.boxes is not None and len(result.boxes.cls) > i:
                            cls_id = int(result.boxes.cls[i])
                            obj_name = result.names[cls_id]
                        else:
                            obj_name = "fruit_piece"

                        # 3. Cálculo de Centroide
                        M = cv2.moments(mask_uint8)
                        if M["m00"] < 50:
                            continue
                        
                        u = int(M["m10"] / M["m00"])
                        v = int(M["m01"] / M["m00"])

                        # 4. Obtener profundidad desde la nube de puntos
                        z_m = 0.0
                        if self.last_cloud is not None:
                            z_m = self.get_depth_from_mask(mask_uint8)

                        if z_m == 0.0:
                            rospy.logwarn_throttle(5, "Sin profundidad válida, saltando objeto")
                            continue

                        # 5. Proyección a coordenadas 3D
                        x_c = (u - self.cx) * z_m / self.fx
                        y_c = (v - self.cy) * z_m / self.fy

                        self.publish_msg(x_c, y_c, z_m)

                        # 6. Visualización
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 2)
                        cv2.circle(frame_bgr, (u, v), 5, (0, 0, 255), -1)
                        label = f"{obj_name} | X:{x_c:.2f} Y:{y_c:.2f} Z:{z_m:.2f}"
                        cv2.putText(frame_bgr, label, (u, v - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Kairós SAM3 - Monitor", frame_bgr)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error en el ciclo de visión: {e}")
        
        finally:
            self.is_processing = False

    def publish_msg(self, x, y, z):
        """Publica el punto 3D en el tópico object_centroid."""
        target_msg = PointStamped()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.header.frame_id = self.cam_frame
        target_msg.point.x = x
        target_msg.point.y = y
        target_msg.point.z = z
        self.pub.publish(target_msg)
        rospy.loginfo_throttle(1, f"Publicando: X={x:.3f}  Y={y:.3f}  Z={z:.3f}")

if __name__ == '__main__':
    try:
        KinovaVisionSAM3()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()