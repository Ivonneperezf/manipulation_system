#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import cv2
import rospkg as rp
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from ultralytics.models.sam import SAM3SemanticPredictor
import sensor_msgs.point_cloud2 as pc2

class KinovaVisionSAM3:

    def __init__(self):
        rospy.init_node('vision_d415_sam3')
        # Publicador de punto centroide
        self.pub = rospy.Publisher('object_centroid', PointStamped, queue_size=10)

        # Carga de topicos
        self.TOPIC_RGB    = rospy.get_param("~topics/rgb_topic", "/d415/color/image_raw")
        self.TOPIC_INFO   = rospy.get_param("~topics/camera_info_topic", "/d415/color/camera_info")
        self.TOPIC_POINTS = rospy.get_param("~topics/points_topic", "/d415/depth/points")

        # Carga del modelo SAM3
        rospy.loginfo("Cargando SAM3...")
        rospack = rp.RosPack()
        package_path = rospack.get_path(rospy.get_param("~paths/pack", 'statemachine'))

        # Conguracion de prompts y pesos de SAM3
        overrides = dict(conf=0.5,task="segment",mode="predict",model= package_path + "/weights/sam3.pt",half=True,)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
       
        # Sincronizacion de camara
        try:
            info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]
            self.cam_frame = info.header.frame_id 
        except rospy.ROSException:
            rospy.logerr("No se detectó la cámara")
            return
        
        # Suscribers 
        rospy.Subscriber(self.TOPIC_RGB, Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.TOPIC_POINTS, PointCloud2, self.cloud_cb, queue_size=1)

        # Variables
        self.objets_segment= ["object in bowl"]
        self.last_cloud = None
    
    # Callback de nube de puntos
    def cloud_cb(self, msg):
        # Lee los puntos x, y y z, ignorando los valores nulos
        puntos = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        # Si hay puntos validos los guarda en un array de atributo de clase
        if len(puntos) > 0:
            self.last_cloud = np.array(puntos, dtype=np.float32)
    
    def rgb_cb(self, msg):
        # pasamos el array de ROS a OpenCV
        frame_rgb = ros_numpy.numpify(msg)
        # Cargamos la imagen en el predictor (esto genera los embeddings)
        self.predictor.set_image(frame_rgb)
        # Realizamos la inferencia por texto
        # SAM3 devuelve una lista de resultados, uno por cada prompt
        results = self.predictor(text=self.objets_segment)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Lista para almacenar los centroides encontrados
        centroids = []
        # Recorremos los resultados de cada prompt
        for result in results:
            # Si el resultado tiene máscaras, procesamos cada una
            if result.masks is not None:
                # Iterar sobre las máscaras encontradas para este prompt
                for i, mask_data in enumerate(result.masks.data):
                    # Convertir máscara a numpy
                    mask = mask_data.cpu().numpy().astype(np.uint8)
                    # Calcular nombre del objeto (basado en el índice del resultado)
                    obj_name = result.names[int(result.boxes.cls[i])] if result.boxes is not None else "objeto"
                    # Obtén la resolución original del frame
                    h_orig, w_orig = frame_rgb.shape[:2]
                    # Resolución de la máscara
                    h_mask, w_mask = mask.shape[:2]
                    # Calcula centroide
                    M = cv2.moments(mask)
                    if M["m00"] == 0: continue
                    u_mask = int(M["m10"] / M["m00"])
                    v_mask = int(M["m01"] / M["m00"])
                    # Escala al espacio original
                    u = int(u_mask * w_orig / w_mask)
                    v = int(v_mask * h_orig / h_mask)

                    z_m = 0.0
                    # Obtenemos la profundidad promedio dentro de la mascara usando la nube de puntos
                    # Pasamos el shape del frame original para escalar correctamente la proyeccion
                    if self.last_cloud is not None:
                        z_m = self.get_depth_from_mask(mask, frame_bgr.shape)
                    if z_m == 0.0:
                        rospy.logwarn_throttle(5, "Proyección de nube fallida, usando Z del centroide como fallback")
                        continue
                    # Obtenemos coordenadas 3D del centroide usando la profundidad promedio de la mascara
                    x_c = (u - self.cx) * z_m / self.fx
                    y_c = (v - self.cy) * z_m / self.fy

                    # Escalar máscara a resolución original y dibujar contorno
                    mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 2)
                    centroids.append((u, v, x_c, y_c, z_m, obj_name))
        # Visualizamos el centroide promedio de los objetos encontrados (si hay varios) y el nombre del objeto
        if centroids:
            
            u_medio = int(np.mean([c[0] for c in centroids]))
            v_medio = int(np.mean([c[1] for c in centroids]))

            x_medio = np.mean([c[2] for c in centroids])
            y_medio = np.mean([c[3] for c in centroids])
            z_medio = np.mean([c[4] for c in centroids])

            cv2.circle(frame_bgr, (u_medio, v_medio), 8, (0, 0, 255), -1)
            cv2.putText(frame_bgr, f"apple ({x_medio:.3f}, {y_medio:.3f}, {z_medio:.3f})", (u_medio + 10, v_medio),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            self.publish_msg(x_medio, y_medio, z_medio)
        cv2.imshow("SAM3 Segmentation", frame_bgr)
        cv2.waitKey(1)

    # Funcion para obtener la profundidad promedio dentro de la mascara de SAM
    def get_depth_from_mask(self, mask, frame_shape):
        # Si no hay nube de puntos, retornamos 0.0 como indicador de fallo
        if self.last_cloud is None:
            return 0.0

        mask_h, mask_w = mask.shape  # Obtenemos dimensiones de la mascara
        img_h, img_w   = frame_shape[:2]  # Obtenemos dimensiones del frame original
        points_3d = self.last_cloud  # Obtenemos la nube de puntos actual


        valid = points_3d[:, 2] > 0  # Filtramos puntos con Z > 0
        points_3d = points_3d[valid]  # Aplicamos el filtro

        # Si no hay puntos validos retornamos 0.0 a manera de indicador de fallo
        if len(points_3d) == 0:
            return 0.0

        # Proyectamos los puntos 3D a coordenadas de imagen usando los intrinsecos del frame original
        u_arr = (points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx).astype(np.int32)
        v_arr = (points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy).astype(np.int32)
        rospy.loginfo_throttle(2, f"DEBUG: u rango=[{u_arr.min()},{u_arr.max()}] v rango=[{v_arr.min()},{v_arr.max()}] | img limites w={img_w} h={img_h}")

        # Filtramos los puntos que caen dentro de los límites del frame original
        in_bounds = (u_arr >= 0) & (u_arr < img_w) & (v_arr >= 0) & (v_arr < img_h)
        u_arr = u_arr[in_bounds]
        v_arr = v_arr[in_bounds]
        z_arr = points_3d[in_bounds, 2]

        # Escalamos las coordenadas del frame original al tamaño de la mascara de SAM
        u_mask = (u_arr * mask_w / img_w).astype(np.int32)
        v_mask = (v_arr * mask_h / img_h).astype(np.int32)

        # Filtramos nuevamente para asegurar que los indices escalados esten dentro de la mascara
        in_bounds2 = (u_mask >= 0) & (u_mask < mask_w) & (v_mask >= 0) & (v_mask < mask_h)
        u_mask = u_mask[in_bounds2]
        v_mask = v_mask[in_bounds2]
        z_arr  = z_arr[in_bounds2]

        # Conservamos solo los puntos que caen dentro de la mascara
        in_mask = mask[v_mask, u_mask] > 0
        z_masked = z_arr[in_mask]

        # En caso de no haber puntos retornamos fallo
        if len(z_masked) == 0:
            return 0.0
        # Retornamos la profundidad promedio de los puntos dentro de la mascara
        return float(np.mean(z_masked))
    
    # Funcion para publicar el mensaje con las coordenadas del centroide
    def publish_msg(self, x, y, z):
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


# #!/usr/bin/env python3
# import rospy
# import ros_numpy
# import numpy as np
# import cv2
# import rospkg
# from sensor_msgs.msg import Image, CameraInfo, PointCloud2
# from geometry_msgs.msg import PointStamped
# from ultralytics.models.sam import SAM3SemanticPredictor
# import sensor_msgs.point_cloud2 as pc2

# class KinovaVisionSAM3:
#     def __init__(self):
#         rospy.init_node('vision_d415_sam3')
        
#         # Publicador del centroide
#         self.pub = rospy.Publisher('object_centroid', PointStamped, queue_size=10)

#         # Tópicos fijos de la RealSense D415
#         self.TOPIC_RGB    = rospy.get_param("~topics/rgb_topic", "/d415/color/image_raw")
#         self.TOPIC_INFO   = rospy.get_param("~topics/camera_info_topic", "/d415/color/camera_info")
#         self.TOPIC_POINTS = rospy.get_param("~topics/points_topic", "/d415/depth/points")
        
#         # Flag para evitar que los callbacks se acumulen (Control de Concurrencia)
#         self.is_processing = False 
        
#         rospy.loginfo("Iniciando nodo Kinova-SAM3...") 
        
#         # Obtener ruta del modelo dinámicamente
#         rp = rospkg.RosPack()
#         try:
#             package_path = rp.get_path(rospy.get_param("~paths/pack", 'statemachine'))
#             model_path = package_path + "/weights/sam3.pt"
#         except Exception as e:
#             rospy.logwarn(f"No se encontró el paquete: {e}. Usando ruta local.")
#             model_path = "sam3.pt"

#         rospy.loginfo(f"Cargando pesos desde: {model_path}")
        
#         # Configuración de SAM3 (Promptable Concept Segmentation)
#         overrides = dict(conf=0.35, task="segment", mode="predict", model=model_path, half=True,)        
#         # Inicialización del predictor
#         try:
#             self.predictor = SAM3SemanticPredictor(overrides=overrides)
#             # Deshabilitamos guardado y visualizacion automatica de Ultralytics
#             self.predictor.args.save = False
#             self.predictor.args.show = False
#         except Exception as e:
#             rospy.logerr(f"Error al cargar SAM3: {e}")
#             return

#         # Definimos qué conceptos queremos que SAM3 busque por texto
#         self.objects_to_find = ["chopped fruit", "mango piece", "apple slice", "apple"]

#         # Configuración de Cámara (Intrínsecos)
#         try:
#             rospy.loginfo("Esperando CameraInfo...")
#             info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
#             self.fx, self.fy = info.K[0], info.K[4]
#             self.cx, self.cy = info.K[2], info.K[5]
#             self.cam_frame = info.header.frame_id 
#         except rospy.ROSException:
#             rospy.logerr("No se detectó la cámara D415. Revisa los tópicos.")
#             return

#         # Variable para la nube de puntos
#         self.last_cloud = None

#         rospy.Subscriber(self.TOPIC_POINTS, PointCloud2, self.cloud_cb, queue_size=1)
#         rospy.Subscriber(self.TOPIC_RGB, Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        
#         rospy.loginfo("Nodo SAM3 listo y procesando...")

#     # Callback de nube de puntos
#     def cloud_cb(self, msg):
#         # Lee los puntos x, y y z, ignorando los valores nulos
#         puntos = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
#         # Si hay puntos validos los guarda en un array de atributo de clase
#         if len(puntos) > 0:
#             self.last_cloud = np.array(puntos, dtype=np.float32)

#     # Funcion para obtener la profundidad promedio dentro de la mascara de SAM
#     def get_depth_from_mask(self, mask, frame_shape):
#         # Si no hay nube de puntos, retornamos 0.0 como indicador de fallo
#         if self.last_cloud is None:
#             return 0.0

#         mask_h, mask_w = mask.shape  # Obtenemos dimensiones de la mascara
#         img_h, img_w   = frame_shape[:2]  # Obtenemos dimensiones del frame original
#         points_3d = self.last_cloud  # Obtenemos la nube de puntos actual

#         valid = points_3d[:, 2] > 0  # Filtramos puntos con Z > 0
#         points_3d = points_3d[valid]  # Aplicamos el filtro

#         # Si no hay puntos validos retornamos 0.0 a manera de indicador de fallo
#         if len(points_3d) == 0:
#             return 0.0

#         # Proyectamos los puntos 3D a coordenadas de imagen usando los intrinsecos del frame original
#         u_arr = (points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx).astype(np.int32)
#         v_arr = (points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy).astype(np.int32)

#         # Filtramos los puntos que caen dentro de los límites del frame original
#         in_bounds = (u_arr >= 0) & (u_arr < img_w) & (v_arr >= 0) & (v_arr < img_h)
#         u_arr = u_arr[in_bounds]
#         v_arr = v_arr[in_bounds]
#         z_arr = points_3d[in_bounds, 2]

#         # Escalamos las coordenadas del frame original al tamaño de la mascara de SAM
#         u_mask = (u_arr * mask_w / img_w).astype(np.int32)
#         v_mask = (v_arr * mask_h / img_h).astype(np.int32)

#         # Filtramos nuevamente para asegurar que los indices escalados esten dentro de la mascara
#         in_bounds2 = (u_mask >= 0) & (u_mask < mask_w) & (v_mask >= 0) & (v_mask < mask_h)
#         u_mask = u_mask[in_bounds2]
#         v_mask = v_mask[in_bounds2]
#         z_arr  = z_arr[in_bounds2]

#         # Conservamos solo los puntos que caen dentro de la mascara
#         in_mask = mask[v_mask, u_mask] > 0
#         z_masked = z_arr[in_mask]

#         # En caso de no haber puntos retornamos fallo
#         if len(z_masked) == 0:
#             return 0.0
#         # Retornamos la profundidad promedio de los puntos dentro de la mascara
#         return float(np.mean(z_masked))

#     def rgb_cb(self, msg):
#         if self.is_processing:
#             return

#         try:
#             self.is_processing = True
            
#             # 1. Preparación de imagen
#             frame_rgb = ros_numpy.numpify(msg)
#             frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

#             # 2. Inferencia SAM3 - deshabilitamos guardado y visualizacion de Ultralytics
#             self.predictor.set_image(frame_rgb)
#             results = self.predictor(text=self.objects_to_find, save=False, show=False, verbose=False)

#             for result in results:
#                 if result.masks is not None:
#                     masks = result.masks.data.cpu().numpy()
                    
#                     for i in range(len(masks)):
#                         mask_uint8 = (masks[i] * 255).astype(np.uint8)
                        
#                         if result.boxes is not None and len(result.boxes.cls) > i:
#                             cls_id = int(result.boxes.cls[i])
#                             obj_name = result.names[cls_id]
#                         else:
#                             obj_name = "apple"

#                         # 3. Cálculo de Centroide
#                         M = cv2.moments(mask_uint8)
#                         if M["m00"] < 50:
#                             continue
                        
#                         u_mask = int(M["m10"] / M["m00"])
#                         v_mask = int(M["m01"] / M["m00"])

#                         # Escalamos el centroide de la mascara al tamaño del frame original
#                         # porque los intrinsecos (fx, fy, cx, cy) corresponden al frame original
#                         img_h, img_w   = frame_bgr.shape[:2]
#                         mask_h, mask_w = mask_uint8.shape
#                         u = int(u_mask * img_w / mask_w)
#                         v = int(v_mask * img_h / mask_h)

#                         # 4. Obtener profundidad desde la nube de puntos
#                         # Pasamos el shape del frame original para escalar correctamente la proyeccion
#                         z_m = 0.0
#                         if self.last_cloud is not None:
#                             z_m = self.get_depth_from_mask(mask_uint8, frame_bgr.shape)

#                         if z_m == 0.0:
#                             rospy.logwarn_throttle(5, "Sin profundidad válida, saltando objeto")
#                             continue

#                         # 5. Proyección a coordenadas 3D
#                         x_c = (u - self.cx) * z_m / self.fx
#                         y_c = (v - self.cy) * z_m / self.fy

#                         self.publish_msg(x_c, y_c, z_m)

#                         # 6. Visualización sobre el frame (solo dibujamos, no mostramos aún)
#                         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                         cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 2)
#                         cv2.circle(frame_bgr, (u, v), 5, (0, 0, 255), -1)
#                         label = f"{obj_name} | X:{x_c:.2f} Y:{y_c:.2f} Z:{z_m:.2f}"
#                         cv2.putText(frame_bgr, label, (u, v - 10), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             # 7. Mostramos una sola ventana al final, fuera del loop de mascaras
#             cv2.imshow("SAM3 - Monitor", frame_bgr)
#             cv2.waitKey(1)

#         except Exception as e:
#             rospy.logerr(f"Error en el ciclo de visión: {e}")
        
#         finally:
#             self.is_processing = False

#     def publish_msg(self, x, y, z):
#         # Publica el punto 3D en el tópico object_centroid
#         target_msg = PointStamped()
#         target_msg.header.stamp = rospy.Time.now()
#         target_msg.header.frame_id = self.cam_frame
#         target_msg.point.x = x
#         target_msg.point.y = y
#         target_msg.point.z = z
#         self.pub.publish(target_msg)
#         rospy.loginfo_throttle(1, f"Publicando: X={x:.3f}  Y={y:.3f}  Z={z:.3f}")

# if __name__ == '__main__':
#     try:
#         KinovaVisionSAM3()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         cv2.destroyAllWindows()