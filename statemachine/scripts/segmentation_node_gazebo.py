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
        # Inicia nodo
        rospy.init_node('vision_simulation')
        # Creacion de publiser de punto final refrente a la camara
        self.pub = rospy.Publisher('/object_centroid', PointStamped, queue_size=10)
        
        # NUEVO: Publisher para visualizar la máscara filtrada en RViz
        self.mask_pub = rospy.Publisher('/object_mask_filtered', Image, queue_size=1)

        # Cargamos topicos por defecto de simulacion
        self.TOPIC_RGB = rospy.get_param("~topics/rgb_topic", "/d415/color/image_raw")
        self.TOPIC_INFO = rospy.get_param("~topics/camera_info_topic", "/d415/color/camera_info")
        self.TOPIC_POINTS = rospy.get_param("~topics/points_topic", "/d415/depth/points")

        # Cargamos los modelos 
        rospy.loginfo("Cargando modelos de YOLOv8 y MobileSAM")
        rospack = rp.RosPack()
        package_path = rospack.get_path(rospy.get_param("~paths/pack", 'statemachine'))
        self.yolo = YOLO(package_path + '/weights/yolov8s.pt')
        self.sam = SAM(package_path  + '/weights/sam2_b.pt')

        # Obtenemos parámetros intrínsecos de la cámara desde el tópico de info
        rospy.loginfo("Sincronizando con cámara Gazebo D415...")
        try:
            info = rospy.wait_for_message(self.TOPIC_INFO, CameraInfo, timeout=10)
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]
            self.cam_frame   = info.header.frame_id
        except rospy.ROSException:
            rospy.logerr("No se detectó la cámara. Revisa que Gazebo esté corriendo.")
            return

        # Variables para callbacks
        self.last_cloud  = None

        # Subscriptores
        rospy.Subscriber(self.TOPIC_POINTS, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber(self.TOPIC_RGB, Image, self.rgb_cb, queue_size=1, buff_size=2**24)

    # Callback de nube de puntos
    def cloud_cb(self, msg):
        # Lee los puntos x, y y z, ignorando los valores nulos
        puntos = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        # Si hay puntos validos los guarda en un array de atributo de clase
        if len(puntos) > 0:
            self.last_cloud = np.array(puntos, dtype=np.float32)

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

        # NUEVO: Generar y publicar una máscara visual que solo contenga los puntos validados por la nube
        # Creamos una imagen en negro del mismo tamaño que la máscara original
        filtered_mask_visual = np.zeros_like(mask, dtype=np.uint8)
        if len(z_masked) > 0:
            # Los píxeles que pasaron todos los filtros y tienen profundidad válida se pintan de blanco (255)
            filtered_mask_visual[v_mask[in_mask], u_mask[in_mask]] = 255
        
        # Convertimos la matriz de OpenCV/NumPy a mensaje de ROS tipo sensor_msgs/Image
        # Se usa codificación 'mono8' al ser una imagen en escala de grises de 8 bits
        mask_msg = ros_numpy.msgify(Image, filtered_mask_visual, encoding='mono8')
        mask_msg.header.stamp = rospy.Time.now()
        mask_msg.header.frame_id = self.cam_frame
        self.mask_pub.publish(mask_msg)

        # En caso de no haber puntos retornamos fallo
        if len(z_masked) == 0:
            return 0.0
        # Retornamos la profundidad promedio de los puntos dentro de la mascara
        return float(np.mean(z_masked))

    # Callback de RGB
    def rgb_cb(self, msg):
        # Convertimos la imagen de ROS a formato OpenCV 
        frame_rgb = ros_numpy.numpify(msg)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Definimos las clases de interes
        clases_interes = [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

        # Redimensionamos la imagen para YOLO y calculamos los factores de escala
        input_size = 320
        img_small = cv2.resize(frame_bgr, (input_size, input_size))
        h_orig, w_orig = frame_bgr.shape[:2]
        sw, sh = w_orig / input_size, h_orig / input_size

        # Detección con YOLOv8
        results = self.yolo(img_small, verbose=False, conf=0.5, classes=clases_interes)[0]

        # Procesamos cada caja detectada por YOLO
        for box in results.boxes:
            # Obtenemos clase y coordenadas de la caja
            class_id = int(box.cls[0])
            class_name = self.yolo.names[class_id]

            # Redimensionamos las coordenadas de la caja a la imagen original
            b = box.xyxy[0].cpu().numpy()
            x1, y1 = int(b[0] * sw), int(b[1] * sh)
            x2, y2 = int(b[2] * sw), int(b[3] * sh)
            cx_box = (x1 + x2) / 2
            cy_box = (y1 + y2) / 2

            # Segmentamos usando SAM con la caja de YOLO como referencia
            sam_results = self.sam.predict(frame_bgr, bboxes=[[x1, y1, x2, y2]], points=[[cx_box, cy_box]], labels=[1], verbose=False, imgsz=320)[0]
            
            z_m = 0.0
            # Si SAM devuelve una máscara válida, calculamos el centroide y la profundidad
            if sam_results.masks is not None:
                # Calculamos el centroide de la máscara para obtener coordenadas (u, v)
                mask = sam_results.masks.data[0].cpu().numpy().astype(np.uint8)
                # Obtenemos el centroide de la máscara usando momentos de imagen
                M = cv2.moments(mask)
                if M["m00"] == 0:
                    continue
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                
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

                # Publicamos el mensaje con las coordenadas del centroide
                self.publish_msg(x_c, y_c, z_m)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 2)
                label = f"{class_name.upper()} | X:{x_c:.2f} Y:{y_c:.2f} Z:{z_m:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame_bgr, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.circle(frame_bgr, (u, v), 5, (0, 0, 255), -1)

        cv2.imshow("D415 Gazebo - YOLO+SAM", frame_bgr)
        cv2.waitKey(1)

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
        KinovaVisionD415()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()