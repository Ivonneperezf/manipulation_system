#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
import cv2
import rospkg
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from ultralytics.models.sam import SAM3SemanticPredictor

class KinovaVisionSAM3:
    def __init__(self):
        rospy.init_node('vision_d415_sam3')
        
        # Publicador del centroide
        self.pub = rospy.Publisher('object_centroid', PointStamped, queue_size=10)

        #Tópicos fijos de la RealSense D415
        self.TOPIC_RGB = "/d415/color/image_raw"
        self.TOPIC_DEPTH = "/d415/aligned_depth_to_color/image_raw"
        self.TOPIC_INFO = "/d415/color/camera_info"
        
        # Flag para evitar que los callbacks se acumulen (Control de Concurrencia)
        self.is_processing = False 
        
        rospy.loginfo("Iniciando nodo Kinova-SAM3...") 
        
        # Obtener ruta del modelo dinámicamente
        rp = rospkg.RosPack()
        try:
            package_path = rp.get_path('sam_segmentation')
            model_path = package_path + "/weights/sam3.pt"
        except Exception as e:
            rospy.logwarn(f"No se encontró el paquete: {e}. Usando ruta local.")
            model_path = "sam3.pt"

        rospy.loginfo(f"Cargando pesos desde: {model_path}")
        
        # Configuración de SAM3 (Promptable Concept Segmentation)
        overrides = dict(
            conf=0.35,      # Umbral de confianza
            task="segment",
            mode="predict",
            model=model_path,
            half=True,      # Usa FP16 para mayor velocidad en GPUs NVIDIA
        )
        
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

        self.last_depth = None 

        #Estado y Suscriptores
        self.last_depth = None # Variable para almacenar la última imagen de profundidad recibida, necesaria para sincronizar con el RGB
        rospy.Subscriber(self.TOPIC_DEPTH, Image, self.depth_cb) # Suscripción a la imagen de profundidad alineada con el color, sin cola para procesar cada frame de profundidad que llega
        rospy.Subscriber(self.TOPIC_RGB, Image, self.rgb_cb, queue_size=1, buff_size=2**24) # Suscripción a la imagen RGB con cola de tamaño 1 para no saturar la memoria si el procesamiento es lento
        
        rospy.loginfo("Nodo SAM3 listo y procesando...")

    def depth_cb(self, msg): 
        # Convertimos la profundidad a numpy una sola vez
        self.last_depth = ros_numpy.numpify(msg) 

    def rgb_cb(self, msg):
        # Si el modelo está ocupado o no hay profundidad, saltamos el frame
        if self.is_processing or self.last_depth is None:
            return

        try:
            self.is_processing = True
            
            # 1. Preparación de imagen
            frame_rgb = ros_numpy.numpify(msg)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 2. Inferencia SAM3
            # set_image calcula los embeddings de la imagen (lo más pesado)
            self.predictor.set_image(frame_rgb)
            
            # Buscamos los conceptos definidos por texto (Zero-shot)
            results = self.predictor(text=self.objects_to_find)

            for result in results:
                if result.masks is not None:
                    # Las máscaras en SAM3 vienen como un tensor (N, H, W)
                    masks = result.masks.data.cpu().numpy()
                    
                    for i in range(len(masks)):
                        # Convertir a binario para OpenCV
                        mask_uint8 = (masks[i] * 255).astype(np.uint8)
                        
                        # Intentar obtener el nombre de la clase si existe detección asociada
                        if result.boxes is not None and len(result.boxes.cls) > i:
                            cls_id = int(result.boxes.cls[i])
                            obj_name = result.names[cls_id]
                        else:
                            obj_name = "fruit_piece"

                        # 3. Cálculo de Centroide
                        M = cv2.moments(mask_uint8)
                        if M["m00"] < 50: continue # Ignorar ruidos muy pequeños
                        
                        u = int(M["m10"] / M["m00"])
                        v = int(M["m01"] / M["m00"])

                        # 4. Obtener Profundidad Filtrada
                        z_m = self.get_filtered_depth(u, v)

                        # Filtro de distancia de seguridad para el Kinova (15cm a 1 metro)
                        if 0.15 < z_m < 1.0:
                            # Proyección de Píxel a Coordenadas Cámara (X, Y, Z)
                            x_c = (u - self.cx) * z_m / self.fx
                            y_c = (v - self.cy) * z_m / self.fy
                            
                            self.publish_msg(x_c, y_c, z_m)

                            # 5. Visualización
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(frame_bgr, contours, -1, (0, 255, 0), 2)
                            
                            label = f"{obj_name} {z_m:.2f}m"
                            cv2.circle(frame_bgr, (u, v), 5, (0, 0, 255), -1)
                            cv2.putText(frame_bgr, label, (u, v - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Mostrar ventana de monitoreo
            cv2.imshow("Kairós SAM3 - Monitor", frame_bgr)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error en el ciclo de visión: {e}")
        
        finally:
            # Liberar el flag para permitir procesar el siguiente frame disponible
            self.is_processing = False

    def get_filtered_depth(self, u, v):
        """Calcula la mediana de profundidad en un área de 7x7 para evitar ruido."""
        try:
            h, w = self.last_depth.shape
            # Asegurar que el punto está dentro de la imagen
            u = np.clip(u, 0, w-1)
            v = np.clip(v, 0, h-1)
            
            # Extraer región de interés (ROI)
            roi = self.last_depth[max(0, v-3):min(h, v+4), max(0, u-3):min(w, u+4)]
            valid_depths = roi[roi > 0] # Filtrar ceros (píxeles sin lectura)
            
            if len(valid_depths) > 0:
                # El sensor D415 entrega profundidad en mm, convertimos a metros
                return np.median(valid_depths) * 0.001
            return 0.0
        except:
            return 0.0

    def publish_msg(self, x, y, z):
        """Publica el punto 3D en el tópico object_centroid."""
        target_msg = PointStamped()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.header.frame_id = self.cam_frame
        target_msg.point.x = x
        target_msg.point.y = y
        target_msg.point.z = z
        self.pub.publish(target_msg)

if __name__ == '__main__':
    try:
        KinovaVisionSAM3()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()