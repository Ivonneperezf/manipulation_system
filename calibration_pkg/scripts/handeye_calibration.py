#!/usr/bin/env python3

import rospy
import tf2_ros
import tf_conversions
import numpy as np
import cv2
import yaml
import threading
import rospkg as rp
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

class HandEyeCalibration:

    def __init__(self, n_captures):
        # Iniciamos nodo de calibracion
        rospy.init_node("handeye_calibration")

        """PARAMETROS DE CALIBRACION"""
        # Definimos el frame base del robot
        self.base_frame = "m1n6s300_link_base"
        # Definimos el frame donde se encuentra la montura de la camara"
        self.ee_frame   = "m1n6s300_link_5" 
        # Definimos el numero de capturas a realizar
        self.n_captures = n_captures

        """PARAMETROS PARA GUARDAR RESULTADOS"""
        # Obtenemos la ruta del paquete 
        rospack = rp.RosPack()
        path_pkg = rospack.get_path("calibration_pkg")
        # Definimos la ruta donde se guardaran los resultados de la calibracion
        self.path = f"{path_pkg}/config"

        """VARIABLES DE CALIBRACION"""
        # Parametro para controlar la captura de poses, se activa al presionar Enter
        self.ready_to_capture = False
        self.size = 0.045 # Tamaño del marcador del ArUco
        self.quat = None # Variable para almacenar la orientacion en cuaterniones de la transformada resultante
        # Listas para almacenar coordenadas y posiciones del EE a la base 
        self.R_gripper2base = []
        self.t_gripper2base = []
        # Listas para almacenar coordenadas y posiciones de la camara respecto al ArUco
        self.R_target2cam = []
        self.t_target2cam = []

        """CONFIGURACION DE LA CAMARA"""
        # Cargar calibracion intrinseca desde el topic camera_info
        rospy.loginfo("Esperando camera_info...")
        msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)
        rospy.loginfo("Intrínsecos cargados desde camera_info")

        """CONFIGURACION DEL ARUCO"""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        """CONFIGURACION DE LAS TRANSFORMACIONES"""
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.bridge = CvBridge()

        """CONFIGURACION DE LA SUSCRIPCION"""
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        """CONFIGURACION DEL HILO DE ENTRADA"""
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        rospy.spin()


    # Hilo para leer la captura de usuario
    def wait_for_input(self):
        while not rospy.is_shutdown():
            if len(self.R_gripper2base) >= self.n_captures:
                break
            input(f"\n[{len(self.R_gripper2base)}/{self.n_captures}] Mueve el robot a una nueva pose y presiona ENTER para capturar...")
            self.ready_to_capture = True


    # Funcion para detectar ArUco por medio de la lectura de la camara
    def image_callback(self, msg):
        try:
            # Conversion de imagen a escala de grises
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Deteccion de marcadores ArUco
            corners, ids, rejected = self.detector.detectMarkers(gray)

            if ids is not None:
                # Devuelve vector de rotacion y traslacion del marcador respecto a la camara
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.size, self.camera_matrix, self.dist_coeffs)

                # Dibujar todos los marcadores detectados para visualizacion
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                for r, t in zip(rvec, tvec):
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, r, t, self.size)

                # Usar SOLO el marcador con id=0 para la calibracion
                ids_flat = ids.flatten()
                target_id = 0
                if target_id not in ids_flat:
                    cv2.imshow("Camera View", cv_image)
                    cv2.waitKey(1)
                    return

                # Obtener indice del marcador id=0
                idx = np.where(ids_flat == target_id)[0][0]

                # Extraer R y t solo de ese marcador
                R_cam_marker, _ = cv2.Rodrigues(rvec[idx])
                t_cam_marker = tvec[idx].reshape(3, 1)

                # Solo captura si el usuario presiono ENTER
                if self.ready_to_capture:
                    self.ready_to_capture = False
                    self.capture_pose(R_cam_marker, t_cam_marker)

            # Mostrar la imagen en ventana
            cv2.imshow("Camera View", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            print("Error deteccion:", e)

    # Funcion para capturar pose del robot
    def capture_pose(self, R_cam_marker, t_cam_marker):
        try:
            # Obtiene las transformadas de la base al EE en ese momento
            trans = self.tfBuffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            # Extrae datos de la transformada
            t = trans.transform.translation
            q = trans.transform.rotation

            # Conversion a matriz homogenea
            T = tf_conversions.transformations.quaternion_matrix(
                [q.x, q.y, q.z, q.w]
            )
            T[0:3, 3] = [t.x, t.y, t.z]

            # Extrae matriz de rotacion y vector de traslacion del EE a la base
            R_base_ee = T[0:3, 0:3]
            t_base_ee = T[0:3, 3].reshape(3, 1)

            # Guardar datos en sus respectivas listas
            self.R_gripper2base.append(R_base_ee)
            self.t_gripper2base.append(t_base_ee)
            self.R_target2cam.append(R_cam_marker)
            self.t_target2cam.append(t_cam_marker)

            print(f"Pose capturada. Total: {len(self.R_gripper2base)}/{self.n_captures}")

            # Si ya se alcanzo el limite de poses entonces guardamos la calibracion
            if len(self.R_gripper2base) >= self.n_captures:
                self.compute_handeye()

        except Exception as e:
            print("Error TF:", e)
            self.ready_to_capture = True


    # Funcion para calcular calibracion
    def compute_handeye(self):

        # Ejecuta calibracion con algoritmo de Daniilidis
        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
            self.R_gripper2base,
            self.t_gripper2base,
            self.R_target2cam,
            self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_DANIILIDIS
        )

        # Construir matriz homogenea de la transformada resultante
        T = np.eye(4)
        T[0:3, 0:3] = R_cam2ee
        T[0:3, 3] = t_cam2ee.reshape(3)

        self.quat = tf_conversions.transformations.quaternion_from_matrix(T)

        print("Resultado T_ee_cam:\n", T)

        # Guardar resultado en archivo YAML
        self.save_calibration(T)

        # Guardar matriz resultante en un archivo .npz para uso futuro
        np.savez(f"{self.path}/handeye_result.npz", T=T)

        # Cerrar ventana de visualizacion
        cv2.destroyAllWindows()

        print("\nCalibracion completada. Archivos guardados en:", self.path)

        # Termina el proceso de calibracion
        rospy.signal_shutdown("Calibracion completada")

    # Funcion para guardar resultado en un archivo YAML
    def save_calibration(self, T):
        data = {
            "camera_to_robot":{
                "translation": {
                    "x": float(T[0, 3]),
                    "y": float(T[1, 3]),
                    "z": float(T[2, 3])
                },
                "rotation": {
                    "x": float(self.quat[0]),
                    "y": float(self.quat[1]),
                    "z": float(self.quat[2]),
                    "w": float(self.quat[3])
                }
            }
        }
        with open(f"{self.path}/handeye_calibration.yaml", "w") as f:
            yaml.dump(data, f)

        print("Archivo handeye_calibration.yaml guardado.")


if __name__ == "__main__":
    HandEyeCalibration(n_captures=20)