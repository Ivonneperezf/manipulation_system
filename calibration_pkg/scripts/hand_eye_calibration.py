#!/usr/bin/env python3

import rospy
import tf2_ros
import tf_conversions
import numpy as np
import cv2
import yaml
import threading
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class HandEyeCalibration:

    def __init__(self):

        rospy.init_node("handeye_calibration")

        # Numero de poses a capturar para la calibracion
        self.base_frame = "m1n6s300_link_base"
        # self.ee_frame   = "m1n6s300_end_effector"
        self.ee_frame   = "m1n6s300_link_5"  # usando el ultimo link en vez del EE para evitar errores de colision con la camara
        self.n_captures = 15

        # Rutas
        self.path = "/home/user/Documentos/ROS/projects/kinova_robot_ws/src/kinova-ros/calibration/config"

        # Bandera de control para captura manual
        self.ready_to_capture = False

        # Almacena en buffer las transformaciones
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.bridge = CvBridge()

        # Listas para almacenar coordenadas y posiciones del EE a la base 
        self.R_gripper2base = []
        self.t_gripper2base = []

        # Listas para almacenar coordenadas y posiciones de la camara respecto al ArUco
        self.R_target2cam = []
        self.t_target2cam = []

        # Suscripcion al topico publicador de la camara
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # Configuracion del ArUco 
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Cargar calibracion intrinseca de la camara
        data = np.load(f"{self.path}/camera_final_calibration.npz")
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']

        # Hilo paralelo para capturar Enter del usuario
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


    # Funcion para detectar ArUco
    def image_callback(self, msg):
        size = 0.045  # tamaño del marcador en metros
        try:
            # Conversion de imagen a escala de grises
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Deteccion de marcadores ArUco
            corners, ids, rejected = self.detector.detectMarkers(gray)

            if ids is not None:
                # Devuelve vector de rotacion y traslacion del marcador respecto a la camara
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, size, self.camera_matrix, self.dist_coeffs)

                # Dibujar los marcadores detectados
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                for r, t in zip(rvec, tvec):
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, r, t, size)

                # Convertir a matriz de rotacion y vector de traslacion
                R_cam_marker, _ = cv2.Rodrigues(rvec[0])
                t_cam_marker = tvec[0].reshape(3, 1)

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

        print("Resultado T_ee_cam:\n", T)

        # Publicar resultado como TF
        self.publish_static_tf(T)

        # Guardar resultado en archivo YAML
        self.save_calibration(T)

        # Guardar matriz resultante en un archivo .npz para uso futuro
        np.savez(f"{self.path}/handeye_result.npz", T=T)

        # Cerrar ventana de visualizacion
        cv2.destroyAllWindows()

        print("\nCalibracion completada. Archivos guardados en:", self.path)

        # Termina el proceso de calibracion
        rospy.signal_shutdown("Calibracion completada")


    # Funcion para publicar resultado en ROS 
    def publish_static_tf(self, T):
        br = tf2_ros.StaticTransformBroadcaster()
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.ee_frame
        t.child_frame_id = "camera_calibrated"

        # Traslacion
        t.transform.translation.x = T[0, 3]
        t.transform.translation.y = T[1, 3]
        t.transform.translation.z = T[2, 3]

        # Rotacion
        self.quat = tf_conversions.transformations.quaternion_from_matrix(T)
        quat = self.quat

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        br.sendTransform(t)
        print("TF calibrado publicado.")

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
    HandEyeCalibration()