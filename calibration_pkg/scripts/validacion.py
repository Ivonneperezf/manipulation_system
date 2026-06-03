#!/usr/bin/env python3
"""
Script de VALIDACION EXTERNA de la calibracion hand-eye.

Carga la matriz de calibracion desde handeye_result.npz y verifica
su calidad moviendo el robot a poses NUEVAS (distintas a las de calibracion).

PRINCIPIO DE FUNCIONAMIENTO:
    El marcador ArUco esta fijo en el mundo. Sin importar desde que pose
    del robot lo veas, su posicion en coordenadas de la base debe ser siempre
    la misma si la calibracion es correcta:

        T_base_marker = T_base_ee @ T_ee_cam @ T_cam_marker

    La primera pose se guarda como referencia. Las siguientes se comparan
    contra ella. Si el error es pequeño, la calibracion es buena.

COMO USARLO:
    1. Pon el ArUco fijo en la escena, NO lo muevas durante la validacion
    2. Corre este script
    3. Mueve el robot a una pose donde vea el ArUco y presiona ENTER
       (esta sera la referencia)
    4. Mueve el robot a 5-10 poses NUEVAS y presiona ENTER en cada una
    5. Revisa el error al final

COMO INTERPRETAR EL ERROR DE TRASLACION:
    < 1 cm  -> Excelente, calibracion muy precisa
    < 3 cm  -> Aceptable para la mayoria de aplicaciones
    > 5 cm  -> Calibracion deficiente, considera recalibrar

COMO INTERPRETAR EL ERROR DE ROTACION:
    < 0.01  -> Excelente
    < 0.05  -> Aceptable
    > 0.10  -> Revisar calibracion

QUE REVISAR SI EL ERROR ES ALTO:
    1. Que el ArUco NO se haya movido entre capturas
    2. Que self.size coincida exactamente con el tamanio real del marcador
    3. Que las poses tengan suficiente variedad de orientaciones
    4. Que la deteccion del ArUco sea buena (sin blur, bien iluminado)
    5. Considera recalibrar con mas capturas y mejor variedad de poses
"""

import rospy
import tf2_ros
import tf_conversions
import numpy as np
import cv2
import threading
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rospkg as rp


# ==============================================================================
# CONFIGURACION: Ajusta estas rutas y parametros a tu setup
# ==============================================================================
rospack = rp.RosPack()
path_pkg = rospack.get_path("calibration_pkg") 
PATH_NPZ   = f"{path_pkg}/config/handeye_result.npz"
BASE_FRAME = "m1n6s300_link_base"
EE_FRAME   = "m1n6s300_link_5"
ARUCO_SIZE = 0.045  # Tamanio del marcador en metros (debe coincidir con el real)
# ==============================================================================


class ValidacionHandEye:

    def __init__(self):
        rospy.init_node("validacion_handeye")

        """PARAMETROS"""
        self.base_frame = BASE_FRAME
        self.ee_frame   = EE_FRAME
        self.size       = ARUCO_SIZE

        """VARIABLES DE VALIDACION"""
        self.ready            = False   # Bandera para capturar cuando el usuario presiona ENTER
        self.t_ref            = None    # Traslacion de referencia (primera captura)
        self.R_ref            = None    # Rotacion de referencia (primera captura)
        self.errores_traslacion = []    # Historial de errores de traslacion en metros
        self.errores_rotacion   = []    # Historial de errores de rotacion

        """CARGAR CALIBRACION DESDE NPZ"""
        rospy.loginfo(f"Cargando calibracion desde: {PATH_NPZ}")
        data          = np.load(PATH_NPZ)
        self.T_cam2ee = data["T"]  # Matriz homogenea 4x4 de camara a EE
        rospy.loginfo(f"Calibracion cargada:\n{np.round(self.T_cam2ee, 4)}")

        """CONFIGURACION DE LA CAMARA"""
        rospy.loginfo("Esperando camera_info...")
        msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs   = np.array(msg.D)
        rospy.loginfo("Intrinsicos cargados.")

        """CONFIGURACION DEL ARUCO"""
        aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        params        = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        """CONFIGURACION DE LAS TRANSFORMACIONES"""
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.bridge = CvBridge()

        """CONFIGURACION DE LA SUSCRIPCION"""
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        """HILO DE ENTRADA DE USUARIO"""
        self.input_thread = threading.Thread(target=self.wait_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        rospy.spin()


    def wait_for_input(self):
        """
        Hilo que espera la entrada del usuario para capturar una validacion.
        La primera captura se guarda como referencia.
        Las siguientes se comparan contra ella.
        """
        print("\n" + "=" * 55)
        print("VALIDACION EXTERNA DE CALIBRACION HAND-EYE")
        print("=" * 55)
        print("IMPORTANTE: El ArUco debe estar FIJO durante toda la validacion.")
        print("\nPon el robot en la primera pose y presiona ENTER")
        print("(esta pose se guardara como referencia)\n")

        while not rospy.is_shutdown():
            if self.t_ref is None:
                input("Primera pose (referencia) -> ENTER para capturar...")
            else:
                input(f"\n[{len(self.errores_traslacion)}/{len(self.errores_traslacion)}] "
                      f"Nueva pose -> ENTER para capturar...")
            self.ready = True


    def image_callback(self, msg):
        """
        Callback de la camara. Detecta el ArUco, promedia todos los marcadores
        visibles usando SVD (igual que en la calibracion) y captura si el
        usuario presiono ENTER.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray     = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.size, self.camera_matrix, self.dist_coeffs)

                # Dibujar marcadores y ejes detectados
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                for r, t in zip(rvec, tvec):
                    cv2.drawFrameAxes(cv_image, self.camera_matrix,
                                      self.dist_coeffs, r, t, self.size)

                if self.ready:
                    self.ready = False

                    # Promediar todos los marcadores visibles con SVD
                    # (mismo metodo que en la calibracion para consistencia)
                    R_cam_marker, t_cam_marker = self.promediar_marcadores(rvec, tvec)

                    # Validar la pose capturada
                    self.validar(R_cam_marker, t_cam_marker)

            # Mostrar imagen con marcadores
            cv2.imshow("Validacion Hand-Eye", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error en image_callback: {e}")


    def promediar_marcadores(self, rvec, tvec):
        """
        Promedia la pose de todos los marcadores ArUco detectados.

        Traslacion: promedio directo (valido en R3).
        Rotacion:   promedio via SVD proyectado a SO(3) (correcto matematicamente).

        Este metodo es identico al usado durante la calibracion, lo cual es
        importante para que la validacion sea consistente con los datos capturados.
        """
        R_matrices = []
        t_vectors  = []

        for r, t in zip(rvec, tvec):
            R, _ = cv2.Rodrigues(r)       # Convertir rvec a matriz de rotacion 3x3
            R_matrices.append(R)
            t_vectors.append(t.reshape(3, 1))

        # Promedio de traslaciones
        t_cam_marker = np.mean(t_vectors, axis=0)

        # Promedio de rotaciones via SVD
        R_sum = np.zeros((3, 3))
        for R in R_matrices:
            R_sum += R
        U, _, Vt = np.linalg.svd(R_sum / len(R_matrices))
        R_cam_marker = U @ Vt

        # Corregir si el resultado es una reflexion (det = -1) en vez de rotacion (det = +1)
        if np.linalg.det(R_cam_marker) < 0:
            U[:, -1] *= -1
            R_cam_marker = U @ Vt

        return R_cam_marker, t_cam_marker


    def validar(self, R_cam_marker, t_cam_marker):
        """
        Valida la calibracion comparando la posicion del ArUco en coordenadas
        de la base calculada desde distintas poses del robot.

        Si la calibracion es correcta, esta posicion debe ser constante
        sin importar desde donde mire el robot al marcador.

        Calculo:
            T_base_marker = T_base_ee  @  T_ee_cam  @  T_cam_marker
                            (TF robot)    (calibracion) (camara->ArUco)
        """
        try:
            # Obtener pose actual del EE respecto a la base por TF
            trans = self.tfBuffer.lookup_transform(
                self.base_frame, self.ee_frame,
                rospy.Time(0), rospy.Duration(1.0))

            t = trans.transform.translation
            q = trans.transform.rotation

            # Convertir a matriz homogenea 4x4
            T_base_ee = tf_conversions.transformations.quaternion_matrix(
                [q.x, q.y, q.z, q.w])
            T_base_ee[0:3, 3] = [t.x, t.y, t.z]

            # Construir matriz homogenea del marcador respecto a la camara
            T_cam_marker = np.eye(4)
            T_cam_marker[0:3, 0:3] = R_cam_marker
            T_cam_marker[0:3, 3]   = t_cam_marker.reshape(3)

            # Calcular posicion del marcador en coordenadas de la base
            # T_base_marker = T_base_ee @ T_ee_cam @ T_cam_marker
            T_base_marker = T_base_ee @ self.T_cam2ee @ T_cam_marker

            t_marker = T_base_marker[0:3, 3]        # Traslacion del marcador en base
            R_marker = T_base_marker[0:3, 0:3]      # Rotacion del marcador en base

            # Primera captura: guardar como referencia
            if self.t_ref is None:
                self.t_ref = t_marker.copy()
                self.R_ref = R_marker.copy()
                self.errores_traslacion.append(0.0)
                self.errores_rotacion.append(0.0)
                print("\nReferencia guardada.")
                print(f"  Posicion del ArUco en base: {np.round(self.t_ref, 4)} m")
                print("\nAhora mueve el robot a poses distintas sin mover el ArUco.")
                return

            # Capturas siguientes: comparar contra la referencia
            error_t = np.linalg.norm(t_marker - self.t_ref)  # metros
            error_R = np.linalg.norm(R_marker - self.R_ref)  # adimensional

            self.errores_traslacion.append(error_t)
            self.errores_rotacion.append(error_R)

            print(f"\nValidacion #{len(self.errores_traslacion) - 1}")
            print(f"  Posicion calculada: {np.round(t_marker, 4)} m")
            print(f"  Error traslacion:   {error_t * 100:.2f} cm")
            print(f"  Error rotacion:     {error_R:.4f}")

            self.mostrar_resumen()

        except Exception as e:
            rospy.logerr(f"Error en validacion TF: {e}")
            self.ready = True  # Permitir reintentar


    def mostrar_resumen(self):
        """
        Muestra un resumen estadistico del error acumulado hasta el momento
        e imprime un diagnostico automatico de la calidad de la calibracion.
        """
        # Ignorar el primer elemento (referencia, error = 0)
        et = np.array(self.errores_traslacion[1:])
        er = np.array(self.errores_rotacion[1:])

        if len(et) == 0:
            return

        print("\n--- Resumen acumulado ---")
        print(f"  Capturas validadas:         {len(et)}")
        print(f"  Error traslacion medio:     {np.mean(et) * 100:.2f} cm")
        print(f"  Error traslacion maximo:    {np.max(et)  * 100:.2f} cm")
        print(f"  Error rotacion medio:       {np.mean(er):.4f}")
        print(f"  Error rotacion maximo:      {np.max(er):.4f}")

        e = np.mean(et)
        print("\n  Diagnostico: ", end="")
        if e < 0.01:
            print("EXCELENTE - Calibracion muy precisa.")
        elif e < 0.03:
            print("ACEPTABLE - Valida para la mayoria de aplicaciones.")
        elif e < 0.05:
            print("REGULAR - Considera recalibrar con mas y mejores poses.")
        else:
            print("DEFICIENTE - Recalibra revisando deteccion ArUco y variedad de poses.")


if __name__ == "__main__":
    ValidacionHandEye()