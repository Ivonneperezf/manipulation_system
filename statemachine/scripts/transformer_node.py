#!/usr/bin/env python3
"""
Cadena cinematica aplicada:
    p_base = T_base_ee @ T_ee_cam @ p_cam

Donde:
    T_base_ee  : transformada del link_5 a la base (obtenida en tiempo real por TF)
    T_ee_cam   : transformada de la camara al link_5 (resultado de la calibracion hand-eye)
    p_cam      : punto detectado en coordenadas de la camara
    p_base     : punto resultante en coordenadas de la base del brazo
"""

import rospy
import numpy as np
import tf2_ros
import tf_conversions
import rospkg as rp
import csv
from geometry_msgs.msg import PointStamped


class KinovaTransformer:

    def __init__(self):
        rospy.init_node('direct_transformer')

        """PARAMETROS DE FRAMES"""
        self.base_frame = "m1n6s300_link_base"
        self.ee_frame   = "m1n6s300_link_5"

        """RUTAS"""
        rospack          = rp.RosPack()
        package_path     = rospack.get_path(rospy.get_param("~pack", "calibration_pkg"))
        self.config_path = f"{package_path}/config"

        """CARGAR CALIBRACION DESDE NPZ"""
        # Carga la matriz homogenea 4x4 resultado de la calibracion hand-eye
        # que representa la transformada de la camara al link_5
        npz_path      = f"{self.config_path}/handeye_result.npz"
        data          = np.load(npz_path)
        self.T_cam2ee = data["T"]
        rospy.loginfo(f"Calibracion cargada desde: {npz_path}")
        rospy.loginfo(f"T_cam2ee:\n{np.round(self.T_cam2ee, 4)}")

        """CONFIGURACION TF"""
        # Buffer para obtener la transformada del link_5 a la base en tiempo real
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        """SUSCRIPTORES Y PUBLICADORES"""
        rospy.Subscriber('/object_centroid', PointStamped, self.callback)
        self.pub = rospy.Publisher('/object_centroid_robot', PointStamped, queue_size=10)

        rospy.loginfo("Nodo listo. Esperando puntos en /object_centroid...")


    def callback(self, msg):
        try:
            # Obtener transformada del link_5 respecto a la base en tiempo real
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rospy.Time(0),
                rospy.Duration(1.0)  # Tiempo de espera al buffer
            )

            t = trans.transform.translation
            q = trans.transform.rotation

            # Convertir a matriz homogenea 4x4: T_base_ee
            T_base_ee = tf_conversions.transformations.quaternion_matrix(
                [q.x, q.y, q.z, q.w])
            T_base_ee[0:3, 3] = [t.x, t.y, t.z]

            # Cadena cinematica completa: camara -> link_5 -> base
            # T_base_cam = T_base_ee @ T_ee_cam
            T_base_cam = T_base_ee @ self.T_cam2ee

            # Punto en coordenadas de la camara (homogeneo)
            p_cam = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0])

            # Transformar al sistema de referencia de la base
            p_base = T_base_cam @ p_cam

            # Publicar punto transformado
            out                 = PointStamped()
            out.header.stamp    = rospy.Time.now()
            out.header.frame_id = self.base_frame
            out.point.x         = p_base[0]
            out.point.y         = p_base[1]
            out.point.z         = p_base[2]
            self.pub.publish(out)

            rospy.loginfo_throttle(2,
                f"Punto inicial -> X:{msg.point.x:.3f} Y:{msg.point.y:.3f} Z:{msg.point.z:.3f}\np_base -> X:{p_base[0]:.3f} Y:{p_base[1]:.3f} Z:{p_base[2]:.3f}")

        except Exception as e:
            rospy.logwarn(f"Error en transformacion: {e}")


if __name__ == '__main__':
    KinovaTransformer()
    rospy.spin()