#!/usr/bin/env python3

import rospy
import tf2_ros
import tf_conversions
import numpy as np
from geometry_msgs.msg import TransformStamped



# Funcion para convertir Transform a matriz 4x4
def transform_to_matrix(transform):

    t = transform.transform.translation
    q = transform.transform.rotation

    translation = np.array([t.x, t.y, t.z])
    quaternion = [q.x, q.y, q.z, q.w]

    T = tf_conversions.transformations.quaternion_matrix(quaternion)
    T[0:3, 3] = translation

    return T


def main():

    rospy.init_node('handeye_matrix_node')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.sleep(2)  # espera a que tf cargue

    # Puntos de referencia para el sistema del robot simulado
    base_frame = "m1n6s300_link_base"
    ee_frame = "m1n6s300_end_effector"
    camera_optical_frame = "d415_color_optical_frame"

    try:

        # Vectores de posicion y rotacion del lente optico al end effector
        trans_cam_ee = tfBuffer.lookup_transform(
            ee_frame,
            camera_optical_frame,
            rospy.Time(0),
            rospy.Duration(5.0)
        )

        # Transformada homogenea correspondiente
        T_cam_ee = transform_to_matrix(trans_cam_ee) 

        # Vectores de posicion y rotacion del end effector a base-link
        trans_ee_base = tfBuffer.lookup_transform(
            base_frame,
            ee_frame,
            rospy.Time(0),
            rospy.Duration(5.0)
        )

        # Transformada homogenea correspondiente
        T_ee_base = transform_to_matrix(trans_ee_base)

        # Producto punto de ambas tranformadas
        T_base_cam = np.dot(T_ee_base, T_cam_ee)

        
        print("\n==============================")
        print("T_cam_ee (Lente -> EE):\n", T_cam_ee)
        print("\nT_ee_base (EE -> Base):\n", T_ee_base)
        print("\nT_base_cam (RESULTADO FINAL):\n", T_base_cam)
        print("==============================\n")

        # Publicar como TF estatico
        br = tf2_ros.StaticTransformBroadcaster()

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = base_frame
        t.child_frame_id = "camera_test"

        # Traslacion
        t.transform.translation.x = T_base_cam[0, 3]
        t.transform.translation.y = T_base_cam[1, 3]
        t.transform.translation.z = T_base_cam[2, 3]

        # Rotacion
        quat = tf_conversions.transformations.quaternion_from_matrix(T_base_cam)

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        br.sendTransform(t)

        print("Transform publicada como base → camera_test")

        rospy.spin()

    except Exception as e:
        print("Error obteniendo transformadas:", e)


if __name__ == '__main__':
    main()
