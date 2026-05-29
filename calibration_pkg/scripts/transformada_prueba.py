#!/usr/bin/env python3
"""
Script de prueba para verificar transformadas TF del robot.
Equivalente en terminal:
    rosrun tf tf_echo base_link j2s7s300_link_7
"""

import rospy
import tf2_ros
import tf_conversions
import numpy as np


def test_tf_transform():
    rospy.init_node('test_tf_transform', anonymous=True)

    tf_buffer   = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)  # noqa: F841

    base_frame = 'base_link'
    ee_frame   = 'j2s7s300_link_7'  # Cambia segun tu robot

    rospy.loginfo(f"Esperando transformada: {base_frame} -> {ee_frame}")
    rospy.sleep(1.0)  # Dar tiempo al buffer para llenarse

    try:
        trans = tf_buffer.lookup_transform(
            base_frame,
            ee_frame,
            rospy.Time(0),
            rospy.Duration(2.0)
        )

        t = trans.transform.translation
        q = trans.transform.rotation

        rospy.loginfo(f"Traslacion: x={t.x:.4f}, y={t.y:.4f}, z={t.z:.4f}")
        rospy.loginfo(f"Cuaternion: x={q.x:.4f}, y={q.y:.4f}, z={q.z:.4f}, w={q.w:.4f}")

        # Conversion a matriz homogenea 4x4
        T = tf_conversions.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0:3, 3] = [t.x, t.y, t.z]

        R_base_ee = T[0:3, 0:3]
        t_base_ee = T[0:3, 3].reshape(3, 1)

        print("\n--- Matriz de Rotacion R_base_ee ---")
        print(np.round(R_base_ee, 4))

        print("\n--- Vector de Traslacion t_base_ee ---")
        print(np.round(t_base_ee, 4))

        print("\n--- Matriz Homogenea T (4x4) ---")
        print(np.round(T, 4))

    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Error al obtener transformada: {e}")


if __name__ == '__main__':
    test_tf_transform()