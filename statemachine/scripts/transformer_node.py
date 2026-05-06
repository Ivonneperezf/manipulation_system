#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from tf.transformations import quaternion_matrix

class KinovaTransformer:
    def __init__(self):
        rospy.init_node('direct_transformer')

        # Cargar calibración
        prefix = "/camera_to_robot"
        t = rospy.get_param(f"{prefix}/translation")
        r = rospy.get_param(f"{prefix}/rotation")

        # Convertir a matriz homogénea
        self.T = quaternion_matrix([r['x'], r['y'], r['z'], r['w']])
        self.T[0:3, 3] = [t['x'], t['y'], t['z']]

        # Subs y pubs
        self.sub = rospy.Subscriber('/object_centroid', PointStamped, self.callback)
        self.pub = rospy.Publisher('/object_centroid_robot', PointStamped, queue_size=10)

        rospy.loginfo("Transformación a sistema de referencia del brazo")

    def callback(self, msg):
        # Punto en camara
        p_cam = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0])

        # Transformacion directa
        p_base = self.T @ p_cam

        # Crear mensaje
        out = PointStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "m1n6s300_link_base"

        out.point.x = p_base[0]
        out.point.y = p_base[1]
        out.point.z = p_base[2]

        self.pub.publish(out)

        rospy.loginfo_throttle(2, f"BASE → X:{p_base[0]:.3f} Y:{p_base[1]:.3f} Z:{p_base[2]:.3f}")

if __name__ == '__main__':
    KinovaTransformer()
    rospy.spin()