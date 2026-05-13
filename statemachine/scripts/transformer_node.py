#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from tf.transformations import quaternion_matrix
import tf_conversions
import tf2_ros
import csv
import rospkg as rp

class KinovaTransformer:
    def __init__(self):
        rospy.init_node('direct_transformer')

        # Frames
        self.base_frame = "m1n6s300_link_base"
        self.ee_frame = "m1n6s300_link_5"

        # Rutas
        rospack = rp.RosPack()
        self.package_path = rospack.get_path(rospy.get_param("~paths/pack", 'statemachine'))

        # Cargar calibración
        prefix = "/camera_to_robot"
        t = rospy.get_param(f"{prefix}/translation")
        r = rospy.get_param(f"{prefix}/rotation")
        
        self.T_cam2ee = tf_conversions.transformations.quaternion_matrix([r['x'], r['y'], r['z'], r['w']])
        self.T_cam2ee[0:3,3]= [t['x'], t['y'], t['z']]

        # Obtenemos la transformada en tiempo real de las articulaciones
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subs y pubs
        self.sub = rospy.Subscriber('/object_centroid', PointStamped, self.callback)
        self.pub = rospy.Publisher('/object_centroid_robot', PointStamped, queue_size=10)

        rospy.loginfo("Transformación a sistema de referencia del brazo")

    def callback(self, msg):
        try:
            # Obtenemos los tf de link_5 y base_link en tiempo real
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rospy.Time(0),
                rospy.Duration(0)
            )

            t = trans.transform.translation
            q = trans.transform.rotation

            T_ee2base = tf_conversions.transformations.quaternion_matrix(
                [q.x, q.y, q.z, q.w]
            )
            T_ee2base[0:3,3] = [t.x, t.y, t.z]

            # Calculamos la transformada respecto a la base
            T_cam2base = T_ee2base @ self.T_cam2ee

            # Transformamos el punto respecto a la base del brazo
            p_cam = np.array([msg.point.x, msg.point.y, msg.point.z,1.0])
            p_base = T_cam2base @ p_cam

            # Publicamos el punto transformado
            out = PointStamped()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = self.base_frame
            out.point.x = p_base[0]
            out.point.y = p_base[1]
            out.point.z = p_base[2]

            with open(f"{self.package_path}/config/transform_points.csv", "a", newline="", encoding="utf-8") as file:
                text = csv.writer(file)
                text.writerow([p_base[0],p_base[1],p_base[2]])
            self.pub.publish(out)

            rospy.loginfo_throttle(2, f"BASE -> X:{p_base[0]:.3f} Y{p_base[1]:.3f} Z{p_base[2]:.3f}")
        except Exception as e:
            rospy.logwarn(f"Error TF: {e}")


if __name__ == '__main__':
    KinovaTransformer()
    rospy.spin()