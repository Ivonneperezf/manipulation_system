#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, TransformStamped

class KinovaTransformer:
    def __init__(self):
        rospy.init_node('kinova_transformer')
        self.PARENT_FRAME = "m1n6s300_joint_5"
        self.CHILD_FRAME = "d415_color_optical_frame" 

        self.ROBOT_BASE = "m1n6s300_link_base"

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub = rospy.Subscriber('/object_centroid', PointStamped, self.callback)
        self.pub = rospy.Publisher( '/object_centroid_robot', PointStamped, queue_size=10)
        rospy.loginfo(f"Nodo Transformador Listo. Cámara montada en: {self.PARENT_FRAME}")

    # def publish_static_tf(self):
    #     try:
    #         prefix = "/camera_to_robot"
    #         t = rospy.get_param(f"{prefix}/translation")
    #         r = rospy.get_param(f"{prefix}/rotation")
    #         broadcaster = tf2_ros.StaticTransformBroadcaster()
    #         static_tf = TransformStamped()
    #         static_tf.header.stamp = rospy.Time.now()
    #         static_tf.header.frame_id = self.PARENT_FRAME
    #         static_tf.child_frame_id  = self.CHILD_FRAME
    #         static_tf.transform.translation.x = t['x']
    #         static_tf.transform.translation.y = t['y']
    #         static_tf.transform.translation.z = t['z']
    #         static_tf.transform.rotation.x = r['x']
    #         static_tf.transform.rotation.y = r['y']
    #         static_tf.transform.rotation.z = r['z']
    #         static_tf.transform.rotation.w = r['w']
    #         broadcaster.sendTransform(static_tf)
    #         rospy.loginfo(f"TF Estática publicada: {self.PARENT_FRAME} -> {self.CHILD_FRAME}")
    #     except KeyError as e:
    #         rospy.logerr(f"Error: No se encontraron parámetros en el YAML. Revisa el prefijo {prefix}")

    def callback(self, msg_cam):
        try:
            rospy.loginfo(msg_cam)
            msg_robot = self.tf_buffer.transform(
                msg_cam, self.ROBOT_BASE, timeout=rospy.Duration(0.2)
            )
            self.pub.publish(msg_robot)
            rospy.loginfo_throttle(2,
                f"Objeto respecto a BASE: "
                f"X:{msg_robot.point.x:.2f} "
                f"Y:{msg_robot.point.y:.2f} "
                f"Z:{msg_robot.point.z:.2f}"
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, "Esperando flujo de TF del brazo Kinova...")

if __name__ == '__main__':
    try:
        KinovaTransformer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass