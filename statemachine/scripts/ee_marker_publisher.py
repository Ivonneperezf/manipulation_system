#!/usr/bin/env python3
import rospy
import tf2_ros
from visualization_msgs.msg import Marker

# Ajusta estos frames a tu setup
BASE_FRAME = "m1n6s300_link_base"
EE_FRAME   = "m1n6s300_end_effector" 

def main():
    rospy.init_node("ee_marker_publisher")

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    marker_pub = rospy.Publisher("/ee_marker", Marker, queue_size=1)

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        try:
            trans = tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, rospy.Time(0))
            t = trans.transform.translation
            q = trans.transform.rotation

            marker = Marker()
            marker.header.frame_id = BASE_FRAME
            marker.header.stamp = rospy.Time.now()
            marker.ns = "ee_marker"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = t.x
            marker.pose.position.y = t.y
            marker.pose.position.z = t.z
            marker.pose.orientation = q

            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker.lifetime = rospy.Duration(0) 
            marker_pub.publish(marker)

            rospy.loginfo_throttle(
                1.0,
                f"EE pos -> x: {t.x:.4f}, y: {t.y:.4f}, z: {t.z:.4f}"
            )

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"TF lookup failed: {e}")

        rate.sleep()

if __name__ == "__main__":
    main()