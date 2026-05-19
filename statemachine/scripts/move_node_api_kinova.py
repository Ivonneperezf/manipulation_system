#!/usr/bin/env python3

import roslib; roslib.load_manifest('kinova_demo')
import rospy
import sys
import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import math
import argparse
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String

"""==================================================================
        ESTE CODIGO ES PROVISIONAL, NO DEFINIREMOS LA ROTACION
    =================================================================="""
class MoveNodeKinova():
    def __init__(self):
        # Cargamos los parametros necesarios
        self.kinova_robotType = rospy.get_param("~kinova_robotType", "m1n6s300")
        #self.unit = rospy.get_param("~unit", "mq")
        self.verbose = rospy.get_param("~verbose", False)
        #self.relative = rospy.get_param("~relative", False)

        # Cargamos los parametros del robot
        self.robot_category = self.kinova_robotType[0]
        self.robot_category_version = int(self.kinova_robotType[1])
        self.wrist_type = self.kinova_robotType[2]
        self.arm_joint_number = int(self.kinova_robotType[3])
        self.robot_mode = self.kinova_robotType[4]
        self.finger_number = int(self.kinova_robotType[5])
        self.prefix = self.kinova_robotType + "_"
        self.finger_maxDist = 18.9/2/1000  # max distance for one finger in meter
        self.finger_maxTurn = 6800  # max thread turn for one finger

        # Variable para alcenar la posicion actual del robot
        self.currentCartesianCommand = [0.0] * 6 # default home in unit mq
        # Inicializamos el nodo 
        rospy.init_node('move_node_kinova')

        # Publishers
        self.status_pub = rospy.Publisher('/motion_done', String, queue_size=10)

        # Recibimos punto para enviar al robot
        rospy.Subscriber('/cartesian_goal', PointStamped, self._cartesian_callback)
        rospy.Subscriber('/joint_goal', Float64MultiArray, self._joint_callback)

    # Callback para recibir el punto deseado y ejecutar el movimiento
    def _cartesian_callback(self, msg):
        orientation = self.currentCartesianCommand[3:]
        pose = [msg.point.x, msg.point.y, msg.point.z] + orientation
        rospy.loginfo(f"Recibido punto: {pose}, ejecutando movimiento...")
        try:
            result = self.cartesian_pose_client(pose[:3], pose[3:])
            rospy.loginfo('Cartesian pose sent!')
            self.status_pub.publish("DONE" if result else "FAILED")
        except Exception as e:
            rospy.logerr(f"Error al ejecutar el movimiento: {e}")
            self.status_pub.publish("FAILED")

    def _joint_callback(self, msg):
        self.joint_goal = msg.data
        try:
            result = self.joint_angle_client(self.joint_goal)
            rospy.loginfo('Joint angles sent!')
            self.status_pub.publish("DONE" if result else "FAILED")
        except Exception as e:
            rospy.logerr(f"Error al ejecutar el movimiento: {e}")
            self.status_pub.publish("FAILED")

    # Obtenemos la posición actual del robot
    def getcurrentCartesianCommand(self):
        # wait to get current position
        topic_address = '/' + self.prefix + 'driver/out/cartesian_command'
        rospy.Subscriber(topic_address, kinova_msgs.msg.KinovaPose, self.setcurrentCartesianCommand)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.KinovaPose)
        rospy.loginfo('position listener obtained message for Cartesian pose. ')
    
    # Callback para almacenar la posición actual del robot
    def setcurrentCartesianCommand(self,feedback):
        currentCartesianCommand_str_list = str(feedback).split("\n")
        for index in range(0,len(currentCartesianCommand_str_list)):
            temp_str=currentCartesianCommand_str_list[index].split(": ")
            self.currentCartesianCommand[index] = float(temp_str[1])

    # Movemos a la posición deseada utilizando el action server de Kinova
    def cartesian_pose_client(self, position, orientation):
        """Send a cartesian goal to the action server."""
        action_address = '/' + self.prefix + 'driver/pose_action/tool_pose'
        client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmPoseAction)
        client.wait_for_server()

        goal = kinova_msgs.msg.ArmPoseGoal()
        goal.pose.header = std_msgs.msg.Header(frame_id=(self.prefix + 'link_base'))
        goal.pose.pose.position = geometry_msgs.msg.Point(
            x=position[0], y=position[1], z=position[2])
        goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

        rospy.loginfo('goal.pose in client 1: {}'.format(goal.pose.pose))

        client.send_goal(goal)

        if client.wait_for_result(rospy.Duration(10.0)):
            return client.get_result()
        else:
            client.cancel_all_goals()
            rospy.loginfo('the cartesian action timed-out')
            return None
    
    def joint_angle_client(self,angle_set):
        """Send a joint angle goal to the action server."""
        action_address = '/' + self.prefix + 'driver/joints_action/joint_angles'
        client = actionlib.SimpleActionClient(action_address,
                                            kinova_msgs.msg.ArmJointAnglesAction)
        client.wait_for_server()

        goal = kinova_msgs.msg.ArmJointAnglesGoal()

        goal.angles.joint1 = angle_set[0]
        goal.angles.joint2 = angle_set[1]
        goal.angles.joint3 = angle_set[2]
        goal.angles.joint4 = angle_set[3]
        goal.angles.joint5 = angle_set[4]
        goal.angles.joint6 = angle_set[5]
        goal.angles.joint7 = angle_set[6]

        client.send_goal(goal)
        if client.wait_for_result(rospy.Duration(20.0)):
            return client.get_result()
        else:
            rospy.logerr('the joint angle action timed-out')
            client.cancel_all_goals()
            return None
        
if __name__ == "__main__":
    try:
        controller = MoveNodeKinova()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Error al iniciar el nodo de movimiento del brazo")