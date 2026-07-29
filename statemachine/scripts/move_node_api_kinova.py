#!/usr/bin/env python3

import roslib; roslib.load_manifest('kinova_demo')
import rospy
import sys
import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import math
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String

class MoveNodeKinova():
    def __init__(self):
        # Cargamos los parametros necesarios
        self.kinova_robotType = rospy.get_param("~kinova_robotType", "m1n6s300")
        self.unit = rospy.get_param("~unit", "mq")
        self.verbose = rospy.get_param("~verbose", False)

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

        # Variable para almacenar la posicion actual del robot
        self.currentJointCommand = [0.0]*7
        self.pose_value = []
        self.currentCartesianCommand = [0.0] * 7

        # Inicializamos el nodo 
        rospy.init_node('move_node_kinova')
        rospy.loginfo('Nodo iniciado, esperando comandos...')

        # Publishers
        self.status_pub = rospy.Publisher('/motion_done', String, queue_size=10)

        # Subscribers
        rospy.Subscriber('/cartesian_goal', PointStamped, self._cartesian_callback)
        rospy.Subscriber('/joint_goal', Float64MultiArray, self._joint_callback)

    # Callback para recibir el punto deseado y ejecutar el movimiento (Siempre es en unidades de mq)
    def _cartesian_callback(self, msg):
        # Obtenemos la posicion actual del robot antes de enviar el comando
        self.getcurrentCartesianCommand()  
        # Convertimos la orientación de Euler XYZ a Quaternion
        orientation_q = self.EulerXYZ2Quaternion(self.currentCartesianCommand[3:])
        rospy.loginfo(f"Orientacion {orientation_q}")
        # Manteniendo la orientacion actual del robot, enviamos el punto cartesiano deseado
        self.pose_value = [msg.point.x, msg.point.y, msg.point.z] + orientation_q
        rospy.loginfo(f"Pose enviada {self.pose_value}")
        try:
            # Enviamos la pose
            poses = [float(n) for n in self.pose_value]
            result = self.cartesian_pose_client(poses[:3], poses[3:])
            rospy.loginfo('Cartesian pose sent!')
            # Publicamos el resultado correspondiente
            self.status_pub.publish("DONE" if result else "FAILED")

        except rospy.ROSInterruptException:
            rospy.logerr("program interrupted before completion")
            self.status_pub.publish("FAILED")


    def _joint_callback(self, msg):
        rospy.loginfo(f"Recibido mensaje en /joint_goal: {msg}")
        self.joint_goal = [float(n) for n in msg.data]
        rospy.loginfo(f"Recibido joint goal: {self.joint_goal}, ejecutando movimiento...")
        # Obtenemos la posición actual del robot antes de enviar el comando
        self.getcurrentJointCommand()
        # Parseamos a grados y a radianes
        joint_degree, joint_radian = self.unitParser("radian", self.joint_goal, False)
        positions = [0]*7
        try:
            # Si no se definieron los grados de los arituclaciones, es decir la lista esta vacia
            if self.arm_joint_number < 1:
                rospy.logerr('Joint number is 0, check with "-h" to see how to use this node.')
                positions = []  # Get rid of static analysis warning that doesn't see the exit()
                sys.exit() 
            # Si no enviamos la posicion
            else:
                for i in range(0,self.arm_joint_number):
                    positions[i] = joint_degree[i] 
            result = self.joint_angle_client(positions)
            rospy.loginfo('Joint angles sent!')
            # Publicamos el resultado
            self.status_pub.publish("DONE" if result else "FAILED")
        except Exception as e:
            rospy.logerr(f"Error al ejecutar el movimiento: {e}")
            self.status_pub.publish("FAILED")
    
    def getcurrentJointCommand(self,):
        # wait to get current position
        topic_address = '/' + self.prefix + 'driver/out/joint_command'
        rospy.Subscriber(topic_address, kinova_msgs.msg.JointAngles, self.setcurrentJointCommand)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.JointAngles)
        rospy.loginfo('position listener obtained message for joint position. ')
    
    def setcurrentJointCommand(self,feedback):
        currentJointCommand_str_list = str(feedback).split("\n")
        for index in range(0,len(currentJointCommand_str_list)):
            temp_str=currentJointCommand_str_list[index].split(": ")
            self.currentJointCommand[index] = float(temp_str[1])

    def unitParser(self,unit, joint_value, relative_):
        """ Argument unit """
        global currentJointCommand

        if unit == 'degree':
            joint_degree_command = joint_value
            # get absolute value
            if relative_:
                joint_degree_absolute_ = [joint_degree_command[i] + currentJointCommand[i] for i in range(0, len(joint_value))]
            else:
                joint_degree_absolute_ = joint_degree_command
            joint_degree = joint_degree_absolute_
            joint_radian = list(map(math.radians, joint_degree_absolute_))
        elif unit == 'radian':
            joint_degree_command = list(map(math.degrees, joint_value))
            # get absolute value
            if relative_:
                joint_degree_absolute_ = [joint_degree_command[i] + currentJointCommand[i] for i in range(0, len(joint_value))]
            else:
                joint_degree_absolute_ = joint_degree_command
            joint_degree = joint_degree_absolute_
            joint_radian = list(map(math.radians, joint_degree_absolute_))
        else:
            raise Exception("Joint value have to be in degree, or radian")

        return joint_degree, joint_radian

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

        rospy.loginfo(f"Enviando joint angles: {angle_set} al action server en {action_address}")
        goal = kinova_msgs.msg.ArmJointAnglesGoal()

        goal.angles.joint1 = angle_set[0]
        goal.angles.joint2 = angle_set[1]
        goal.angles.joint3 = angle_set[2]
        goal.angles.joint4 = angle_set[3]
        goal.angles.joint5 = angle_set[4]
        goal.angles.joint6 = angle_set[5]
        goal.angles.joint7 = angle_set[6]

        client.send_goal(goal)
        rospy.loginfo(f"Joint angles enviados al action server: {goal.angles}")
        if client.wait_for_result(rospy.Duration(20.0)):
            return client.get_result()
        else:
            rospy.logerr('the joint angle action timed-out')
            client.cancel_all_goals()
            return None
    
    def QuaternionNorm(self,Q_raw):
        qx_temp,qy_temp,qz_temp,qw_temp = Q_raw[0:4]
        qnorm = math.sqrt(qx_temp*qx_temp + qy_temp*qy_temp + qz_temp*qz_temp + qw_temp*qw_temp)
        qx_ = qx_temp/qnorm
        qy_ = qy_temp/qnorm
        qz_ = qz_temp/qnorm
        qw_ = qw_temp/qnorm
        Q_normed_ = [qx_, qy_, qz_, qw_]
        return Q_normed_


    def Quaternion2EulerXYZ(self,Q_raw):
        Q_normed = self.QuaternionNorm(Q_raw)
        qx_ = Q_normed[0]
        qy_ = Q_normed[1]
        qz_ = Q_normed[2]
        qw_ = Q_normed[3]

        tx_ = math.atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_))
        ty_ = math.asin(2 * qw_ * qy_ + 2 * qx_ * qz_)
        tz_ = math.atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_))
        EulerXYZ_ = [tx_,ty_,tz_]
        return EulerXYZ_


    def EulerXYZ2Quaternion(self,EulerXYZ_):
        tx_, ty_, tz_ = EulerXYZ_[0:3]
        sx = math.sin(0.5 * tx_)
        cx = math.cos(0.5 * tx_)
        sy = math.sin(0.5 * ty_)
        cy = math.cos(0.5 * ty_)
        sz = math.sin(0.5 * tz_)
        cz = math.cos(0.5 * tz_)

        qx_ = sx * cy * cz + cx * sy * sz
        qy_ = -sx * cy * sz + cx * sy * cz
        qz_ = sx * sy * cz + cx * cy * sz
        qw_ = -sx * sy * sz + cx * cy * cz

        Q_ = [qx_, qy_, qz_, qw_]
        return Q_
        
if __name__ == "__main__":
    try:
        controller = MoveNodeKinova()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Error al iniciar el nodo de movimiento del brazo")