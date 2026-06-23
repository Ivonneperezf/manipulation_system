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

class KinovaTelemetryNode:

    def __init__(self):
        rospy.init_node('kinova_telemetry_node', anonymous=True)

        # Carga de parametros
        self.kinova_robotType = rospy.get_param("~kinova_robotType", "m1n6s300")
        self.prefix = self.kinova_robotType + "_"

        # Estructuras para almacenar los datos actuales
        self.current_joints = [0.0] * 7
        self.current_cartesian = [0.0] * 7

        # Publicadores
        self.joints_pub = rospy.Publisher('/kinova_current_joints', Float64MultiArray, queue_size=10)
        self.cartesian_pub = rospy.Publisher('/kinova_current_cartesian', Float64MultiArray, queue_size=10)

        # Suscriptores
        rospy.Subscriber('/' + self.prefix + 'driver/out/joint_command', kinova_msgs.msg.JointAngles, self.joints_callback)
        rospy.Subscriber('/' + self.prefix + 'driver/out/cartesian_command', kinova_msgs.msg.KinovaPose, self.cartesian_callback)

        # Informacion de inicio
        rospy.loginfo(f"Nodo de telemetria Kinova iniciado para el robot: {self.kinova_robotType}")

    # Callback para recibir los datos de las articulaciones del brazo
    def joints_callback(self, msg):
        msg_str_list = str(msg).split("\n")
        for index in range(0, min(len(msg_str_list), len(self.current_joints))):
            temp_str = msg_str_list[index].split(": ")
            if len(temp_str) > 1:
                self.current_joints[index] = float(temp_str[1])

    # Callback para recibir los datos de la pose cartesiana del brazo
    def cartesian_callback(self, msg):
        msg_str_list = str(msg).split("\n")
        for index in range(0, min(len(msg_str_list), len(self.current_cartesian))):
            temp_str = msg_str_list[index].split(": ")
            if len(temp_str) > 1:
                self.current_cartesian[index] = float(temp_str[1])

    # Mantiene el bucle activo publicando a la frecuencia indicada (Hz)
    def start_publishing(self, frequency=10):
        rate = rospy.Rate(frequency)
        
        while not rospy.is_shutdown():
            # Publicar Articulaciones
            joints_msg = Float64MultiArray()
            joints_msg.data = self.current_joints
            self.joints_pub.publish(joints_msg)
            # Publicar Cartesianas
            cartesian_msg = Float64MultiArray()
            cartesian_msg.data = self.current_cartesian
            self.cartesian_pub.publish(cartesian_msg)
            # Esperar hasta la siguiente iteración
            rate.sleep()
    
def main():
    try:
        telemetry = KinovaTelemetryNode()
        telemetry.start_publishing(frequency=10)
    except rospy.ROSInterruptException:
        rospy.loginfo("Nodo de telemetria Kinova interrumpido.")