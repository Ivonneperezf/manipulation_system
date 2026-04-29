#!/usr/bin/env python3

import rospy
import sys
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, fabs, cos, sqrt
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String

# Las bases escenciales, como rectificación de errores, funciones de distancia, etc, son provenientes de los códigos de ejemplo 
# presentados en los tutoriales de MoveIt en el siguiente repositorio: https://github.com/moveit/moveit_tutorials/blob/master/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py

"""==================================================================
        FUNCIONES Y VARIABLES GLOBALES PARA CONTROL GENERAL
====================================================================="""

# Función para calcular la distancia entre dos puntos en el espacio usando la fórmula de distancia euclidiana.
def dist(p, q):
    return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

# Función para verificar si el estado actual del robot está dentro de una tolerancia aceptable con respecto a un objetivo deseado, el valor de tolerancia
# se define por el usuario
def all_close(goal, actual, tolerance):
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)
    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        d = dist((x1, y1, z1), (x0, y0, z0))
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)
    return True

# VARIABLES GLOBALES PARA CONTROLAR LOS PARÁMETROS DE MOVIMIENTO DEL BRAZO
PLANNING_TIME = 10.0 # Tiempo máximo de planificación de movimientos
PLANNING_ATTEMPTS = 5 # Máximo de intentos para encontrar un plan de movimiento exitoso
VELOCITY_SCALING = 0.5 # Escalado de velocidad para movimientos más suaves
ACCELERATION_SCALING = 0.2 # Escalado de aceleración para movimientos más suaves
WORKSPACE = [-1.0, -1.0, -0.2, 1.0, 1.0, 1.5] # Tamaño del workspace definido como un cubo con límites en x, y, z
OFFSET = 0.05 # Offset para determinar por ejemplo el tamaño del tenedor (5 cm)
TOLERANCE = 0.01 # Tolerancia para considerar que el robot ha alcanzado la posición objetivo (1 cm)

"""==================================================================
            CLASE PARA CONTROLAR EL BRAZO CON MOVEIT
====================================================================="""

# Clase para manejar los nodos que usen movimiento del brazo
class MoveGroupController():
    def __init__(self):

        # Inicialización de MoveIt y configuración del entorno de planificación
        moveit_commander.roscpp_initialize(sys.argv) 
        rospy.init_node("kinova_moveit", anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("arm")

        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        # Setters para configurar los parámetros de planificación, velocidad y workspace del robot
        self.move_group.set_planning_time(PLANNING_TIME)
        self.move_group.set_num_planning_attempts(PLANNING_ATTEMPTS)
        self.move_group.set_max_velocity_scaling_factor(VELOCITY_SCALING)
        self.move_group.set_max_acceleration_scaling_factor(ACCELERATION_SCALING)
        self.move_group.set_workspace(WORKSPACE)

        # Getters para obtener información del robot
        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link       = self.move_group.get_end_effector_link()
        self.group_names    = self.robot.get_group_names()

        # Información actual del robot
        print("Frame de referencia:", self.planning_frame)
        print("End effector:",        self.eef_link)
        print("Grupos disponibles:",  self.group_names)
        print("Estado actual del robot:")
        print(self.robot.get_current_state())

        self.point_msg = None # Almacena PointStamped cada que se recibe un punto
        self.joint_goal = None # Almacena la posición articular cada que se recibe un objetivo articular
        rospy.loginfo("Esperando mensajes para movimiento en los tópicos /cartesian_goal y /joint_goal ...")

        # Publishers
        self.status_pub = rospy.Publisher('/motion_done', String, queue_size=10)

        # Subscribers
        rospy.Subscriber('/cartesian_goal', PointStamped, self._cartesian_callback)
        rospy.Subscriber('/joint_goal', Float64MultiArray, self._joint_callback)
    
    # Funciones de callback para recibir y confirmar los objetivos de movimiento
    def _cartesian_callback(self, msg):
        print(f"Mensaje recibido \n{msg} \nEJECUTANDO MOVIMIENTO")
        self.point_msg = msg
        self.status_pub.publish( "DONE" if self.go_to_cartesian_goal() else "FAILED")

    def _joint_callback(self, msg):
        self.joint_goal = msg.data
        self.status_pub.publish( "DONE" if self.go_to_joint_goal() else "FAILED")

    # Constriye el punto recibido PointStamped a punto para MoveIt
    # (MANTIENE LA POSE ACTUAL DEL END EFFECTOR, ES PROVISIONAL, PUEDE MODIFICARSE)
    def _build_pose_from_point(self):
        current_pose = self.move_group.get_current_pose().pose
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = self.point_msg.point.x
        pose_goal.position.y = self.point_msg.point.y
        pose_goal.position.z = self.point_msg.point.z + OFFSET
        pose_goal.orientation = current_pose.orientation
        return pose_goal
    
    # Realiza el movimiento cartesiano del end effector, devuelve True si el movimiento se realizó con éxito, False en caso contrario
    def go_to_cartesian_goal(self):
        rospy.loginfo(f"Moviendo a -> X:{self.point_msg.point.x:.3f}  Y:{self.point_msg.point.y:.3f} Z:{self.point_msg.point.z:.3f}")
        pose_goal = self._build_pose_from_point()
        self.move_group.set_pose_target(pose_goal) # Construye la pose objetivo a partir del punto recibido (resuelve cinemática inversa)
        success = self.move_group.go(wait=True) # Ejecuta el movimiento hacia la pose objetivo
        self.move_group.stop() # Detiene cualquier movimiento residual
        self.move_group.clear_pose_targets() # Limpia los objetivos de pose para evitar interferencias
        if not success:
            rospy.logwarn("go_to_pose_goal: el planificador no encontró solución")
            return False
        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, TOLERANCE)
    
    # Realiza el movimiento articular del brazo robótico con valores en radianes, devuelve True si el movimiento se realizó con éxito, False en caso contrario
    def go_to_joint_goal(self):
        current_joint_values = self.move_group.get_current_joint_values()
        for i in range (len(self.joint_goal)):
            current_joint_values[i] = self.joint_goal[i]
        success = self.move_group.go(current_joint_values, wait=True)
        self.move_group.stop()
        if not success:
            rospy.logwarn("El planificador no encontró solución")
            return False
        current_joints = self.move_group.get_current_joint_values()
        return all_close(self.joint_goal, current_joints, TOLERANCE)

if __name__ == "__main__":
    try:
        controller = MoveGroupController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Error al iniciar el nodo de movimiento del brazo")