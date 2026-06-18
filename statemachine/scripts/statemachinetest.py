#!/usr/bin/env python3

import rospy
import smach
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String, Bool

SLEEP = 2.0

"""Estado HOME"""
class Home(smach.State):
    def __init__(self):
        # Transiciones posibles -> Done: movimiento completado, Failed: movimiento fallido
        smach.State.__init__(self, outcomes=['Done', 'Failed'])

        # Publicación de tópicos
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        self.joint_position = rospy.Publisher('/joint_goal', Float64MultiArray, queue_size=10)

        # Parametro para usar pose de simulacion o real
        self.use_sim = rospy.get_param('~use_sim', False)

        # Definicion de pose HOME cartesiana
        # self.home_pose = PointStamped()
        # self.home_pose.point.x = -2.943
        # self.home_pose.point.y = -0.027
        # self.home_pose.point.z = -2.638

        # Definición de pose HOME articular (en radianes)
        # Simulacion
        self.home_joint_goal_simulation = Float64MultiArray()
        self.home_joint_goal_simulation.data = [-3.1917, 3.8806, 2.9837, -1.4455, 3.1411, -2.4153]
        # Brazo real
        self.home_joint_goal = Float64MultiArray()
        self.home_joint_goal.data = [4.544290785011106, 3.400400246220348, 2.1462572351452898, 5.563524612129654, 2.070292828737706, 8.390330026975427]

    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: HOME")

        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.joint_position.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        
        # Enviamos la pose de movimiento
        self.joint_position.publish(self.home_joint_goal_simulation if self.use_sim else self.home_joint_goal)

        # Esperamos el mensaje de confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.sleep(SLEEP)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'

"""Estado ESPERAR_PUNTO"""
class Esperar_Punto(smach.State):
    def __init__ (self):
        # Bandera para indicar que debe iniciar la segmentacion
        self.segmentation_flag = rospy.Publisher('/segmentation_flag', Bool, queue_size=10)
        smach.State.__init__(self, outcomes=['received_point'],
                             input_keys=['point_received'],
                             output_keys=['point_received']) 
    
    def execute(self, userdata):
        # Publicamos la bandera para iniciar la segmentacion para que se comience la segmentacion
        rospy.loginfo("Ejecutando estado: ESPERAR_PUNTO")
        self.segmentation_flag.publish(Bool(data=True))
        # Esperamos el centroide del objeto detectado en el tópico correspondiente
        point_robot = rospy.wait_for_message('/object_centroid_robot', PointStamped)
        userdata.point_received = point_robot
        self.segmentation_flag.publish(Bool(data=False))
        rospy.sleep(SLEEP)
        return 'received_point'

"""Estado MOVER_A_PUNTO"""
class Mover_A_Punto(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Done', 'Failed'],
                             input_keys=['point_to_move']) 
        # Nodo publicador en el tópico de movimiento
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        
    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: MOVER_A_PUNTO")
        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.cartesian_point.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        # Se envia el punto recibido del centro del objeto al nodo de movimiento
        self.cartesian_point.publish(userdata.point_to_move)

        # Esoeramos a que se reciba la confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.sleep(SLEEP)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'

def main():
    rospy.init_node('state_machine_')

    # Creamos la maquina de estados
    sm = smach.StateMachine(outcomes=['END'])

    # Estados y transiciones
    with sm:
        # Estado HOME
        smach.StateMachine.add('HOME', Home(), 
                               transitions={'Done':'ESPERAR_PUNTO', 'Failed':'HOME'})

        # Estado ESPERAR_PUNTO
        smach.StateMachine.add('ESPERAR_PUNTO', Esperar_Punto(), 
                               transitions={'received_point':'MOVER_A_PUNTO'},
                               remapping={'point_received':'shared_point'}) 

        # Estado MOVER_A_PUNTO
        smach.StateMachine.add('MOVER_A_PUNTO', Mover_A_Punto(), 
                               transitions={'Done':'HOME', 'Failed':'HOME'},
                               remapping={'point_to_move':'shared_point'})

    # Ejecutamos la maquina de estados
    outcome = sm.execute()

if __name__ == '__main__':
    main()
