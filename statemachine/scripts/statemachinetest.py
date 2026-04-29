#!/usr/bin/env python3

# POR EL MOMENTO LA MAQUINA DE ESTADOS SE QUEDA HASTA AQUI

import rospy
import smach
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String

# Estado HOME
class Home(smach.State):
    def __init__(self):
        # Transiciones posibles -> Done: movimiento completado, Failed: movimiento fallido
        smach.State.__init__(self, outcomes=['Done', 'Failed']) # PERSONAL: DEFINIR QUE HACER EN CASO DE FALLO

        # Publicación de tópicos
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        self.joint_position = rospy.Publisher('/joint_goal', Float64MultiArray, queue_size=10)

        # Definicion de pose HOME cartesiana
        self.home_pose = PointStamped()
        self.home_pose.point.x = 0.434
        self.home_pose.point.y = -0.002
        self.home_pose.point.z = 0.362

        # Definición de pose HOME articular (en radianes)
        self.home_joint_goal = Float64MultiArray()
        self.home_joint_goal.data = [-3.1917, 3.8806, 2.9837, -1.4455, 3.1411, -2.4153]

    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: HOME")

        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.joint_position.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        
        self.joint_position.publish(self.home_joint_goal)

        # Esperamos el mensaje de confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'

# Estado ESPERAR_PUNTO
class Esperar_Punto(smach.State):
    def __init__ (self):
        smach.State.__init__(self, outcomes=['received_point'],
                             input_keys=['point_received'],
                             output_keys=['point_received']) 
    
    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: ESPERAR_PUNTO")
        point_robot = rospy.wait_for_message('/object_centroid_robot', PointStamped)
        userdata.point_received = point_robot
        return 'received_point'
    
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
