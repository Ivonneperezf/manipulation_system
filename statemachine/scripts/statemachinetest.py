#!/usr/bin/env python3

# POR EL MOMENTO LA MAQUINA DE ESTADOS SE QUEDA HASTA AQUI

import rospy
import smach
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String, Bool, Float64

SLEEP = 2.0

# Estado HOME
class Home(smach.State):
    def __init__(self):
        # Transiciones posibles -> Done: movimiento completado, Failed: movimiento fallido
        smach.State.__init__(self, outcomes=['Done', 'Failed'])

        # Publicación de tópicos
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        self.joint_position = rospy.Publisher('/joint_goal', Float64MultiArray, queue_size=10)

        # Parametro para usar pose de simulacion o real
        self.use_sim = rospy.get_param('~use_sim', True)

        # Definicion de pose HOME cartesiana
        self.home_pose = PointStamped()
        self.home_pose.point.x = -2.943
        self.home_pose.point.y = -0.027
        self.home_pose.point.z = -2.638

        # Definición de pose HOME articular (en radianes)
        self.home_joint_goal_simulation = Float64MultiArray()
        self.home_joint_goal_simulation.data = [-3.1917, 3.8806, 2.9837, -1.4455, 3.1411, -2.4153]
        self.home_joint_goal = Float64MultiArray()
        self.home_joint_goal.data = [4.4802891650621035, 3.3580727628029554, 2.0120645425578316, 5.45262632328101, 2.0693458086540732, 7.871249037850339]
        #self.home_joint_goal.data = [4.06356970191141, 3.3038859567353307, 2.259471675193975, 5.08504457908726, 2.563036086788292, 2.0200945058766386]
        #self.home_joint_goal.data = [4.094269291377299, 3.3711478184908077, 2.1719462201782482, 5.457468482774992, 2.170500523180499, 2.1681103361021528]

    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: HOME")

        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.joint_position.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        
        if self.use_sim:
            self.joint_position.publish(self.home_joint_goal_simulation)
        else:
            self.joint_position.publish(self.home_joint_goal)

        # Esperamos el mensaje de confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.sleep(SLEEP)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'

# Estado ESPERAR_PUNTO
class Esperar_Punto(smach.State):
    def __init__ (self):
        # Bandera para indicar que debe iniciar la segmentacion
        smach.State.__init__(self, outcomes=['received_point'],
                             input_keys=['point_received'],
                             output_keys=['point_received']) 
    
    def execute(self, userdata):
        # Publicamos la bandera para iniciar la segmentacion para que se comience la segmentacion
        rospy.loginfo("Ejecutando estado: ESPERAR_PUNTO")
        # Esperamos el centroide del objeto detectado en el tópico correspondiente
        point_robot = rospy.wait_for_message('/object_centroid_robot', PointStamped)
        userdata.point_received = point_robot
        rospy.sleep(SLEEP)
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
        rospy.sleep(SLEEP)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'

class Bajar_Z(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Done', 'Failed'],
                             input_keys=['point_to_move']) 
        # Nodo publicador en el tópico de movimiento
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        self.OFFSET = 0.18
        
    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: BAJAR_Z")
        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.cartesian_point.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        rospy.loginfo("Esperando la lectura del valor de Z desde /z_value...")
        try:
            # Escuchamos el mensaje Float64 del nodo que lee la altura de la nube
            z_superficie_msg = rospy.wait_for_message('/z_value', Float64, timeout=3.0)
            z_superficie = z_superficie_msg.data
            rospy.loginfo(f"Z de la superficie detectada: {z_superficie}")
        except rospy.ROSException:
            rospy.logerr("Timeout: No se recibió respuesta en el tópico /z_value")
            return 'Failed'
        
        goal_pose = PointStamped()
        goal_pose.header = userdata.point_to_move.header
        goal_pose.header.stamp = rospy.Time.now()
        
        # Mantenemos las componentes X e Y intactas del objetivo inicial
        goal_pose.point.x = userdata.point_to_move.point.x
        goal_pose.point.y = userdata.point_to_move.point.y
        
        # Asignamos la nueva coordenada Z aplicando la resta con tu OFFSET
        goal_pose.point.z = z_superficie + self.OFFSET
        rospy.loginfo(f"Publicando punto con Z modificado: X={goal_pose.point.x}, Y={goal_pose.point.y}, Z={goal_pose.point.z}")
        
        # Se envia el punto recibido del centro del objeto al nodo de movimiento
        self.cartesian_point.publish(goal_pose)

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
                               transitions={'Done':'ESPERAR_PUNTO', 'Failed':'ESPERAR_PUNTO'})

        # Estado ESPERAR_PUNTO
        smach.StateMachine.add('ESPERAR_PUNTO', Esperar_Punto(), 
                               transitions={'received_point':'MOVER_A_PUNTO'},
                               remapping={'point_received':'shared_point'}) 

        # Estado MOVER_A_PUNTO (Cambias la transición para que vaya a BAJAR_Z en lugar de HOME)
        smach.StateMachine.add('MOVER_A_PUNTO', Mover_A_Punto(), 
                               transitions={'Done':'BAJAR_Z', 'Failed':'HOME'},
                               remapping={'point_to_move':'shared_point'})

        # Estado BAJAR_Z (Agregado a la estructura principal)
        smach.StateMachine.add('BAJAR_Z', Bajar_Z(),
                               transitions={'Done':'HOME', 'Failed':'HOME'},
                               remapping={'point_to_move':'shared_point'})
    # Ejecutamos la maquina de estados
    outcome = sm.execute()

if __name__ == '__main__':
    main()
