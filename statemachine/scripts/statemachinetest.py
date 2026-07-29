#!/usr/bin/env python3

import rospy
import smach
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray, String, Bool

SLEEP = 6.0
TENEDOR = 0.1
OFFSET_Z = 0.05 # Distancia a subir o bajar en Z para evitar colisiones con el objeto
OFFSET = TENEDOR+ OFFSET_Z

"""
==========================================================================
Estado HOME
==========================================================================
"""
class Home(smach.State):
    def __init__(self):
        # Transiciones posibles -> Done: movimiento completado, Failed: movimiento fallido
        smach.State.__init__(self, outcomes=['Done', 'Failed'])

        # Publicación de tópicos (El topico catesian_goal es definido por si se requiere movimiento cartesiano)
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        self.joint_position = rospy.Publisher('/joint_goal', Float64MultiArray, queue_size=10)

        # Parametro para usar pose de simulacion o real
        self.use_sim = rospy.get_param('~use_sim', False)

        # Definición de pose HOME articular (en radianes)
        """Simulacion"""
        self.home_joint_goal_simulation = Float64MultiArray()
        self.home_joint_goal_simulation.data = [-3.1917, 3.8806, 2.9837, -1.4455, 3.1411, -2.4153]
        """Brazo real"""
        self.home_joint_goal = Float64MultiArray()
        self.home_joint_goal.data = [4.368214025327754, 3.617058520910436, 2.2687591831795175, 5.649283725552385, 2.0857627320358127, 7.0008326524454185]

    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: HOME")
        # Espera 0.1 segundos hasta que haya al menos un suscriptor conectado al topico
        while self.joint_position.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        
        # Enviamos la pose de movimiento
        self.joint_position.publish(self.home_joint_goal_simulation if self.use_sim else self.home_joint_goal)

        # Esperamos el mensaje de confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.sleep(6)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'

"""
==========================================================================
Estado ESPERAR_PUNTO
==========================================================================
"""
class Esperar_Punto(smach.State):
    def __init__ (self):
        # Definimos entradas y salidas del estado
        smach.State.__init__(self, outcomes=['received_point'],
                             output_keys=['point']) 
        # Bandera para indicar que debe iniciar la segmentacion
        #self.segmentation_flag = rospy.Publisher('/segmentation_flag', Bool, queue_size=10)
    
    def execute(self, userdata):
        # Publicamos la bandera para iniciar la segmentacion para que se comience la segmentacion
        rospy.loginfo("Ejecutando estado: ESPERAR_PUNTO")
        # Indica segmentacion activa
        #self.segmentation_flag.publish(Bool(data=True))
        # Esperamos el centroide del objeto detectado en el tópico correspondiente
        rospy.sleep(5)
        point = rospy.wait_for_message('/object_centroid', PointStamped)
        userdata.point = point
        rospy.loginfo(f"Punto recibido x={point.point.x:.3f} y={point.point.y:.3f} z={point.point.z:.3f}")
        # Indica segmentacion inactiva
        #self.segmentation_flag.publish(Bool(data=False))
        return 'received_point'

"""
==========================================================================
Estado TRANSFORMAR_PUNTO
==========================================================================
"""
class Transformar_Punto(smach.State):
    def __init__ (self):
        # Definimos entradas y salidas del estado
        smach.State.__init__(self, outcomes=['received_point'],
                             input_keys=['point'],
                             output_keys=['transform_point']) 
        self.point_pub = rospy.Publisher('/object_centroid_sm', PointStamped, queue_size=10)
    
    def execute(self, userdata):
        # Publicamos la bandera para iniciar la segmentacion para que se comience la segmentacion
        rospy.loginfo("Ejecutando estado: TRANSFORMAR_PUNTO")
        # Publicamos el centroide
        self.point_pub.publish(userdata.point)
        # Esperamos a que se realice la transformacion
        point_robot = rospy.wait_for_message('/object_centroid_robot', PointStamped)
        rospy.loginfo(f"Punto a mover x={point_robot.point.x:.3f} y={point_robot.point.y:.3f} z={point_robot.point.z:.3f}")
        userdata.transform_point = point_robot
        rospy.sleep(2)
        return 'received_point'

"""
==========================================================================
Estado MOVER_A_CENTROIDE
==========================================================================
"""
class Mover_A_Centroide(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Done', 'Failed'],
                             input_keys=['transform_point']) 
        # Nodo publicador en el tópico de movimiento
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        
    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: MOVER_A_CENTROIDE")
        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.cartesian_point.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        # Definimos el nuevo punto a mover
        new_point = PointStamped()
        new_point.header = userdata.transform_point.header
        new_point.point = userdata.transform_point.point
        new_point.point.z = new_point.point.z + OFFSET
        rospy.loginfo(f"Punto a mover centroide: X:{new_point.point.x:.3f} Y:{new_point.point.y:.3f} Z:{new_point.point.z:.3f}")
        rospy.sleep(10)
        # Se envia el punto recibido del centro del objeto al nodo de movimiento
        self.cartesian_point.publish(new_point)

        # Esperamos a que se reciba la confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.loginfo(f"============\nStatus: {status_msg}\n============")
        rospy.sleep(2)
        #rospy.sleep(10)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'
        
"""
==========================================================================
Estado BAJAR_EN_Z
==========================================================================
"""
class Bajar_En_Z(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Done', 'Failed']) 
        # Nodo publicador en el tópico de movimiento
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        
    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: BAJAR_EN_Z")
        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.cartesian_point.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'

        # Definimos el nuevo punto a mover, bajando en Z
        actual_point = rospy.wait_for_message('/kinova_current_cartesian', Float64MultiArray)
        new_point = PointStamped()
        new_point.header.stamp = rospy.Time.now()
        new_point.header.frame_id = "m1n6s300_link_base"
        new_point.point.x = actual_point.data[0]
        new_point.point.y = actual_point.data[1]
        new_point.point.z = actual_point.data[2] - OFFSET_Z
        rospy.loginfo(f"Punto a mover bajada en Z: X:{new_point.point.x:.3f} Y:{new_point.point.y:.3f} Z:{new_point.point.z:.3f}")
        #rospy.sleep(10)
        # Se envia el punto recibido del centro del objeto al nodo de movimiento
        self.cartesian_point.publish(new_point)

        # Esperamos a que se reciba la confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.sleep(2)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'
    
"""
==========================================================================
Estado SUBIR_EN_Z
==========================================================================
"""
class Subir_En_Z(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['Done', 'Failed']) 
        # Nodo publicador en el tópico de movimiento
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)


    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: SUBIR_EN_Z")
        # Espera hasta que haya al menos un suscriptor conectado al topico
        while self.cartesian_point.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'

        # Definimos el nuevo punto a mover, subiendo en Z
        actual_point = rospy.wait_for_message('/kinova_current_cartesian', Float64MultiArray)
        new_point = PointStamped()
        new_point.header.stamp = rospy.Time.now()
        new_point.header.frame_id = "m1n6s300_link_base"
        new_point.point.x = actual_point.data[0]
        new_point.point.y = actual_point.data[1]
        new_point.point.z = actual_point.data[2] + OFFSET_Z  # Subimos la distancia definida en OFFSET_Z

        rospy.loginfo(f"Punto a mover subida en Z: X:{new_point.point.x:.3f} Y:{new_point.point.y:.3f} Z:{new_point.point.z:.3f}")
        # Se envia el punto recibido del centro del objeto al nodo de movimiento
        self.cartesian_point.publish(new_point)

        # Esperamos a que se reciba la confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.sleep(2)
        if status_msg.data == "DONE":
            return 'Done'
        else:
            return 'Failed'

"""
==========================================================================
Estado ALIMENTAR
==========================================================================
"""
class Alimentar(smach.State):
    def __init__(self):
        # Transiciones posibles -> Done: movimiento completado, Failed: movimiento fallido
        smach.State.__init__(self, outcomes=['Done', 'Failed'])

        # Publicación de tópicos (El topico catesian_goal es definido por si se requiere movimiento cartesiano)
        self.cartesian_point = rospy.Publisher('/cartesian_goal', PointStamped, queue_size=10)
        self.joint_position = rospy.Publisher('/joint_goal', Float64MultiArray, queue_size=10)
        # Definir una pose que simula alimentar
        self.joint_goal = Float64MultiArray()
        self.joint_goal.data = [3.9767362693049404, 3.8982113666673, 1.890399359738107, 6.383132677625824, 2.4971944888421542, 6.281265700665857]

    def execute(self, userdata):
        rospy.loginfo("Ejecutando estado: ALIMENTAR")
        # Espera 0.1 segundos hasta que haya al menos un suscriptor conectado al topico
        while self.joint_position.get_num_connections() == 0:
            rospy.sleep(0.1)
            # Si el nodo esta apagado retorna error
            if rospy.is_shutdown(): return 'Failed'
        
        # Enviamos la pose de movimiento
        self.joint_position.publish(self.joint_goal)

        # Esperamos el mensaje de confirmacion de movimiento
        status_msg = rospy.wait_for_message('/motion_done', String)
        rospy.sleep(5)
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
                               transitions={'received_point':'TRANSFORMAR_PUNTO'},
                               remapping={'point':'shared_point'}) # Punto del centroide
        
        # Estado TRANSFORMAR_PUNTO
        smach.StateMachine.add('TRANSFORMAR_PUNTO', Transformar_Punto(),
                               transitions={'received_point':'MOVER_A_CENTROIDE'},
                               remapping={'point':'shared_point', 'transform_point':'transformed_point'})
                               #                Punto de centroide     Punto transformado
        # Estado MOVER_A_CENTROIDE
        smach.StateMachine.add('MOVER_A_CENTROIDE', Mover_A_Centroide(), 
                               transitions={'Done':'BAJAR_EN_Z', 'Failed':'HOME'},
                               remapping={'transform_point':'transformed_point'})
                               #               Punto transformado
        # Estado BAJAR_EN_Z
        smach.StateMachine.add('BAJAR_EN_Z', Bajar_En_Z(),
                               transitions={'Done':'SUBIR_EN_Z', 'Failed':'HOME'})
        
        # Estado SUBIR_EN_Z
        smach.StateMachine.add('SUBIR_EN_Z', Subir_En_Z(),
                               transitions={'Done':'ALIMENTAR', 'Failed':'HOME'})
        
        # Estado ALIMENTAR
        smach.StateMachine.add('ALIMENTAR', Alimentar(),
                               transitions={'Done':'HOME', 'Failed':'HOME'})
        

    # Ejecutamos la maquina de estados
    outcome = sm.execute()

if __name__ == '__main__':
    main()
