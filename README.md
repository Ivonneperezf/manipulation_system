# REPOSITORIO PAQUETES DE SIMULACION

En este repositorio ya estan integrados todos los archivos para realizar la simulacion usando Gazebo, ademas incluye la montura de la cámara en la muñeca del mismo.

## Comandos para simulación 

Los comandos se indican a continuación:

#### Comando para lanzar Gazebo

``` bash
roslaunch gazebo_sim gazebo_kinova_sim.launch
```

#### Comando para lanzar Rviz

``` bash
roslaunch gazebo_sim rviz_kinova_sim.launch
```

#### Comando para lanzar nodo de movimiento

``` bash
roslaunch statemachine move_node_gazebo.launch
```

#### Comando para lanzar segmentacion

``` bash
roslaunch statemachine segmentation_gazebo.launch
```

#### Comando para lanzar maquina de estados

``` bash
roslaunch statemachine statemachine_node_gazebo.launch
```

Los archivos de Moveit! sirven para la implementación del brazo fisico, aunque ya existen algunos elementos que deben de ajustarse.

Las pruebas para ver SAM3 se deben de lanzar mediante el siguiente ccomando:

``` bash
roslaunch statemachine segmentation_SAM3_gazebo.launch
```

En caso de lanzar cargando los parametros de simulacion, no se debe de modificar ningun parametro, en caso contrario ejecutar el siguiente comando:
``` bash
roslaunch statemachine segmentation_SAM3_gazebo.launch use_simulation_params:=false
```

De igual manera para SAM 2
``` bash
roslaunch statemachine segmentation_gazebo.launch use_simulation_params:=false
```

Finalmente, para implementar el nodo de movimiento para el brazo fisico ejecutar el siguiente comando:
``` bash
roslaunch statemachine move_node_gazebo.launch use_sim:=false
```