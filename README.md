# ia-ft-iot-control-robot

ajouter ca au .bashrc
`source <path>/ia-ft-iot-control-robot/ros2_packages/install/setup.bash`

puis faire 
`source .bashrc`

---
## notes supplémentaires

Dans ros2_packages/tests c'est juste moi qui joue avec la lib pour comprendre quoi faire avant de le mettre en package ros

## lancer la simu

faire
```
export PROJECT_MODEL=turtlebot3_burger_d435i
colcon build
source install/setup.bash
```
pour setup le projet

puis
```
ros2 launch turtlebot3_gazebo projects_house_world.launch.py
```
qui lance :
- gazebo sur un modele de maison
- rviz
- le modele IA

puis il faut lancer, dans 2 autres terminaux 
```
ros2 run turtlebot3_wanderer explorator
ros2 run coordinator coordinator
```