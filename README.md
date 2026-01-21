# Systèmes intelligents autonomes

Pour toutes les informations sur le projet "Edge Computing et IA Embarquée", le contenu du projet se trouve dans le dossier `Benchmarking/`

## Description du projet

Le projet vise à développer un modèle IA sur turtlebot3 a destination embarqué, plus précisement, notre modèle détecte les balles avec la Intel D435i, et le robot doit s'y diriger pour la pousser.

### Liste des packages

- ball_detection: Le modèle IA qui détecte les balles
- coordinator : l'orchestrateur de notre robot
- realsense_publisher : le package qui permet de publier sur deux topics l'image rgb et l'image profondeur de la D435i
- turtlebot3_wanderer : le package qui contrôle le déplacement du robot
- turtlebot3_descriptions : les descriptions du robot pour la simulation
- turtlebot3_gazebo : les fichiers de mondes et les modèles pour la simulation

## Démarrer le projet

### Pré-requis

- Ubuntu 22.04 (Jammy)
- Python3 avec pip

### Installer ROS2 Humble + Gazebo

- https://foxglove.dev/blog/installing-ros2-humble-on-ubuntu (installer ros2 desktop version)
- Aussi installer `colcon`, le compilateur ROS2
```bash
sudo apt install python3-colcon-common-extensions
```
- Et cyclone DDS
```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
```
- Ensuite
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
source .bashrc
printenv | grep -i ROS
```
- Pour gazebo
```bash
sudo apt install ros-humble-ros-gz
```

### Installer les dépendances python

Installer les dépendances pour les fichiers python
```bash
cd src/
pip -r install requirements.txt
```

### Démarrer la simulation

Lancer la commande 
```bash
./run.sh
```
Le robot devrait commencer a bouger après 20 secondes, ce qui permet de mettre en place la visualisation dans RVIZ

Pour voir les informations importantes, allez sur `Add > By topic > /ball_debug > Image` pour afficher l'image avec la bounding box, puis `Add > By topic > /ball_marker > Marker` pour voir le marqueur de la balle dans le champ de vision du robot.



## Démonstration du projet

[demo.webm](https://github.com/user-attachments/assets/fa0be8ee-7a7e-4fdc-a6dd-702e5762a50f)

