# TurtleBot3 Wanderer

## Objectif 
Comportement d'exploration / recherche de balle pour TurtleBot3. Le noeud écoute la position cible fournie par le `coordinator` et publie des commandes de vitesse (`/cmd_vel`) pour approcher la cible tout en évitant les obstacles.

## Topics publiés
  - **`/cmd_vel`** — `geometry_msgs/Twist` (commande de vitesse envoyée au robot).

## Topics abonnés
  - **`/scan`** — `sensor_msgs/LaserScan` (pour l'évitement d'obstacles).
  - **`/coordinator/point`** — `geometry_msgs/PointStamped` (position cible).
  - **`/coordinator/state`** — `std_msgs/UInt8` (état global / mode).

## Comment lancer
  - Depuis un workspace ROS2 sourcé (node installé):
```bash
ros2 run turtlebot3_wanderer explorator
```
  - Le noeud est aussi lancé par certains `launch` de simulation (p.ex. dans `turtlebot3_gazebo` pour les scénarios de test).

## Fichiers utiles
  - Comportement principal: `src/turtlebot3_wanderer/turtlebot3_wanderer/wanderer.py`
  - Entrypoint / installation: `src/turtlebot3_wanderer/setup.py`
