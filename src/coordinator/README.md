# Coordinator

## Objectif 
Orchestrer le comportement global du robot (états: STOP, ROTATE, WANDERER, LOCK_IN, GO). Le `coordinator` décide de l'état courant et diffuse la position cible (ex: position de la balle) aux autres noeuds.

## Topics publiés
  - **`/coordinator/state`**: `std_msgs/UInt8` — état courant du système.
  - **`/coordinator/point`**: `geometry_msgs/PointStamped` — position cible (ex: balle).

## Topics abonnés
  - Noeuds de détection (position de la balle)

## Comment lancer
  - Depuis un workspace ROS2 sourcé:
```bash
ros2 run coordinator coordinator
```

## Fichiers utiles
  - `src/coordinator/coordinator/coordo.py`
  - `src/coordinator/setup.py`
