# Realsense Publisher

## Objectif 
Publier les flux (couleur et profondeur) et les `CameraInfo` d'une caméra Intel RealSense (alignement Depth→Color, hole-filling, 6 FPS). Le noeud principal est `PointCloudPublisher` défini dans `realsense_p.py`.

## Topics publiés
  - **`Realsense/Image/Color`** — `sensor_msgs/Image` (format `bgr8`).
  - **`Realsense/Image/Depth`** — `sensor_msgs/Image` (profondeur, format `z16` / passthrough).
  - **`Realsense/CameraInfo`** — `sensor_msgs/CameraInfo` (intrinsèques de la caméra).

## Dépendances
  - `pyrealsense2`, `cv_bridge`, `numpy`, `rclpy`, `sensor_msgs`.

## Comment lancer
  - Depuis un workspace ROS2 sourcé (node installé):
```bash
ros2 run realsense_publisher realsense_p
```
  - Il existe aussi un `launch` helper pour démarrer avec Foxglove / WebSocket (si présent dans le package):
```bash
ros2 launch realsense_publisher realsense_publisher_with_foxglove_launch.py
```

## Fichiers utiles
  - Node principal: `src/realsense_publisher/realsense_publisher/realsense_p.py`
  - Entrypoint / installation: `src/realsense_publisher/setup.py`
