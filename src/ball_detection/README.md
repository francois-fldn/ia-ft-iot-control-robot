# Ball Detection

## Objectif
Détecter une balle dans le flux caméra (inférence TFLite) et publier sa position 3D estimée. Le détecteur utilise un modèle YOLO quantifié (`.tflite`), la profondeur et les `CameraInfo` pour reprojecter la position de la balle en coordonnées du robot.

## Topics publiés
  - **`ball_3d`** — `geometry_msgs/PointStamped` : position 3D estimée de la balle (frame_id=`base_footprint`).
  - **`ball_marker`** — `visualization_msgs/Marker` : marqueur RViz pour visualiser la balle.
  - **`ball_debug`** — `sensor_msgs/Image` : image debug avec boîte englobante et information de distance.

## Topics abonnés
  - **`rgb_camera/image`** — `sensor_msgs/Image` (image couleur, format `bgr8`).
  - **`depth_camera/image`** — `sensor_msgs/Image` (image de profondeur, encoding `passthrough`).
  - **`rgb_camera/camera_info`** — `sensor_msgs/CameraInfo` (intrinsèques utilisées pour reprojection).

## Dépendances
  - `rclpy`, `cv_bridge`, `numpy`, `opencv-python` (ou `opencv` du système), `tflite_runtime` (ou `tensorflow-lite`/`tensorflow` selon plateforme).
  - Le modèle TFLite est déployé dans `share/ball_detection/models/` via `setup.py`.

## Comment lancer
  - Depuis un workspace ROS2 sourcé et après avoir installé/built le package :
```bash
ros2 run ball_detection ball_detector
```
  - Le package contient un `console_script` pointant vers `ball_detection.b_d:main` (voir `src/ball_detection/setup.py`).

## Fichiers utiles
  - Node principal: `src/ball_detection/ball_detection/b_d.py`
  - Entrypoint / installation: `src/ball_detection/setup.py`
  - Modèles: `src/ball_detection/models/*.tflite`
