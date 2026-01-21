# src — Packages et instructions d'installation

Ce fichier donne une vue d'ensemble des packages présents dans `src/`, leurs objectifs et les étapes d'installation/prise en main pour exécuter les noeuds Python ROS2 fournis.

## Paquets présents
- `ball_detection` — Détecteur de balle (TFLite). Publié: `ball_3d` (`geometry_msgs/PointStamped`), `ball_marker` (`visualization_msgs/Marker`), `ball_debug` (`sensor_msgs/Image`). L'entrypoint est `ball_detection.b_d:main`.
- `coordinator` — Orchestrateur d'état (STOP / ROTATE / WANDERER / LOCK_IN / GO). Publie l'état et le point cible (`/coordinator/state`, `/coordinator/point`).
- `realsense_publisher` — Noeud pour Intel RealSense : publie `Realsense/Image/Color`, `Realsense/Image/Depth` et `Realsense/CameraInfo`.
- `turtlebot3_wanderer` — Comportement d'exploration / suivi de balle ; publie `/cmd_vel` et s'abonne à `/coordinator/point`, `/coordinator/state` et `/scan`.
- `turtlebot3_descriptions`, `turtlebot3_gazebo` — descriptions et launch/simulation (Gazebo) pour TurtleBot3 (scénarios, modèles, launches).

---

## Installation (prérequis)

Ces instructions supposent que vous avez déjà installé ROS2 (ex: Humble, Iron, ou autre) et que vous connaissez votre distribution ROS (`<ros2-distro>`). Adaptez les commandes si besoin.

### 1. Installer dépendances système (exemples ; adaptez selon distro):

```bash
sudo apt update
# dépendances courantes pour OpenCV / CV bridge
sudo apt install -y python3-opencv build-essential libusb-1.0-0-dev pkg-config
# si utilisation de RealSense
sudo apt install -y librealsense2-dkms librealsense2-utils librealsense2-dev
```

Remarque: `cv_bridge` est fourni via les paquets ROS (`ros-<ros2-distro>-cv-bridge`) ; installez-le si nécessaire.

### 2. Installer les dépendances Python listées dans `src/requirements.txt`.

Recommandation: utilisez un environnement virtuel (`venv`) ou installez en utilisateur pour éviter de polluer le système:

#### depuis la racine du dépôt
```bash
cd src/
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

Notes sur certains paquets listés:
- `pyrealsense2`: sur certaines plateformes il est préférable d'installer la version du système (`librealsense2` + `pip install pyrealsense2`) ou via le dépôt Intel.
- `tflite-runtime`: sur Raspberry Pi / Coral il existe des builds spécifiques ; sinon installer `tensorflow` si compatible.