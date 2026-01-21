# ia-ft-iot-control-robot

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

Installer les dépendances pour les fichiers python
```bash
cd src/
pip -r install requirements.txt
```

Lancer la commande `run.sh`



## Démonstration du projet

[demo.webm](https://github.com/user-attachments/assets/fa0be8ee-7a7e-4fdc-a6dd-702e5762a50f)

