# Benchmarking YOLO - Detection de Balle

Système de benchmarking pour modèles YOLO (TFLite/ONNX) sur PC, Raspberry Pi 4, et Raspberry Pi 4 + Coral TPU.
Python <= 3.10 est nécessaire pour le Coral TPU.

## Demarrage Rapide

### 1. Enregistrer les donnees RealSense

```bash
# Terminal 1: Lancer le publisher RealSense
cd ../GitProjet/ia-ft-iot-control-robot
ros2 run realsense_publisher realsense_p

# Terminal 2: Enregistrer 300 frames (~50s)
cd /path/to/DETEC_BALL_PROJET
python3 record_realsense.py --output tennis_ball_dataset.pkl.gz --frames 300
```

**Options:**
- `--frames N` : Nombre de frames (défaut: 300)
- `--gazebo` : Utiliser Gazebo au lieu de RealSense

**Prévisualiser le dataset:**
```bash
python3 preview_dataset.py tennis_ball_dataset.pkl.gz

# Avec visualisation de la bounding box :
python3 preview_dataset.py tennis_ball_dataset.pkl.gz --detect modeles_yolo/256/best-int8_256.tflite
```

### 2. Installer les dependances

```bash
# PC ou Raspberry Pi 4
pip install -r requirements.txt

# Coral TPU (Raspberry Pi 4 uniquement)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install libedgetpu1-std python3-pycoral
```

### 3. Lancer le benchmark

- 24 modèles pour Raspberry Pi 4 et PC
- 4 modèles pour Coral TPU

```bash
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform pc
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4_coral
```

### 4. Analyser les resultats

#### Analyse avec agregation (plusieurs repetitions)
Pour améliorer la fiabilité des résultats, exécutez le benchmark **10 fois**.

**Avantages de l'agrégation :**
- **Moyenne** : Valeur representative de la performance
- **Ecart-type (std)** : Mesure de la stabilite (plus faible = plus fiable)
- Elimine les valeurs aberrantes dues aux pics de charge systeme

#### Analyse de la consommation electrique (Raspberry Pi uniquement)
Si un fichier `benchmark_conso_Amp.json` est présent dans le dossier des résultats, l'analyse inclut automatiquement :
- **Consommation en Watts** (Ampères × 5V)
- **Efficacité énergétique** (FPS / Watt)

Exemple de structure attendue pour `benchmark_conso_Amp.json` :
```json
[
  {
    "model_name": "best-int8_256.tflite",
    "conso_ampere_mean": 0.85,
    "conso_ampere_std": 0.02
  }
]
```

#### Comparer Pi4 et Coral (tout-en-un)

Analyse complète avec comparaison Pi4 vs Coral, avec ecart-type sur les 10 repetitions:

```bash
# Analyse complète avec comparaison Pi4 vs Coral
python3 analyze_results.py benchmark_results/
```

**Graphiques générés dans `benchmark_results/plots/` :**
- `pi4_*.png` : Métriques Pi4 (séparées standard/pruned)
- `comparison_*.png` : Comparaison Pi4 vs Coral
- `scatter_efficiency_pi4_all.png` : Scatter plot FPS vs Mémoire (tous modèles Pi4)
- `scatter_comparison.png` : Scatter plot FPS vs Mémoire (modèles de comparaison)


---

## Metriques Mesurees

| Métrique | Description | Unité |
|----------|-------------|-------|
| **Inference Time** | Temps YOLO pur (mean, std, min, max, p50, p95, p99) | ms |
| **Total Time** | Pipeline complet (preprocessing + inference + 3D) | ms |
| **FPS** | Images par seconde (mean, min, max) | fps |
| **CPU Usage** | Utilisation processeur (mean, max) | % |
| **Memory Usage** | Consommation RAM (mean, max) | MB |
| **Temperature** | Température système (si disponible) | °C |
| **Power** | Consommation électrique (Raspberry Pi uniquement) | W |
| **Efficiency** | Efficacité énergétique (FPS / Watt) | fps/W |
| **Detections 2D** | Balles détectées dans l'image | count |
| **Detections 3D** | Balles avec coordonnées 3D valides | count |
| **Confidence** | Score de confiance (mean, std, max) | 0-1 |

**std** (écart-type) = stabilité des performances. Plus c'est faible, plus c'est stable.

**Info** : Avec l'agregation de 10 repetitions, toutes les metriques incluent automatiquement leur ecart-type, montrant la reproductibilite.

---

## Structure

```
DETEC_BALL_PROJET/
├── record_realsense.py      # Enregistrer donnees RealSense
├── preview_dataset.py        # Previsualiser le dataset
├── ball_detector.py          # Detecteur TFLite/ONNX
├── run_benchmark.py          # Lancer le benchmark
├── analyze_results.py        # Analyser les resultats
├── benchmark_config.yaml     # Configuration
└── modeles_yolo/            # Modeles (256, 320, pruned)
    ├── 256/
    ├── 320/
    ├── 256_pruned/
    └── 320_pruned/
```

**Resultats generes** (dans `benchmark_results/`):
- `*.json` : Donnees brutes
- `*.csv` : Tableau Excel
- `*.png` : Graphiques

## Configuration 

Éditez `benchmark_config.yaml` pour modifier:
- Seuil de confiance (défaut: 0.60)
- Dossiers de modèles
- Nombre d'itérations warmup

## Modeles Testes

Le système teste automatiquement:
- **TFLite** : FP32, FP16, INT8 (standard et pruned)
- **TFLite EdgeTPU** : Int8, Optimisé pour Coral TPU
- **ONNX** : FP32, FP16, INT8 (avec XNNPACK sur ARM)

**Résolutions** : 256×256 et 320×320 pixels

---

## Fichiers Utiles

- **record_realsense.py** : Capture RGB + Depth + CameraInfo
  - Enregistre au format `.pkl.gz` (compressé)
  - Reproductible sur toutes les plateformes

- **ball_detector.py** : Pipeline complet
  - Preprocessing (resize, normalize)
  - Inférence (TFLite/ONNX)
  - Postprocessing (NMS)
  - Lookup profondeur + projection 3D

- **preview_dataset.py** : Previsualiser le dataset
  - Affiche les images + annotations et la meilleure bounding box

- **run_benchmark.py** : Orchestrateur
  - Découverte automatique des modèles
  - Détection de plateforme
  - Sauvegarde JSON + CSV

- **analyze_results.py** : Visualisation
  - Graphiques comparatifs (PNG)


---

