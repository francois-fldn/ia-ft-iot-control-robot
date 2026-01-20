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

- 24 modèles pour Raspberry Pi 4, PC et Jetson
- 4 modèles pour Coral TPU

```bash
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform pc
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4_coral
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform jetson_orin
```

### 4. Analyser les resultats

#### Analyse d'un benchmark unique
```bash
# Analyse d'un seul fichier JSON
python3 analyze_results.py benchmark_results/benchmark_results_TIMESTAMP.json
```

#### Analyse avec agregation (plusieurs repetitions)
Pour améliorer la fiabilité des résultats, exécutez le benchmark **10 fois** et analysez la moyenne :

```bash
# 1. Répéter le benchmark 10 fois
for i in {1..10}; do
    python3 run_benchmark.py tennis_ball_dataset.pkl.gz
done

# 2. Analyser le dossier entier (calcule automatiquement moyenne + écart-type)
python3 analyze_results.py benchmark_results/benchmark_results_PLATFORM/

```

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

**Graphiques générés :**
- `power_consumption.png` : Consommation en Watts par modèle
- `efficiency_fps_per_watt.png` : Efficacité (FPS/W)

#### Comparer plusieurs plateformes
```bash
# Analyser et comparer 2 plateformes
python3 compare_platforms.py benchmark_results/benchmark_results_pi4/ benchmark_results/benchmark_results_coral/


**Graphiques de comparaison générés :**
- Temps d'inférence par plateforme
- FPS par plateforme
- Speedup relatif (heatmap)
- Consommation électrique (si disponible)
- Efficacité énergétique (FPS/Watt)


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
├── compare_platforms.py      # Comparer plateformes
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

- **compare_platforms.py** : Comparaison multi-plateforme
  - Heatmap de speedup
  - Tableaux comparatifs

---

