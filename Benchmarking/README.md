# 🎾 Benchmarking YOLO - Détection de Balle

Système de benchmarking pour modèles YOLO (TFLite/ONNX) sur PC, Raspberry Pi 4, et Raspberry Pi 4 + Coral TPU.

## 🚀 Démarrage Rapide

### 1️⃣ Enregistrer les données RealSense

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

### 2️⃣ Installer les dépendances

```bash
# PC ou Raspberry Pi 4
pip install opencv-python tqdm psutil pandas matplotlib seaborn pyyaml
pip install onnxruntime  # Pour modèles ONNX

# Coral TPU (Raspberry Pi 4 uniquement)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install libedgetpu1-std python3-pycoral
```

### 3️⃣ Lancer le benchmark

```bash
# Auto-détection de la plateforme
python3 run_benchmark.py tennis_ball_dataset.pkl.gz

# Ou spécifier la plateforme
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform pc
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4_coral
```

**Durée estimée:** PC: 15-30 min | RPi4: 30-60 min | RPi4+Coral: 15-20 min

### 4️⃣ Analyser les résultats

```bash
# Analyse individuelle
python3 analyze_results.py benchmark_results/benchmark_results_TIMESTAMP.json

# Ouvrir le rapport HTML
open benchmark_results/benchmark_report_TIMESTAMP.html

# Comparer plusieurs plateformes
python3 compare_platforms.py results_pc.json results_rpi4.json results_coral.json
```

---

## 📊 Métriques Mesurées

| Métrique | Description | Unité |
|----------|-------------|-------|
| **Inference Time** | Temps YOLO pur (mean, std, min, max, p50, p95, p99) | ms |
| **Total Time** | Pipeline complet (preprocessing + inference + 3D) | ms |
| **FPS** | Images par seconde (mean, min, max) | fps |
| **CPU Usage** | Utilisation processeur (mean, max) | % |
| **Memory Usage** | Consommation RAM (mean, max) | MB |
| **Temperature** | Température système (si disponible) | °C |
| **Detections 2D** | Balles détectées dans l'image | count |
| **Detections 3D** | Balles avec coordonnées 3D valides | count |
| **Confidence** | Score de confiance (mean, std) | 0-1 |

**std** (écart-type) = stabilité des performances. Plus c'est faible, plus c'est stable.

---

## 📁 Structure

```
DETEC_BALL_PROJET/
├── record_realsense.py      # 📹 Enregistrer données RealSense
├── preview_dataset.py        # 👀 Prévisualiser le dataset
├── ball_detector.py          # 🤖 Détecteur TFLite/ONNX
├── run_benchmark.py          # 🎯 Lancer le benchmark
├── analyze_results.py        # 📊 Analyser les résultats
├── compare_platforms.py      # 🔄 Comparer plateformes
├── benchmark_config.yaml     # ⚙️ Configuration
└── modeles_yolo/            # 🤖 Modèles (256, 320, pruned)
    ├── 256/
    ├── 320/
    ├── 256_pruned/
    └── 320_pruned/
```

**Résultats générés** (dans `benchmark_results/`):
- `*.json` : Données brutes
- `*.csv` : Tableau Excel
- `*.html` : Rapport interactif avec graphiques

---

## 🎯 Workflow Complet

```bash
# 1. Enregistrer le dataset (sur machine avec RealSense)
python3 record_realsense.py --output tennis_dataset.pkl.gz --frames 300

# 2. Benchmarker sur PC
python3 run_benchmark.py tennis_dataset.pkl.gz --platform pc

# 3. Transférer le dataset sur Raspberry Pi
scp tennis_dataset.pkl.gz pi@raspberrypi:~/

# 4. Benchmarker sur RPi4
ssh pi@raspberrypi
python3 run_benchmark.py tennis_dataset.pkl.gz --platform raspberry_pi4

# 5. Benchmarker sur RPi4 + Coral (si disponible)
python3 run_benchmark.py tennis_dataset.pkl.gz --platform raspberry_pi4_coral

# 6. Récupérer tous les JSON et comparer
python3 compare_platforms.py results_*.json
```

---

## 🔧 Configuration

Éditez `benchmark_config.yaml` pour modifier:
- Seuil de confiance (défaut: 0.30)
- Dossiers de modèles
- Nombre d'itérations warmup

---

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'onnxruntime'` | `pip install onnxruntime` |
| Coral TPU non détecté | `lsusb \| grep "Global Unichip"` + réinstaller drivers |
| Pas de température | Normal sur PC. Sur RPi: vérifier `/sys/class/thermal/thermal_zone0/temp` |
| Dataset trop gros | Réduire `--frames` (ex: 100) |
| Détections 3D = 0 | Vérifier données profondeur RealSense |

---

## 📈 Interprétation des Résultats

### FPS (Images/sec)
- **> 30 FPS** : Temps réel ✅
- **10-30 FPS** : Acceptable pour certaines applications
- **< 10 FPS** : Trop lent ❌

### Mémoire
- **< 100 MB** : Excellent pour embarqué ✅
- **100-300 MB** : Acceptable
- **> 300 MB** : Risque sur Raspberry Pi

### Température (Raspberry Pi)
- **< 60°C** : Normal ✅
- **60-80°C** : Chaud mais acceptable
- **> 80°C** : Risque de throttling ⚠️

---

## 🎓 Modèles Testés

Le système teste automatiquement:
- **TFLite** : FP32, FP16, INT8 (standard et pruned)
- **TFLite EdgeTPU** : Optimisé pour Coral TPU
- **ONNX** : FP32, FP16, INT8 (avec XNNPACK sur ARM)

**Résolutions** : 256×256 et 320×320 pixels

---

## 💡 Recommandations

Pour votre projet ROS2 de détection de balle:

| Critère | Priorité | Seuil |
|---------|----------|-------|
| FPS | ⭐⭐⭐⭐⭐ | > 30 fps |
| Latence | ⭐⭐⭐⭐ | < 33 ms |
| Mémoire | ⭐⭐⭐ | < 200 MB |
| Précision | ⭐⭐⭐⭐ | Conf > 0.7 |

**Conseil** : Privilégiez les modèles **INT8** avec **pruning** (ex: `best-int8_256_pruned`) pour un bon compromis vitesse/précision sur Raspberry Pi.

---

## 📚 Fichiers Utiles

- **record_realsense.py** : Capture RGB + Depth + CameraInfo
  - Enregistre au format `.pkl.gz` (compressé)
  - Reproductible sur toutes les plateformes

- **ball_detector.py** : Pipeline complet
  - Preprocessing (resize, normalize)
  - Inférence (TFLite/ONNX)
  - Postprocessing (NMS)
  - Lookup profondeur + projection 3D

- **run_benchmark.py** : Orchestrateur
  - Découverte automatique des modèles
  - Détection de plateforme
  - Sauvegarde JSON + CSV

- **analyze_results.py** : Visualisation
  - Graphiques comparatifs (PNG)
  - Rapport HTML interactif

- **compare_platforms.py** : Comparaison multi-plateforme
  - Heatmap de speedup
  - Tableaux comparatifs

---

**🚀 Bon benchmarking !**

Pour plus de détails sur l'architecture, consultez les anciennes documentations ou le code source.
