# Protocole Complet de Benchmarking YOLO

**Projet** : Détection de balle avec YOLO + RealSense  
**Objectif** : Comparer les performances de modèles YOLO (FP32, FP16, INT8, pruned) sur PC, Raspberry Pi 4, et Raspberry Pi 4 + Coral TPU  
**Date** : 2026-01-14

---

## Prerequis

### Matériel
- RealSense D435i (pour enregistrement)
- PC (pour benchmark)
- Raspberry Pi 4 (optionnel)
- Google Coral TPU USB (optionnel)

### Logiciels Installés
```bash
# PC et Raspberry Pi
pip install opencv-python tqdm psutil pandas matplotlib seaborn pyyaml onnxruntime

# Raspberry Pi avec Coral TPU uniquement
sudo apt-get install libedgetpu1-std python3-pycoral
```

---

## ETAPE 1 : Enregistrer le Dataset RealSense

### 1.1 Démarrer le Publisher RealSense

```bash
# Terminal 1
cd ~/GitProjet/ia-ft-iot-control-robot
ros2 run realsense_publisher realsense_p
```

### 1.2 Enregistrer 300 Frames

```bash
# Terminal 2
cd ~/DETEC_BALL_PROJET
python3 record_realsense.py --output tennis_ball_dataset.pkl.gz --frames 300
```

**Durée** : ~50 secondes à 6 FPS  
**Taille fichier** : ~50-100 MB (compressé)

### 1.3 Vérifier le Dataset

```bash
# Sans détection (juste visualiser)
python3 preview_dataset.py tennis_ball_dataset.pkl.gz

# Avec détection en temps réel
python3 preview_dataset.py tennis_ball_dataset.pkl.gz \
    --detect modeles_yolo/256/best_int8_256.onnx \
    --conf 0.3 \
    --fps 10
```

**Controles** :
- `ESPACE` : Pause/Reprendre
- `→` / `←` : Frame suivante/précédente (en pause)
- `S` : Sauvegarder la frame
- `Q` / `ESC` : Quitter

**Vérifications** :
- Bounding box bien alignee sur la balle
- Distance 3D (Z en metres) affichee
- Confiance > 0.3 pour la balle
- Pas de superposition de boxes

---

## ETAPE 2 : Benchmark sur PC

### 2.1 Activer l'Environnement Virtuel

```bash
source ~/env/bin/activate
# OU directement :
/path/to/env/bin/python3 run_benchmark.py ...
```

### 2.2 Lancer le Benchmark

```bash
cd ~/DETEC_BALL_PROJET
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform pc
```

**Durée estimée** : 15-30 minutes (12 modèles ONNX testés)

### 2.3 Résultats Générés

```
benchmark_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json  # Données brutes
└── benchmark_results_YYYYMMDD_HHMMSS.csv   # Tableau Excel
```

### 2.4 Métriques Mesurées

| Catégorie | Métrique | Description |
|-----------|----------|-------------|
| **Performance** | `inference_time` | Temps YOLO pur (mean, std, min, max, p50, p95, p99) |
| | `total_time` | Pipeline complet (preprocessing + inference + 3D) |
| | `fps` | Images/sec (mean, min, max) |
| **Ressources** | `memory_usage` | RAM en MB (mean, max) |
| | `cpu_usage` | CPU en % (mean, max) |
| | `temperature` | °C système (mean, max, min) |
| **Qualité** | `detection_count` | Nombre de détections 2D |
| | `detection_3d_count` | Nombre de détections 3D valides |
| | `confidence` | Confiance moyenne de toutes détections |
| | `max_confidence` | **NOUVEAU** Confiance de la meilleure détection par frame |

---

## ETAPE 3 : Benchmark sur Raspberry Pi 4

### 3.1 Transférer le Dataset

```bash
scp tennis_ball_dataset.pkl.gz pi@raspberrypi:~/benchmark/
```

### 3.2 Sur le Raspberry Pi

```bash
ssh pi@raspberrypi
cd ~/benchmark
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4
```

**Durée estimée** : 30-60 minutes

---

## ETAPE 4 : Benchmark sur Raspberry Pi 4 + Coral TPU

### 4.1 Vérifier que le Coral est Détecté

```bash
lsusb | grep "Global Unichip"
# Doit afficher : Bus XXX Device XXX: ID 1a6e:089a Global Unichip Corp.
```

### 4.2 Lancer le Benchmark

```bash
python3 run_benchmark.py tennis_ball_dataset.pkl.gz --platform raspberry_pi4_coral
```

**Durée estimée** : 15-20 minutes (moins de modèles EdgeTPU)

---

## ETAPE 5 : Analyser les Resultats

### 5.1 Analyse Individuelle (une plateforme)

```bash
python3 analyze_results.py benchmark_results/benchmark_results_TIMESTAMP.json
```

**Génère** :
- `plots/inference_time_comparison.png`
- `plots/fps_comparison.png`
- `plots/memory_usage.png`
- `plots/cpu_usage.png`
- `plots/efficiency_scatter.png`
- `benchmark_report_TIMESTAMP.html` (rapport interactif)

### 5.2 Ouvrir le Rapport

```bash
open benchmark_results/benchmark_report_TIMESTAMP.html
```

### 5.3 Comparaison Multi-Plateformes

```bash
python3 compare_platforms.py \
    benchmark_results/benchmark_results_PC.json \
    benchmark_results/benchmark_results_RPI4.json \
    benchmark_results/benchmark_results_CORAL.json
```

**Génère** :
- Heatmap de speedup
- Graphiques comparatifs
- Tableau CSV de comparaison
- Rapport HTML multi-plateformes

---

## Interpretation des Resultats

### Performance (FPS)
- **> 30 FPS** : Temps réel ✅
- **10-30 FPS** : Acceptable pour certaines applications
- **< 10 FPS** : Trop lent ❌

### Confiance
- **`confidence_mean`** : Confiance moyenne de toutes les détections (incluant duplicatas)
- **`max_confidence_mean`** : Confiance moyenne de la **meilleure détection** par frame
  - Si `max_confidence_mean` > 0.7 → Modèle fiable
  - Si `max_confidence_mean` < 0.5 → Modèle hésite

### Stabilité (std)
- **Faible** (< 0.5 ms) : Très stable ✅
- **Moyen** (0.5-1.5 ms) : Acceptable
- **Élevé** (> 1.5 ms) : Performance imprévisible ⚠️

### Mémoire
- **< 100 MB** : Excellent pour embarqué ✅
- **100-300 MB** : Acceptable
- **> 300 MB** : Risque sur Raspberry Pi

### Température (Raspberry Pi)
- **< 60°C** : Normal ✅
- **60-80°C** : Chaud mais acceptable
- **> 80°C** : Risque de throttling ⚠️

---

## Depannage

| Problème | Solution |
|----------|----------|
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'onnxruntime'` | `pip install onnxruntime` |
| Dataset trop gros | Réduire `--frames` (ex: 100) |
| Coral TPU non détecté | `lsusb \| grep Unichip` + réinstaller drivers |
| Bbox mal alignée | ✅ Corrigé (utilise `model_w` au lieu de 320 hardcodé) |
| Détections 3D = 0 | ✅ Corrigé (conversion mm→m avant test de validité) |
| Boxes qui se superposent | Normal dans le benchmark (mesure toutes détections). Preview affiche seulement la meilleure. |

---

## Corrections Appliquees

### 1. Format d'Entrée ONNX (NHWC → NCHW)
**Problème** : Dimension mismatch  
**Solution** : Transposition dans `ball_detector.py` ligne 315
```python
input_data = np.transpose(input_data, (0, 3, 1, 2))  # NHWC -> NCHW
```

### 2. Type de Données FP16
**Problème** : Modèles FP16 attendaient float16  
**Solution** : Détection automatique du type dans `ball_detector.py` ligne 111-114
```python
input_type = self.session.get_inputs()[0].type
self.input_dtype = np.float16 if 'float16' in input_type else np.float32
```

### 3. Détection 3D
**Problème** : Toujours 0 détections 3D  
**Solution** : Conversion mm→m **avant** le test de validité (ligne 221)
```python
dist_z_m = dist_z / 1000.0 if dist_z > 100 else dist_z
if 0.1 < dist_z_m < 10.0:  # Maintenant utilise dist_z_m
```

### 4. Alignement Bounding Box
**Problème** : Bbox décalée sur modèles 256×256  
**Solution** : Utilise `self.model_w` au lieu de 320 hardcodé (ligne 188)
```python
scale_x = cam_w / float(self.model_w)  # Au lieu de 320.0
```

### 5. Métrique Max Confidence
**Ajout** : Nouvelle métrique pour mesurer la confiance de la meilleure détection par frame
- `max_confidence_mean` : Moyenne
- `max_confidence_std` : Stabilité
- `max_confidence_min` / `max_confidence_max` : Plage

---

## Recommandations Finales

### Critères de Sélection

| Critère | Priorité | Seuil Minimum |
|---------|----------|---------------|
| FPS | ⭐⭐⭐⭐⭐ | > 30 fps |
| Latence | ⭐⭐⭐⭐ | < 33 ms |
| Max Confidence | ⭐⭐⭐⭐ | > 0.7 |
| Mémoire | ⭐⭐⭐ | < 200 MB |
| Température | ⭐⭐ | < 70°C |

### Modèles Recommandés

**Pour Raspberry Pi** :
1. `best-int8_256_pruned.onnx` : Meilleur compromis vitesse/taille
2. `best-int8_320_pruned.onnx` : Si on veut plus de précision

**Pour Raspberry Pi + Coral** :
1. `best-int8_edgetpu256.tflite` : Accélération TPU maximale
2. `best-int8_edgetpu320.tflite` : Si précision prioritaire

**Pour PC (développement)** :
- `best-fp32_256.onnx` ou `best-fp32_320.onnx` : Meilleure précision

---

## Fichiers du Systeme

```
DETEC_BALL_PROJET/
├── record_realsense.py           # Enregistrer donnees RealSense
├── preview_dataset.py             # Previsualiser + detection live
├── ball_detector.py               # Detecteur TFLite/ONNX
├── run_benchmark.py               # Orchestrateur de benchmark
├── analyze_results.py             # Analyse individuelle
├── compare_platforms.py           # Comparaison multi-plateformes
├── benchmark_config.yaml          # Configuration
├── README.md                      # Documentation
├── BENCHMARK_PROTOCOL.md          # Ce fichier
└── modeles_yolo/                  # Modeles a tester
    ├── 256/
    ├── 320/
    ├── 256_pruned/
    └── 320_pruned/
```

---

**Version** : 1.1 (avec corrections 3D, bbox, max_confidence)  
**Auteur** : Benchmarking YOLO RealSense  
**Dernière mise à jour** : 2026-01-14
