#!/usr/bin/env python3
"""
Script pour prévisualiser un dataset RealSense enregistré
Affiche les frames RGB et Depth avec OpenCV
"""

import pickle
import gzip
import cv2
import numpy as np
import argparse
from pathlib import Path
from ball_detector import BallDetectorBenchmark


def preview_dataset(dataset_path: str, start_frame: int = 0, fps: int = 10, 
                   detect_model: str = None, conf_threshold: float = 0.3):
    """
    Prévisualise un dataset RealSense
    
    Args:
        dataset_path: Chemin vers le fichier .pkl.gz
        start_frame: Frame de départ
        fps: Vitesse de lecture (frames par seconde)
        detect_model: Chemin vers un modèle YOLO pour afficher les détections (optionnel)
        conf_threshold: Seuil de confiance pour les détections
    """
    print(f"\n📂 Chargement du dataset: {dataset_path}")
    
    # Charger le dataset
    with gzip.open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Afficher les métadonnées
    print("\n" + "="*60)
    print("📊 MÉTADONNÉES DU DATASET")
    print("="*60)
    print(f"Frames: {dataset['metadata']['num_frames']}")
    print(f"Résolution: {dataset['metadata']['resolution']}")
    print(f"Date: {dataset['metadata']['recording_date']}")
    print(f"Intrinsics: fx={dataset['camera_info']['k'][0]:.1f}, fy={dataset['camera_info']['k'][4]:.1f}")
    print("="*60)
    
    frames = dataset['frames']
    total_frames = len(frames)
    camera_info = dataset['camera_info']
    
    # Charger le détecteur si demandé
    detector = None
    if detect_model:
        model_path = Path(detect_model)
        if not model_path.exists():
            print(f"⚠️  Modèle introuvable: {detect_model}")
            print("   Prévisualisation sans détection")
        else:
            print(f"\n🤖 Chargement du modèle: {model_path.name}")
            runtime = 'onnx' if model_path.suffix == '.onnx' else 'tflite'
            input_size = 256 if '256' in model_path.name else 320
            use_edgetpu = 'edgetpu' in model_path.name
            
            detector = BallDetectorBenchmark(
                model_path=str(model_path),
                input_size=input_size,
                runtime=runtime,
                conf_threshold=conf_threshold,
                use_edgetpu=use_edgetpu
            )
            print(f"   Seuil de confiance: {conf_threshold}")
    
    print(f"\n▶️  Lecture à partir de la frame {start_frame}/{total_frames}")
    print(f"⏱️  Vitesse: {fps} FPS")
    if detector:
        print(f"🎯 Détection activée")
    print("\nCommandes:")
    print("  ESPACE  - Pause/Reprendre")
    print("  →       - Frame suivante (en pause)")
    print("  ←       - Frame précédente (en pause)")
    print("  Q/ESC   - Quitter")
    print("  S       - Sauvegarder la frame actuelle")
    print()
    
    # Variables de contrôle
    current_frame = start_frame
    paused = False
    delay = int(1000 / fps)  # Délai en ms
    
    while current_frame < total_frames:
        frame_data = frames[current_frame]
        rgb = frame_data['rgb']
        depth = frame_data['depth']
        
        # Normaliser la profondeur pour l'affichage
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Ajouter le numéro de frame
        rgb_display = rgb.copy()
        depth_display = depth_colored.copy()
        
        # Détecter si le détecteur est actif
        if detector:
            detections, _ = detector.detect(rgb, depth, camera_info, record_metrics=False)
            
            # Prendre uniquement la détection avec la plus haute confiance (comme b_d.py qui fait break)
            if detections:
                best_det = max(detections, key=lambda d: d['confidence'])
                detections = [best_det]  # Afficher seulement la meilleure
            
            # Dessiner les bounding boxes
            for det in detections:
                x, y, w, h = det['bbox']
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                conf = det['confidence']
                
                # Couleur selon confiance (vert = haute, jaune = moyenne, rouge = basse)
                if conf > 0.7:
                    color = (0, 255, 0)  # Vert
                elif conf > 0.5:
                    color = (0, 255, 255)  # Jaune
                else:
                    color = (0, 165, 255)  # Orange
                
                # Bounding box
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), color, 2)
                
                # Label avec confiance
                label = f"Ball {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(rgb_display, (x1, y1 - label_size[1] - 4), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(rgb_display, label, (x1, y1 - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Point 3D si disponible
                if det['pos_3d']:
                    pos_3d = det['pos_3d']
                    depth_text = f"Z:{pos_3d['x']:.2f}m"
                    cv2.putText(rgb_display, depth_text, (x1, y2 + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Nombre de détections
            det_text = f"Detections: {len(detections)}"
            cv2.putText(rgb_display, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 2)
        
        text = f"Frame {current_frame}/{total_frames-1}"
        cv2.putText(rgb_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        cv2.putText(depth_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Afficher côte à côte
        combined = np.hstack([rgb_display, depth_display])
        cv2.imshow('Dataset Preview - RGB (gauche) | Depth (droite)', combined)
        
        # Gestion des touches
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q') or key == 27:  # Q ou ESC
            break
        elif key == ord(' '):  # ESPACE
            paused = not paused
            status = "⏸️  PAUSE" if paused else "▶️  LECTURE"
            print(f"\r{status} - Frame {current_frame}/{total_frames-1}", end='', flush=True)
        elif key == 83:  # Flèche droite
            if paused and current_frame < total_frames - 1:
                current_frame += 1
                print(f"\r⏸️  PAUSE - Frame {current_frame}/{total_frames-1}", end='', flush=True)
            continue
        elif key == 81:  # Flèche gauche
            if paused and current_frame > 0:
                current_frame -= 1
                print(f"\r⏸️  PAUSE - Frame {current_frame}/{total_frames-1}", end='', flush=True)
            continue
        elif key == ord('s'):  # Sauvegarder
            output_dir = Path('preview_frames')
            output_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(output_dir / f'frame_{current_frame:04d}_rgb.jpg'), rgb)
            cv2.imwrite(str(output_dir / f'frame_{current_frame:04d}_depth.png'), depth_colored)
            print(f"\n💾 Frame {current_frame} sauvegardée dans {output_dir}/")
            continue
        
        if not paused:
            current_frame += 1
    
    cv2.destroyAllWindows()
    print("\n\n✅ Prévisualisation terminée")


def main():
    parser = argparse.ArgumentParser(description="Prévisualise un dataset RealSense")
    parser.add_argument('dataset', type=str, help='Chemin vers le fichier .pkl.gz')
    parser.add_argument('--start', type=int, default=0, 
                       help='Frame de départ (défaut: 0)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Vitesse de lecture en FPS (défaut: 10)')
    parser.add_argument('--detect', type=str, default=None,
                       help='Chemin vers un modèle YOLO pour afficher les détections (ex: modeles_yolo/256/best_int8_256.onnx)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Seuil de confiance pour les détections (défaut: 0.3)')
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    if not Path(args.dataset).exists():
        print(f"❌ Erreur: Fichier introuvable: {args.dataset}")
        return
    
    preview_dataset(args.dataset, args.start, args.fps, args.detect, args.conf)


if __name__ == '__main__':
    main()
