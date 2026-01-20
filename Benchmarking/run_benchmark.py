#!/usr/bin/env python3
"""
Script principal de benchmarking
Rejoue les données RealSense enregistrées et teste tous les modèles
"""

import os
import sys
import yaml
import json
import pickle
import gzip
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from ball_detector import BallDetectorBenchmark, get_temperature


class BenchmarkRunner:
    # Gestionnaire de benchmarking avec données RealSense
    
    def __init__(self, config_path: str = "benchmark_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        
        # Créer le dossier de sortie
        self.output_dir = Path(self.config['output']['directory'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Timestamp pour cette session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_config(self) -> Dict:
        # Charge la configuration
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_dataset(self, dataset_path: str) -> Dict:
        # Charge le dataset RealSense
        print(f"\nChargement du dataset: {dataset_path}")
        
        with gzip.open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Dataset charge:")
        print(f"  - Frames: {dataset['metadata']['num_frames']}")
        print(f"  - Résolution: {dataset['metadata']['resolution']}")
        print(f"  - Date: {dataset['metadata']['recording_date']}")
        
        return dataset
    
    def _get_platform_info(self) -> str:
        # Détecte la plateforme actuelle
        import platform
        
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi 4' in model:
                    try:
                        from pycoral.utils import edgetpu
                        devices = edgetpu.list_edge_tpus()
                        if devices:
                            return "raspberry_pi4_coral"
                    except:
                        pass
                    return "raspberry_pi4"
        
        return "pc"
    
    def _discover_models(self) -> List[Dict]:
        # Découvre tous les modèles à tester
        models = []
        base_path = Path(os.path.dirname(self.config_path))
        
        for model_dir_config in self.config['model_directories']:
            dir_path = base_path / model_dir_config['path']
            
            if not dir_path.exists():
                continue
            
            #  TFLite models
            for tflite_file in dir_path.glob("*.tflite"):
                is_edgetpu = "edgetpu" in tflite_file.name.lower()
                
                models.append({
                    'path': str(tflite_file),
                    'name': tflite_file.name,
                    'directory': model_dir_config['path'],
                    'input_size': model_dir_config['input_size'],
                    'type': model_dir_config['type'],
                    'is_edgetpu': is_edgetpu,
                    'runtime': 'tflite'
                })
            
            # ONNX models
            for onnx_file in dir_path.glob("*.onnx"):
                models.append({
                    'path': str(onnx_file),
                    'name': onnx_file.name,
                    'directory': model_dir_config['path'],
                    'input_size': model_dir_config['input_size'],
                    'type': model_dir_config['type'],
                    'is_edgetpu': False,
                    'runtime': 'onnx'
                })
        
        return models
    
    def _filter_models_for_platform(self, models: List[Dict], platform: str) -> List[Dict]:
        # Filtre les modèles compatibles avec la plateforme
        platform_config = self.config['platforms'].get(platform, {})
        use_edgetpu = platform_config.get('use_edgetpu', False)
        
        filtered = []
        for model in models:
            # Si EdgeTPU, ne garder que les modèles EdgeTPU
            if use_edgetpu:
                if model['is_edgetpu']:
                    filtered.append(model)
            else:
                # Sinon, exclure les modèles EdgeTPU
                if not model['is_edgetpu']:
                    filtered.append(model)
        
        return filtered
    
    def benchmark_model(self, model_info: Dict, dataset: Dict, platform: str) -> Dict:
        # Benchmark un modèle avec le dataset
        print(f"\n{'='*70}")
        print(f"Modèle: {model_info['name']}")
        print(f"Runtime: {model_info['runtime'].upper()}")
        print(f"Plateforme: {platform}")
        print(f"{'='*70}")
        
        # Créer le détecteur
        try:
            detector = BallDetectorBenchmark(
                model_path=model_info['path'],
                input_size=model_info['input_size'],
                runtime=model_info['runtime'],
                conf_threshold=self.config['benchmark']['confidence_threshold'],
                use_edgetpu=model_info['is_edgetpu']
            )
        except Exception as e:
            print(f"Erreur chargement: {e}")
            return None
        
        # Warmup
        warmup_iters = self.config['benchmark']['warmup_iterations']
        detector.warmup(iterations=warmup_iters)
        
        # Test sur le dataset
        frames = dataset['frames']
        camera_info = dataset['camera_info']
        
        print(f"\nTest sur {len(frames)} frames...")
        
        temperatures = []
        
        for i, frame_data in enumerate(tqdm(frames, desc="Processing")):
            rgb = frame_data['rgb']
            depth = frame_data['depth']
            
            # Détecter
            detections, timings = detector.detect(rgb, depth, camera_info, record_metrics=True)
            
            # Mesurer la température périodiquement
            if i % 10 == 0:
                temp = get_temperature()
                if temp is not None:
                    temperatures.append(temp)
        
        # Récupérer les métriques
        summary = detector.get_summary_metrics()
        
        # Résultat complet
        result = {
            'timestamp': self.timestamp,
            'model_name': model_info['name'],
            'model_path': model_info['path'],
            'model_directory': model_info['directory'],
            'model_type': model_info['type'],
            'input_size': model_info['input_size'],
            'is_edgetpu': model_info['is_edgetpu'],
            'runtime': model_info['runtime'],
            'platform': platform,
            'dataset_frames': len(frames),
            'confidence_threshold': self.config['benchmark']['confidence_threshold'],
            **summary
        }
        
        # Températures
        if temperatures:
            result['temperature_mean'] = float(np.mean(temperatures))
            result['temperature_max'] = float(np.max(temperatures))
            result['temperature_min'] = float(np.min(temperatures))
        
        # Afficher résumé
        self._print_summary(result, summary)
        
        return result
    
    def _print_summary(self, result: Dict, summary: Dict):
        # Affiche un résumé des résultats
        print(f"\n{'─'*70}")
        print(f"RÉSULTATS:")
        print(f"{'─'*70}")
        print(f"  Inférence:      {summary['inference_time_mean']:.2f} ± {summary['inference_time_std']:.2f} ms")
        print(f"  Total (3D):     {summary['total_time_mean']:.2f} ms")
        print(f"  FPS:            {summary['fps_mean']:.2f} (min: {summary['fps_min']:.2f}, max: {summary['fps_max']:.2f})")
        print(f"  Mémoire:        {summary['memory_usage_mean']:.1f} MB (max: {summary['memory_usage_max']:.1f} MB)")
        print(f"  CPU:            {summary['cpu_usage_mean']:.1f}% (max: {summary['cpu_usage_max']:.1f}%)")
        if 'temperature_mean' in result:
            print(f"  Température:    {result['temperature_mean']:.1f}°C (max: {result['temperature_max']:.1f}°C)")
        print(f"  Détections 2D:  {summary['detection_count_total']}")
        print(f"  Détections 3D:  {summary['detection_3d_count_total']}")
        if summary['confidence_mean'] > 0:
            print(f"  Confiance (moy):{summary['confidence_mean']:.3f} ± {summary['confidence_std']:.3f}")
            print(f"  Confiance (max):{summary['max_confidence_mean']:.3f} (min: {summary['max_confidence_min']:.3f}, max: {summary['max_confidence_max']:.3f})")
        print(f"{'─'*70}")
    
    def run(self, dataset_path: str, platform: str = None):
        # Lance le benchmarking
        dataset = self.load_dataset(dataset_path)
        
        # Détecter la plateforme
        if platform is None:
            platform = self._get_platform_info()
            print(f"\nPlateforme detectee: {platform}")
        
        # Découvrir les modèles
        print("\nDecouverte des modeles...")
        all_models = self._discover_models()
        print(f"{len(all_models)} modeles trouves")
        
        # Filtrer pour la plateforme
        models = self._filter_models_for_platform(all_models, platform)
        print(f"{len(models)} modeles compatibles avec {platform}")
        
        if not models:
            print("Aucun modele a tester!")
            return
        
        # Benchmarker chaque modèle
        for model_info in models:
            result = self.benchmark_model(model_info, dataset, platform)
            if result:
                self.results.append(result)
        
        self.save_results()
        
        print(f"\n{'='*70}")
        print(f"Benchmarking termine!")
        print(f"Resultats dans: {self.output_dir}")
        print(f"{'='*70}")
    
    def save_results(self):
        # Sauvegarde les résultats
        if not self.results:
            return
        
        # JSON
        json_path = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResultats JSON: {json_path}")
        
        # CSV
        import csv
        csv_path = self.output_dir / f"benchmark_results_{self.timestamp}.csv"
        fieldnames = list(self.results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"Resultats CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark des modèles YOLO avec dataset RealSense")
    parser.add_argument('dataset', type=str, help='Chemin vers le dataset RealSense (.pkl.gz)')
    parser.add_argument('--config', type=str, default='benchmark_config.yaml',
                       help='Fichier de configuration')
    parser.add_argument('--platform', type=str, choices=['pc', 'raspberry_pi4', 'raspberry_pi4_coral', 'jetson_orin'],
                       help='Plateforme (auto-détection si non spécifié)')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(config_path=args.config)
    runner.run(dataset_path=args.dataset, platform=args.platform)


if __name__ == '__main__':
    main()
