#!/usr/bin/env python3
"""
Détecteur de balle pour benchmarking
Supporte TFLite et ONNX (avec XNNPACK)
"""

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional
import psutil
import os


class BallDetectorBenchmark:
    """Détecteur de balle optimisé pour le benchmarking"""
    
    def __init__(self, model_path: str, input_size: int, runtime: str = 'tflite',
                 conf_threshold: float = 0.25, use_edgetpu: bool = False):
        """
        Args:
            model_path: Chemin vers le modèle
            input_size: Taille d'entrée (256 ou 320)
            runtime: 'tflite' ou 'onnx'
            conf_threshold: Seuil de confiance
            use_edgetpu: Utiliser EdgeTPU (TFLite uniquement)
        """
        self.model_path = model_path
        self.model_w = input_size
        self.model_h = input_size
        self.conf_threshold = conf_threshold
        self.runtime = runtime
        self.use_edgetpu = use_edgetpu
        
        # Métriques
        self.metrics = {
            'inference_times': [],
            'preprocessing_times': [],
            'postprocessing_times': [],
            'depth_lookup_times': [],
            'projection_3d_times': [],
            'total_times': [],
            'confidence_scores': [],
            'max_confidence_scores': [],  # Meilleure confiance par frame
            'detection_counts': [],
            'detection_3d_counts': [],
            'memory_usage': [],
            'cpu_usage': [],
        }
        
        # Charger le modèle
        self._load_model()
        
    def _load_model(self):
        """Charge le modèle selon le runtime"""
        if self.runtime == 'tflite':
            self._load_tflite()
        elif self.runtime == 'onnx':
            self._load_onnx()
        else:
            raise ValueError(f"Runtime non supporté: {self.runtime}")
    
    def _load_tflite(self):
        """Charge un modèle TFLite"""
        try:
            # Essayer d'importer tflite_runtime, sinon fallback sur tensorflow
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            
            if self.use_edgetpu:
                # Chargement direct du délégué EdgeTPU sans pycoral
                try:
                    # Essayer plusieurs noms possibles pour la librairie
                    libs = ['libedgetpu.so.1', 'libedgetpu.so.1.0', 'libedgetpu.so']
                    delegate = None
                    for lib in libs:
                        try:
                            delegate = tflite.load_delegate(lib)
                            print(f"Delegue EdgeTPU charge: {lib}")
                            break
                        except ValueError:
                            continue
                    
                    if delegate is None:
                        raise ValueError("Impossible de charger libedgetpu.so.1")
                        
                    self.interpreter = tflite.Interpreter(
                        model_path=self.model_path,
                        experimental_delegates=[delegate]
                    )
                    print(f"Modele EdgeTPU charge: {os.path.basename(self.model_path)}")
                    
                except Exception as e:
                    print(f"Echec chargement EdgeTPU direct: {e}")
                    # Fallback sur pycoral si installé (ancienne méthode)
                    try:
                        from pycoral.utils import edgetpu
                        self.interpreter = edgetpu.make_interpreter(self.model_path)
                        print(f"Modele EdgeTPU charge (via pycoral): {os.path.basename(self.model_path)}")
                    except ImportError:
                        raise ValueError(f"EdgeTPU requis mais impossible à charger (ni libedgetpu ni pycoral): {e}")
            else:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
                print(f"Modele TFLite charge: {os.path.basename(self.model_path)}")
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        except Exception as e:
            print(f"Erreur chargement TFLite: {e}")
            raise e
    
    def _load_onnx(self):
        """Charge un modèle ONNX avec XNNPACK"""
        try:
            import onnxruntime as ort
            
            # Options pour XNNPACK
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Providers: XNNPACK pour ARM, CPU sinon
            providers = ['CPUExecutionProvider']
            
            # Essayer d'utiliser GPU (CUDA/TensorRT) ou XNNPACK si disponible
            available_providers = ort.get_available_providers()
            
            # Priorité 1: TensorRT (Jetson/NVIDIA)
            if 'TensorrtExecutionProvider' in available_providers:
                providers.insert(0, 'TensorrtExecutionProvider')
                print(f"TensorRT active")
            
            # Priorité 2: CUDA (NVIDIA)
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
                print(f"CUDA active")
                
            # Priorité 3: XNNPACK (ARM CPU)
            if 'XnnpackExecutionProvider' in available_providers:
                providers.insert(0, 'XnnpackExecutionProvider')
                print(f"XNNPACK active")
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Détecter le type de données attendu (float16 ou float32)
            input_type = self.session.get_inputs()[0].type
            self.input_dtype = np.float16 if 'float16' in input_type else np.float32
            
            print(f"Modele ONNX charge: {os.path.basename(self.model_path)}")
            print(f"  Providers: {self.session.get_providers()}")
            print(f"  Input type: {input_type}")
            
        except Exception as e:
            print(f"Erreur chargement ONNX: {e}")
            raise e
    
    def warmup(self, iterations: int = 10):
        """Chauffe le modèle"""
        print(f"Warmup: {iterations} itérations...")
        dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_depth = np.random.randint(0, 5000, (480, 640), dtype=np.uint16)
        dummy_intrinsics = {'k': [600, 0, 320, 0, 600, 240, 0, 0, 1]}
        
        for i in range(iterations):
            self.detect(dummy_rgb, dummy_depth, dummy_intrinsics, record_metrics=False)
        
        print("Warmup termine")
    
    def detect(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
               camera_intrinsics: Dict, record_metrics: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Détecte les balles et calcule les coordonnées 3D
        
        Args:
            rgb_image: Image RGB (OpenCV format)
            depth_image: Image de profondeur
            camera_intrinsics: Intrinsics caméra {'k': [fx, 0, cx, 0, fy, cy, 0, 0, 1]}
            record_metrics: Enregistrer les métriques
            
        Returns:
            detections: Liste des détections [{bbox, confidence, pos_3d}]
            timings: Dictionnaire des temps
        """
        t_start = time.perf_counter()
        
        # --- PREPROCESSING ---
        t_pre_start = time.perf_counter()
        cam_h, cam_w = rgb_image.shape[:2]
        img_resized = cv2.resize(rgb_image, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)
        t_pre_end = time.perf_counter()
        preprocess_time = (t_pre_end - t_pre_start) * 1000
        
        # --- INFERENCE ---
        t_inf_start = time.perf_counter()
        
        if self.runtime == 'tflite':
            detections_raw = self._infer_tflite(input_data)
        else:
            detections_raw = self._infer_onnx(input_data)
        
        t_inf_end = time.perf_counter()
        inference_time = (t_inf_end - t_inf_start) * 1000
        
        # --- POSTPROCESSING ---
        t_post_start = time.perf_counter()
        
        candidates = detections_raw[detections_raw[:, 4] > self.conf_threshold]
        detections = []
        confidence_scores = []
        detections_3d = 0
        
        t_depth_total = 0
        t_proj_total = 0
        
        for det in candidates:
            cx_norm, cy_norm, w_norm, h_norm, conf = det[0], det[1], det[2], det[3], det[4]
            
            # Conversion en coordonnées image
            if w_norm > 1.0:
                scale_x = cam_w / float(self.model_w)
                scale_y = cam_h / float(self.model_h)
                cx = int(cx_norm * scale_x)
                cy = int(cy_norm * scale_y)
                w = int(w_norm * scale_x)
                h = int(h_norm * scale_y)
            else:
                cx = int(cx_norm * cam_w)
                cy = int(cy_norm * cam_h)
                w = int(w_norm * cam_w)
                h = int(h_norm * cam_h)
            
            x_tl = int(cx - w/2)
            y_tl = int(cy - h/2)
            
            detection = {
                'bbox': (x_tl, y_tl, w, h),
                'center': (cx, cy),
                'confidence': float(conf),
                'pos_3d': None
            }
            
            # --- DEPTH LOOKUP ---
            t_depth_start = time.perf_counter()
            
            if depth_image is not None:
                d_h, d_w = depth_image.shape[:2]
                u_depth = int(cx * (d_w / cam_w))
                v_depth = int(cy * (d_h / cam_h))
                
                if 0 <= u_depth < d_w and 0 <= v_depth < d_h:
                    dist_z = depth_image[v_depth, u_depth]
                    
                    # Convertir en mètres si nécessaire (RealSense donne en mm)
                    dist_z_m = dist_z / 1000.0 if dist_z > 100 else dist_z
                    
                    t_depth_end = time.perf_counter()
                    t_depth_total += (t_depth_end - t_depth_start) * 1000
                    
                    # --- 3D PROJECTION ---
                    if 0.1 < dist_z_m < 10.0 and not math.isnan(dist_z_m):
                        t_proj_start = time.perf_counter()
                        
                        # Extraire les intrinsics
                        k = camera_intrinsics['k']
                        fx = k[0]
                        cx_opt = k[2]
                        fy = k[4]
                        cy_opt = k[5]
                        
                        # Projection 3D
                        raw_x = (u_depth - cx_opt) * dist_z_m / fx
                        raw_y = (v_depth - cy_opt) * dist_z_m / fy
                        raw_z = dist_z_m
                        
                        # Transformation vers base_footprint (comme dans b_d.py)
                        detection['pos_3d'] = {
                            'x': float(raw_z),
                            'y': -float(raw_x),
                            'z': float(raw_y)
                        }
                        
                        detections_3d += 1
                        
                        t_proj_end = time.perf_counter()
                        t_proj_total += (t_proj_end - t_proj_start) * 1000
            
            detections.append(detection)
            confidence_scores.append(float(conf))
        
        t_post_end = time.perf_counter()
        postprocess_time = (t_post_end - t_post_start) * 1000
        
        t_end = time.perf_counter()
        total_time = (t_end - t_start) * 1000
        
        timings = {
            'preprocess': preprocess_time,
            'inference': inference_time,
            'postprocess': postprocess_time,
            'depth_lookup': t_depth_total,
            'projection_3d': t_proj_total,
            'total': total_time
        }
        
        # Enregistrer les métriques
        if record_metrics:
            self.metrics['preprocessing_times'].append(preprocess_time)
            self.metrics['inference_times'].append(inference_time)
            self.metrics['postprocessing_times'].append(postprocess_time)
            self.metrics['depth_lookup_times'].append(t_depth_total)
            self.metrics['projection_3d_times'].append(t_proj_total)
            self.metrics['total_times'].append(total_time)
            self.metrics['detection_counts'].append(len(detections))
            self.metrics['detection_3d_counts'].append(detections_3d)
            self.metrics['confidence_scores'].extend(confidence_scores)
            # Meilleure confiance de cette frame (ou 0 si aucune détection)
            max_conf = max(confidence_scores) if confidence_scores else 0.0
            self.metrics['max_confidence_scores'].append(max_conf)
            
            # Métriques système
            process = psutil.Process(os.getpid())
            self.metrics['memory_usage'].append(process.memory_info().rss / 1024 / 1024)
            self.metrics['cpu_usage'].append(process.cpu_percent(interval=0.01))
        
        return detections, timings
    
    def _infer_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """Inférence TFLite"""
        # Quantization si nécessaire
        input_d = self.input_details[0]
        if input_d['dtype'] == np.int8 or input_d['dtype'] == np.uint8:
            scale, zero = input_d['quantization']
            input_data = (input_data / 255.0 / scale + zero).astype(input_d['dtype'])
        else:
            input_data = (input_data.astype(np.float32) / 255.0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_d = self.output_details[0]
        raw = self.interpreter.get_tensor(output_d['index'])[0]
        
        # Dequantization si nécessaire
        if output_d['dtype'] == np.int8 or output_d['dtype'] == np.uint8:
            scale, zero = output_d['quantization']
            return (raw.astype(np.float32) - zero) * scale
        else:
            return raw
    
    def _infer_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """Inférence ONNX"""
        # ONNX attend NCHW (batch, channels, height, width)
        # input_data est en NHWC (batch, height, width, channels)
        input_data = input_data.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (0, 3, 1, 2))  # NHWC -> NCHW
        
        # Convertir au type attendu (float16 ou float32)
        if self.input_dtype == np.float16:
            input_data = input_data.astype(np.float16)
        
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0][0]
    
    def get_summary_metrics(self) -> Dict:
        """Calcule les métriques résumées"""
        if not self.metrics['inference_times']:
            return {}
        
        def safe_mean(lst):
            return np.mean(lst) if lst else 0.0
        
        def safe_std(lst):
            return np.std(lst) if lst else 0.0
        
        def safe_percentile(lst, p):
            return np.percentile(lst, p) if lst else 0.0
        
        summary = {
            # Temps d'inférence
            'inference_time_mean': safe_mean(self.metrics['inference_times']),
            'inference_time_std': safe_std(self.metrics['inference_times']),
            'inference_time_min': min(self.metrics['inference_times']) if self.metrics['inference_times'] else 0,
            'inference_time_max': max(self.metrics['inference_times']) if self.metrics['inference_times'] else 0,
            'inference_time_p50': safe_percentile(self.metrics['inference_times'], 50),
            'inference_time_p95': safe_percentile(self.metrics['inference_times'], 95),
            'inference_time_p99': safe_percentile(self.metrics['inference_times'], 99),
            
            # FPS
            'fps_mean': 1000.0 / safe_mean(self.metrics['total_times']) if self.metrics['total_times'] else 0,
            'fps_min': 1000.0 / max(self.metrics['total_times']) if self.metrics['total_times'] else 0,
            'fps_max': 1000.0 / min(self.metrics['total_times']) if self.metrics['total_times'] else 0,
            
            # Temps total et composants
            'total_time_mean': safe_mean(self.metrics['total_times']),
            'total_time_std': safe_std(self.metrics['total_times']),
            'preprocess_time_mean': safe_mean(self.metrics['preprocessing_times']),
            'postprocess_time_mean': safe_mean(self.metrics['postprocessing_times']),
            'depth_lookup_time_mean': safe_mean(self.metrics['depth_lookup_times']),
            'projection_3d_time_mean': safe_mean(self.metrics['projection_3d_times']),
            
            # Détections
            'detection_count_mean': safe_mean(self.metrics['detection_counts']),
            'detection_3d_count_mean': safe_mean(self.metrics['detection_3d_counts']),
            'detection_count_total': sum(self.metrics['detection_counts']),
            'detection_3d_count_total': sum(self.metrics['detection_3d_counts']),
            
            # Confiance
            'confidence_mean': safe_mean(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0,
            'confidence_std': safe_std(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0,
            
            # Meilleure confiance par frame
            'max_confidence_mean': safe_mean(self.metrics['max_confidence_scores']) if self.metrics['max_confidence_scores'] else 0,
            'max_confidence_std': safe_std(self.metrics['max_confidence_scores']) if self.metrics['max_confidence_scores'] else 0,
            'max_confidence_min': min(self.metrics['max_confidence_scores']) if self.metrics['max_confidence_scores'] else 0,
            'max_confidence_max': max(self.metrics['max_confidence_scores']) if self.metrics['max_confidence_scores'] else 0,
            
            # Ressources
            'memory_usage_mean': safe_mean(self.metrics['memory_usage']),
            'memory_usage_max': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            'cpu_usage_mean': safe_mean(self.metrics['cpu_usage']),
            'cpu_usage_max': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            
            # Métadonnées
            'total_iterations': len(self.metrics['inference_times']),
        }
        
        return summary
    
    def reset_metrics(self):
        """Réinitialise les métriques"""
        for key in self.metrics:
            self.metrics[key] = []


def get_temperature() -> Optional[float]:
    """Récupère la température du système"""
    try:
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                return temp
        
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
        
        return None
    except:
        return None
