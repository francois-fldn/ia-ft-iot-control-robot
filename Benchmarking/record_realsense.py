import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pickle
import gzip
from pathlib import Path
from datetime import datetime
import numpy as np


class RealSenseRecorder(Node):
    # Enregistre les données RealSense dans un fichier
    
    def __init__(self, output_file: str, max_frames: int = 300, use_realsense_topics: bool = True):

        super().__init__('realsense_recorder')
        
        self.output_file = Path(output_file)
        self.max_frames = max_frames
        self.bridge = CvBridge()
        
        # Données enregistrées
        self.frames = []
        self.camera_info = None
        self.frame_count = 0
        
        # Subscribers
        if use_realsense_topics:
            self.get_logger().info("Mode RealSense: topics Realsense/*")
            self.sub_rgb = self.create_subscription(
                Image, 'Realsense/Image/Color', self.callback_rgb, 10)
            self.sub_depth = self.create_subscription(
                Image, 'Realsense/Image/Depth', self.callback_depth, 10)
            self.sub_info = self.create_subscription(
                CameraInfo, 'Realsense/CameraInfo', self.callback_info, 10)
        else:
            self.get_logger().info("Mode Gazebo: topics *_camera/*")
            self.sub_rgb = self.create_subscription(
                Image, 'rgb_camera/image', self.callback_rgb, 10)
            self.sub_depth = self.create_subscription(
                Image, 'depth_camera/image', self.callback_depth, 10)
            self.sub_info = self.create_subscription(
                CameraInfo, 'rgb_camera/camera_info', self.callback_info, 10)
        
        self.latest_rgb = None
        self.latest_depth = None
        
        self.get_logger().info(f"Enregistrement demarre - Max {max_frames} frames")
        self.get_logger().info(f"Sortie: {self.output_file}")
    
    def callback_info(self, msg):
        # Stocke les infos caméra (une seule fois)
        if self.camera_info is None:
            self.camera_info = {
                'width': msg.width,
                'height': msg.height,
                'k': list(msg.k),
                'distortion_model': msg.distortion_model,
                'd': list(msg.d),
                'p': list(msg.p)
            }
            self.get_logger().info(f"CameraInfo capture: {msg.width}x{msg.height}")
    
    def callback_depth(self, msg):
        # Stocke la dernière image de profondeur
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Erreur depth: {e}")
    
    def callback_rgb(self, msg):
        # Enregistre une frame complète (RGB + Depth)
        if self.frame_count >= self.max_frames:
            if self.frame_count == self.max_frames:
                self.get_logger().info("Nombre max de frames atteint - Sauvegarde...")
                self.save_dataset()
                self.frame_count += 1  # Pour ne sauvegarder qu'une fois
            return
        
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Erreur RGB: {e}")
            return
        
        # Attendre que depth et camera_info soient disponibles
        if self.latest_depth is None or self.camera_info is None:
            return
        
        # Enregistrer la frame
        frame_data = {
            'frame_id': self.frame_count,
            'rgb': rgb.copy(),
            'depth': self.latest_depth.copy(),
            'timestamp': self.get_clock().now().nanoseconds
        }
        
        self.frames.append(frame_data)
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            self.get_logger().info(f"{self.frame_count}/{self.max_frames} frames enregistrees")
    
    def save_dataset(self):
        # Sauvegarde le dataset dans un fichier compressé
        if not self.frames:
            self.get_logger().info("Aucune frame a sauvegarder!")
            return
        
        dataset = {
            'camera_info': self.camera_info,
            'frames': self.frames,
            'metadata': {
                'num_frames': len(self.frames),
                'recording_date': datetime.now().isoformat(),
                'resolution': f"{self.camera_info['width']}x{self.camera_info['height']}"
            }
        }
        
        # Créer le dossier si nécessaire
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder avec compression
        self.get_logger().info("Compression et sauvegarde...")
        with gzip.open(self.output_file, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = self.output_file.stat().st_size / (1024 * 1024)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("ENREGISTREMENT TERMINE")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Fichier: {self.output_file}")
        self.get_logger().info(f"Frames: {len(self.frames)}")
        self.get_logger().info(f"Taille: {file_size_mb:.1f} MB")
        self.get_logger().info(f"Resolution: {self.camera_info['width']}x{self.camera_info['height']}")
        self.get_logger().info("=" * 60)
        
        # Arrêter le node
        rclpy.shutdown()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enregistre les données RealSense pour benchmarking")
    parser.add_argument('--output', '-o', type=str, 
                       default='realsense_dataset.pkl.gz',
                       help='Fichier de sortie (défaut: realsense_dataset.pkl.gz)')
    parser.add_argument('--frames', '-n', type=int, default=300,
                       help='Nombre de frames à enregistrer (défaut: 300)')
    parser.add_argument('--gazebo', action='store_true',
                       help='Utiliser les topics Gazebo au lieu de RealSense')
    
    args, ros_args = parser.parse_known_args()
    
    rclpy.init(args=ros_args)
    
    recorder = RealSenseRecorder(
        output_file=args.output,
        max_frames=args.frames,
        use_realsense_topics=not args.gazebo
    )
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        recorder.get_logger().warning("Enregistrement interrompu!")
        if recorder.frames:
            recorder.save_dataset()
    finally:
        recorder.destroy_node()


if __name__ == '__main__':
    main()
