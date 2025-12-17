import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloudpublisher')
        
        # --- 1. CONFIGURATION ROS PUBLISHERS ---
        self.depth_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Depth', 10)
        self.color_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Color', 10)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, 'Realsense/CameraInfo', 10)
        
        # Timer à 15Hz (0.066s) -> Compromis idéal pour Pi 4 + Traitement Python
        self.timer = self.create_timer(0.066, self.timer_callback)
        
        # --- 2. CONFIGURATION REALSENSE ---
        self.pipe = rs.pipeline()
        self.config = rs.config()
        
        # Résolution 424x240 (Parfait pour le Pi4, réduit la charge CPU)
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)

        # --- NOUVEAU : FILTRES ---
        # Filtre "Hole Filling" : Indispensable pour que la balle soit "pleine" et pas un anneau
        self.hole_filling = rs.hole_filling_filter(1) # 1 = ffill (bouchage modéré)

        # Alignement : Profondeur VERS Couleur
        self.align = rs.align(rs.stream.color)

        # Démarrage du flux
        profile = self.pipe.start(self.config)
        
        # --- 3. REGLAGES HARDWARE AVANCÉS (CRITIQUE POUR LA BALLE) ---
        depth_sensor = profile.get_device().first_depth_sensor()
        
        if depth_sensor.supports(rs.option.visual_preset):
            # Preset 4 = "High Density". Privilégie le remplissage des objets proches.
            depth_sensor.set_option(rs.option.visual_preset, 4)
            self.get_logger().info("✅ Preset 'High Density' activé.")
            
        if depth_sensor.supports(rs.option.laser_power):
            # Laser à fond (360) pour bien voir les objets sombres/mats
            depth_sensor.set_option(rs.option.laser_power, 360.0)
            self.get_logger().info("✅ Laser Power set to Max (360mW).")

        # --- 4. INTRINSÈQUES (Alignées sur Color) ---
        stream_profile = profile.get_stream(rs.stream.color) 
        self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()

        # --- 5. REGLAGES RGB (Anti-Crash USB & Luminosité) ---
        try:
            color_sensor = profile.get_device().query_sensors()[1]
            # Désactivation exposition auto pour stabilité USB + éviter le flou de mouvement
            color_sensor.set_option(rs.option.enable_auto_exposure, 0) 
            # Exposition manuelle (150 est standard, baissez à 100 si flou, montez à 300 si trop noir)
            color_sensor.set_option(rs.option.exposure, 150.0) 
            self.get_logger().info("✅ Auto-Exposure RGB désactivé.")
        except Exception as e:
            self.get_logger().warn(f"⚠️ Warning Exposure: {e}")

        self.bridge = CvBridge()
        self.get_logger().info("Publisher RealSense Démarré avec filtres Balle.")

    def timer_callback(self):
        try:
            # 1. Attente des frames
            frames = self.pipe.wait_for_frames(timeout_ms=5000)
        
            # 2. ALIGNEMENT
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return
            
            # --- 3. APPLICATION DU FILTRE HOLE FILLING ---
            # On bouche les trous APRES l'alignement (plus simple à gérer sur Pi)
            depth_frame = self.hole_filling.process(depth_frame)

            # 4. Conversion Numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 5. Header (Timestamp synchronisé)
            header = Header()
            header.frame_id = "camera_color_optical_frame"
            header.stamp = self.get_clock().now().to_msg()

            # 6. Conversion ROS
            ros_color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            
            # 7. Camera Info
            camera_info_msg = self.get_camera_info_msg(self.intrinsics)

            # Assignation des headers
            camera_info_msg.header = header
            ros_color_image_msg.header = header
            ros_depth_image_msg.header = header

            # 8. Publication
            self.camera_info_publisher_.publish(camera_info_msg)
            self.color_image_publisher_.publish(ros_color_image_msg)
            self.depth_image_publisher_.publish(ros_depth_image_msg)

        except RuntimeError as e:
            self.get_logger().warn(f"Frame Timeout/Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Erreur inattendue: {e}")

    def get_camera_info_msg(self, intrinsics):
        camera_info_msg = CameraInfo()
        camera_info_msg.width = intrinsics.width
        camera_info_msg.height = intrinsics.height

        # K: Matrice Intrinsèque
        camera_info_msg.k = [
            float(intrinsics.fx), 0.0, float(intrinsics.ppx),
            0.0, float(intrinsics.fy), float(intrinsics.ppy),
            0.0, 0.0, 1.0
        ]
        
        # Distortion
        distortion_model_map = {
            rs.distortion.none: 'none',
            rs.distortion.brown_conrady: 'plumb_bob',
            rs.distortion.inverse_brown_conrady: 'plumb_bob',
            rs.distortion.ftheta: 'fisheye',
            rs.distortion.kannala_brandt4: 'fisheye',
        }
        camera_info_msg.distortion_model = distortion_model_map.get(intrinsics.model, 'plumb_bob')
        camera_info_msg.d = list(intrinsics.coeffs)
        
        # P: Matrice Projection
        camera_info_msg.p = [
            float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0,
            0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        return camera_info_msg

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()