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
        
        # Timer : 30 FPS (0.033s). Si le Pi rame trop, passez à 0.066 (15 FPS).
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        # --- 2. CONFIGURATION REALSENSE PIPELINE ---
        self.pipe = rs.pipeline()
        self.config = rs.config()
        
        # Résolution 424x240 (Optimisé pour Raspberry Pi et YOLO)
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

        # Objet d'alignement : Indispensable pour que le pixel (x,y) couleur corresponde au pixel (x,y) profondeur
        self.align = rs.align(rs.stream.color)

        # Démarrage du flux
        profile = self.pipe.start(self.config)
        
        # --- 3. CORRECTION OPTIQUE (INTRINSÈQUES) ---
        # On récupère les infos de la caméra COULEUR car on aligne la profondeur dessus.
        stream_profile = profile.get_stream(rs.stream.color) 
        self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()

        # --- 4. CONFIGURATION MATÉRIELLE AVANCÉE ---
        try:
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            color_sensor = device.query_sensors()[1] # Souvent l'index 1 pour RGB sur D435i

            # A. STABILITÉ USB (Capteur RGB)
            # Désactive l'auto-exposure pour éviter de surcharger le CPU/USB du Pi
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0) # OFF
                # Réglage manuel de luminosité (150 est une moyenne, ajustez si image noire/blanche)
                color_sensor.set_option(rs.option.exposure, 150.0)
                self.get_logger().info("✅ RGB : Exposition Manuelle activée (Stabilité USB).")

            # B. QUALITÉ PROFONDEUR (Capteur Depth) - Pour voir la balle !
            # Active le projecteur Laser IR à fond
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0) # ON
                depth_sensor.set_option(rs.option.laser_power, 300.0)   # Puissance Max (souvent 360 max)
                self.get_logger().info("✅ Depth : Laser Emitter activé (Puissance 300).")

            # Applique le preset "High Density" (4.0) pour remplir les trous sur les objets
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 4.0) 
                self.get_logger().info("✅ Depth : Preset 'High Density' appliqué.")

        except Exception as e:
            self.get_logger().warn(f"⚠️ Avertissement config capteurs : {e}")

        self.bridge = CvBridge()
        self.get_logger().info("Publisher RealSense prêt et configuré.")

    def timer_callback(self):
        try:
            # 1. Attente des frames
            frames = self.pipe.wait_for_frames()
        
            # 2. ALIGNEMENT (Indispensable pour la précision XYZ)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return
            
            # 3. Conversion en Array Numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 4. Création du Header commun (Même temps pour tout le monde)
            header = Header()
            header.frame_id = "camera_color_optical_frame" # Repère optique standard
            header.stamp = self.get_clock().now().to_msg()

            # 5. Conversion en Messages ROS
            # 'bgr8' pour OpenCV Couleur
            ros_color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            # 'passthrough' garde le format 16-bit (mm) brut pour la profondeur
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            
            # 6. Message Camera Info
            camera_info_msg = self.get_camera_info_msg(self.intrinsics)

            # Assignation des headers
            camera_info_msg.header = header
            ros_color_image_msg.header = header
            ros_depth_image_msg.header = header

            # 7. Publication
            self.camera_info_publisher_.publish(camera_info_msg)
            self.color_image_publisher_.publish(ros_color_image_msg)
            self.depth_image_publisher_.publish(ros_depth_image_msg)

        except RuntimeError as e:
            # Gestion des timeouts USB sans crasher
            self.get_logger().warn(f"Frame drop (USB busy?): {e}")

    def get_camera_info_msg(self, intrinsics):
        camera_info_msg = CameraInfo()
        camera_info_msg.width = intrinsics.width
        camera_info_msg.height = intrinsics.height

        # Matrice K (Intrinsèque)
        camera_info_msg.k = [
            float(intrinsics.fx), 0.0, float(intrinsics.ppx),
            0.0, float(intrinsics.fy), float(intrinsics.ppy),
            0.0, 0.0, 1.0
        ]
        
        # Modèle de distorsion
        distortion_model_map = {
            rs.distortion.none: 'none',
            rs.distortion.brown_conrady: 'plumb_bob',
            rs.distortion.inverse_brown_conrady: 'plumb_bob',
            rs.distortion.ftheta: 'fisheye',
            rs.distortion.kannala_brandt4: 'fisheye',
        }
        camera_info_msg.distortion_model = distortion_model_map.get(intrinsics.model, 'plumb_bob')
        
        # Coefficients D
        camera_info_msg.d = list(intrinsics.coeffs)
        
        # Matrice P (Projection)
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