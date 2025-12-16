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
        
        # Timer à 15Hz (suffisant pour le Pi et plus stable que 30Hz)
        # Si vous voulez 30Hz, mettez 0.033
        self.timer = self.create_timer(0.066, self.timer_callback)
        
        # --- 2. CONFIGURATION REALSENSE ---
        self.pipe = rs.pipeline()
        self.config = rs.config()
        
        # On garde votre résolution 424x240 (Très bien pour le Pi)
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

        # Objet d'alignement : Aligner la Profondeur VERS la Couleur
        self.align = rs.align(rs.stream.color)

        # Démarrage du flux
        profile = self.pipe.start(self.config)
        
        # --- 3. CORRECTION MAJEURE : INTRINSÈQUES ---
        # Puisqu'on aligne sur la couleur, on doit récupérer les infos de la lentille COULEUR
        stream_profile = profile.get_stream(rs.stream.color) 
        self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()

        # --- 4. STABILISATION (Anti-Crash USB) ---
        try:
            # On récupère le capteur RGB
            color_sensor = profile.get_device().query_sensors()[1]
            
            # Désactivation de l'exposition auto pour éviter la saturation USB
            color_sensor.set_option(rs.option.enable_auto_exposure, 0) # 0 = Off
            # Réglage manuel (Ajustez 150.0 si l'image est trop sombre/claire)
            color_sensor.set_option(rs.option.exposure, 150.0) 
            
            self.get_logger().info("✅ Auto-Exposure désactivé pour stabilité USB.")
        except Exception as e:
            self.get_logger().warn(f"⚠️ Impossible de configurer l'exposition : {e}")

        self.bridge = CvBridge()
        self.get_logger().info("Publisher RealSense Démarré (Mode Aligné).")

    def timer_callback(self):
        try:
            # 1. Attente des frames
            frames = self.pipe.wait_for_frames()
        
            # 2. ALIGNEMENT (Indispensable)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return
            
            # 3. Conversion Numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 4. Préparation du Header (Timestamp unique pour tout le monde)
            header = Header()
            header.frame_id = "camera_color_optical_frame" # Frame standard pour RGB
            header.stamp = self.get_clock().now().to_msg()

            # 5. Conversion ROS
            # CORRECTION : Encoding bgr8 (et non rgb8) pour OpenCV
            ros_color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            
            # 6. Camera Info
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
            # On attrape les erreurs de timeout sans crasher tout le node
            self.get_logger().warn(f"Erreur Frame RealSense : {e}")

    def get_camera_info_msg(self, intrinsics):
        camera_info_msg = CameraInfo()
        camera_info_msg.width = intrinsics.width
        camera_info_msg.height = intrinsics.height

        # Matrice Intrinsèque K [fx, 0, cx, 0, fy, cy, 0, 0, 1]
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
        
        # Matrice de Projection P (identique à K pour une caméra simple sans rectification stéréo complexe)
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