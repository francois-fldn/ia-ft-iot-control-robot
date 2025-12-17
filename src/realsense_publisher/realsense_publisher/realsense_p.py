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
        
        # --- 1. CONFIGURATION ROS ---
        # Queue size 10 est standard
        self.depth_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Depth', 10)
        self.color_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Color', 10)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, 'Realsense/CameraInfo', 10)
        
        # Timer calé sur 6 FPS (1/6 ≈ 0.167s)
        self.timer = self.create_timer(0.167, self.timer_callback)
        
        self.bridge = CvBridge()

        # --- 2. CONFIGURATION REALSENSE (MINIMALISTE) ---
        self.pipe = rs.pipeline()
        self.config = rs.config()
        
        # On force 6 FPS pour soulager le Raspberry Pi et l'USB
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

        # Outils logiciels (ne touchent pas au hardware)
        self.hole_filling = rs.hole_filling_filter(1) 
        self.align = rs.align(rs.stream.color)

        self.get_logger().info("Démarrage du flux (Mode Stable)...")

        try:
            # Démarrage simple, sans options exotiques
            profile = self.pipe.start(self.config)
            
            # Récupération des infos techniques de la lentille (Intrinsèques)
            stream_profile = profile.get_stream(rs.stream.color) 
            self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
            
            self.get_logger().info("✅ Caméra démarrée avec succès.")

        except Exception as e:
            self.get_logger().fatal(f"❌ Impossible de démarrer la caméra : {e}")
            self.get_logger().fatal("👉 Si ce message persiste : Débranchez/Rebranchez l'USB.")

    def timer_callback(self):
        try:
            # Timeout généreux (2s) pour éviter les crashs si une frame saute
            frames = self.pipe.wait_for_frames(timeout_ms=2000)
            
            # 1. Alignement (Depth -> Color)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return

            # 2. Filtre Logiciel (Bouchage des trous)
            # On le garde car c'est purement mathématique (CPU Pi) et aide la détection
            depth_frame = self.hole_filling.process(depth_frame)

            # 3. Conversion en images Numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 4. Création des messages ROS
            header = Header()
            header.frame_id = "camera_color_optical_frame"
            header.stamp = self.get_clock().now().to_msg()

            # Message Couleur
            ros_color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            ros_color_image_msg.header = header
            
            # Message Profondeur
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            ros_depth_image_msg.header = header
            
            # Message Info Caméra
            camera_info_msg = self.get_camera_info_msg(self.intrinsics)
            camera_info_msg.header = header

            # 5. Publication
            self.color_image_publisher_.publish(ros_color_image_msg)
            self.depth_image_publisher_.publish(ros_depth_image_msg)
            self.camera_info_publisher_.publish(camera_info_msg)

        except RuntimeError:
            # On ignore silencieusement les frames ratées (fréquent sur USB Pi4)
            pass
        except Exception as e:
            self.get_logger().error(f"Erreur inattendue : {e}")

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
        
        # Modèle de distorsion standard
        camera_info_msg.distortion_model = 'plumb_bob'
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