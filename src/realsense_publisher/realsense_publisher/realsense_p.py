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
        self.depth_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Depth', 10)
        self.color_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Color', 10)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, 'Realsense/CameraInfo', 10)
        
        # Timer 6 FPS
        self.timer = self.create_timer(0.167, self.timer_callback)
        self.bridge = CvBridge()

        # --- 2. CONFIGURATION REALSENSE ---
        self.pipe = rs.pipeline()
        self.config = rs.config()
        
        # 6 FPS (Stabilité USB maximale)
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

        # --- 3. FILTRES LOGICIELS (C'est la clé !) ---
        # 1 = mode 'flling' (remplit les zones vides avec la valeur voisine)
        self.hole_filling = rs.hole_filling_filter(1) 
        
        # Alignement Depth -> Color
        self.align = rs.align(rs.stream.color)

        self.get_logger().info("Démarrage Realsense (Mode Stable + Hole Filling)...")

        try:
            # Démarrage du flux
            profile = self.pipe.start(self.config)
            
            # Récupération Intrinsèques
            stream_profile = profile.get_stream(rs.stream.color) 
            self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
            
            self.get_logger().info("✅ Caméra opérationnelle.")

        except Exception as e:
            self.get_logger().fatal(f"🔥 Erreur Démarrage : {e}")

    def timer_callback(self):
        try:
            # Timeout large
            frames = self.pipe.wait_for_frames(timeout_ms=2000)
            
            # 1. ALIGNEMENT
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return

            # 2. FILTRE HOLE FILLING (CRITIQUE)
            # Transforme les '0' en valeurs estimées.
            # Augmente drastiquement le nombre de pixels valides sur la balle.
            depth_frame = self.hole_filling.process(depth_frame)

            # 3. Conversion
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            header = Header()
            header.frame_id = "camera_color_optical_frame"
            header.stamp = self.get_clock().now().to_msg()

            # Publication
            ros_color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            ros_color_image_msg.header = header
            self.color_image_publisher_.publish(ros_color_image_msg)

            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            ros_depth_image_msg.header = header
            self.depth_image_publisher_.publish(ros_depth_image_msg)
            
            # Info
            camera_info_msg = self.get_camera_info_msg(self.intrinsics)
            camera_info_msg.header = header
            self.camera_info_publisher_.publish(camera_info_msg)

        except RuntimeError:
            pass # On ignore les frames perdues
        except Exception:
            pass

    def get_camera_info_msg(self, intrinsics):
        camera_info_msg = CameraInfo()
        camera_info_msg.width = intrinsics.width
        camera_info_msg.height = intrinsics.height
        camera_info_msg.k = [float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0, 0.0, 1.0]
        camera_info_msg.distortion_model = 'plumb_bob'
        camera_info_msg.d = list(intrinsics.coeffs)
        camera_info_msg.p = [float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0, 0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0, 0.0, 0.0, 1.0, 0.0]
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