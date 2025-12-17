import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import time

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloudpublisher')
        
        # --- 1. CONFIGURATION ROS ---
        self.depth_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Depth', 10)
        self.color_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Color', 10)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, 'Realsense/CameraInfo', 10)
        
        # Timer à 6 FPS
        self.timer = self.create_timer(0.167, self.timer_callback)
        
        # --- 2. CONFIGURATION REALSENSE ---
        self.pipe = rs.pipeline()
        self.config = rs.config()
        
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

        # Filtres logiciels
        self.hole_filling = rs.hole_filling_filter(1) 
        self.align = rs.align(rs.stream.color)

        # --- 3. REGLAGES HARDWARE (AVANT LE START) ---
        # C'est ici la correction majeure : on configure le device avant de saturer l'USB
        self.get_logger().info("Recherche du périphérique Realsense...")
        
        try:
            # On récupère le contexte pour accéder au device sans démarrer le pipe
            ctx = rs.context()
            if len(ctx.devices) > 0:
                dev = ctx.devices[0] # La première caméra trouvée
                self.get_logger().info(f"Caméra trouvée : {dev.get_info(rs.camera_info.name)}")
                
                # Réglages Capteur Profondeur
                depth_sensor = dev.first_depth_sensor()
                self.set_option_robust(depth_sensor, rs.option.visual_preset, 4, "High Density Preset")
                self.set_option_robust(depth_sensor, rs.option.laser_power, 360.0, "Laser Power Max")

                # Réglages Capteur Couleur
                # Note: Sur D435i, le capteur couleur est souvent le 2ème (index 1)
                for s in dev.query_sensors():
                    if s.get_info(rs.camera_info.name) == 'RGB Camera':
                        self.set_option_robust(s, rs.option.enable_auto_exposure, 0, "Auto-Exposure OFF")
                        self.set_option_robust(s, rs.option.exposure, 150.0, "Exposure Manual")
            else:
                self.get_logger().warn("⚠️ Aucune caméra Realsense détectée par le contexte !")

        except Exception as e:
            self.get_logger().error(f"Erreur Configuration Initiale : {e}")

        # --- 4. DEMARRAGE DU FLUX ---
        self.get_logger().info("Démarrage du flux vidéo...")
        try:
            profile = self.pipe.start(self.config)
            
            # Récupération Intrinsèques
            stream_profile = profile.get_stream(rs.stream.color) 
            self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
            
            self.bridge = CvBridge()
            self.get_logger().info("✅ Publisher RealSense Démarré & Configuré.")
            
        except Exception as e:
            self.get_logger().fatal(f"Impossible de démarrer le flux : {e}")
            # Si ça plante ici, c'est souvent un problème de câble USB

    def set_option_robust(self, sensor, option, value, name):
        """ Tente d'appliquer une option 3 fois avant d'abandonner """
        if not sensor.supports(option):
            return

        for i in range(3):
            try:
                sensor.set_option(option, value)
                self.get_logger().info(f"   -> OK : {name}")
                return # Succès
            except RuntimeError:
                time.sleep(0.2) # Petite pause et on réessaie
        
        self.get_logger().warn(f"   -> ⚠️ ECHEC : {name} (Timeout)")

    def timer_callback(self):
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=5000)
            
            try:
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    return

                depth_frame = self.hole_filling.process(depth_frame)

            except RuntimeError:
                return

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            header = Header()
            header.frame_id = "camera_color_optical_frame"
            header.stamp = self.get_clock().now().to_msg()

            ros_color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            ros_depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            
            camera_info_msg = self.get_camera_info_msg(self.intrinsics)
            camera_info_msg.header = header
            ros_color_image_msg.header = header
            ros_depth_image_msg.header = header

            self.camera_info_publisher_.publish(camera_info_msg)
            self.color_image_publisher_.publish(ros_color_image_msg)
            self.depth_image_publisher_.publish(ros_depth_image_msg)

        except Exception as e:
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