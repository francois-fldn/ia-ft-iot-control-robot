import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Header
import pyrealsense2 as rs
import numpy as np
from rclpy.qos import qos_profile_sensor_data

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloudpublisher')
        
        # Publishers
        self.depth_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Depth', qos_profile_sensor_data)
        self.color_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Color', qos_profile_sensor_data)
        self.camera_info_publisher_ = self.create_publisher(CameraInfo, 'Realsense/CameraInfo', qos_profile_sensor_data)
        
        # Timer 6 FPS (1/6 ≈ 0.167)
        self.timer = self.create_timer(0.167, self.timer_callback)
        self.bridge = CvBridge()

        # Realsense Config
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

        # Alignement CRITIQUE : Depth -> Color
        self.align = rs.align(rs.stream.color)
        
        # Filtre pour boucher les trous (aide à densifier le nuage)
        self.hole_filling = rs.hole_filling_filter(1)

        self.get_logger().info("Démarrage Realsense (Mode PointCloud Ready)...")
        try:
            profile = self.pipe.start(self.config)
            stream_profile = profile.get_stream(rs.stream.color) 
            self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
            self.get_logger().info("Caméra OK.")
        except Exception as e:
            self.get_logger().fatal(f"Erreur Caméra: {e}")

    def timer_callback(self):
        try:
            frames = self.pipe.poll_for_frames()
            # aligned_frames = self.align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                self.get_logger().warning(f"Aucune frame")
                return

            # Filtre
            depth_frame = self.hole_filling.process(depth_frame)

            # Conversion
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            header = Header()
            header.frame_id = "camera_color_optical_frame"
            header.stamp = self.get_clock().now().to_msg()

            # Publication
            msg_color = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            msg_color.header = header
            self.color_image_publisher_.publish(msg_color)

            msg_depth = self.bridge.cv2_to_imgmsg(depth_image, "passthrough")
            msg_depth.header = header
            self.depth_image_publisher_.publish(msg_depth)

            cam_info = self.get_camera_info_msg(self.intrinsics)
            cam_info.header = header
            self.camera_info_publisher_.publish(cam_info)

        except Exception as e:
            self.get_logger().error(f"{e}")

    def get_camera_info_msg(self, intrinsics):
        info = CameraInfo()
        info.width = intrinsics.width
        info.height = intrinsics.height
        info.k = [float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0, 0.0, 1.0]
        info.distortion_model = 'plumb_bob'
        info.d = list(intrinsics.coeffs)
        info.p = [float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0, 0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

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
