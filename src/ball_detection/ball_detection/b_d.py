import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from ament_index_python.packages import get_package_share_directory
import os
# IMPORT INDISPENSABLE POUR GAZEBO
from rclpy.qos import qos_profile_sensor_data

class BallDetectorPointCloud(Node):
    def __init__(self):
        super().__init__('ball_detector_pointcloud')

        # --- CONFIGURATION IA ---
        package_share_directory = get_package_share_directory('ball_detection')
        model_filename = 'yolov5n-int8_320.tflite' # VERIFIE CE NOM
        self.model_path = os.path.join(package_share_directory, 'models', model_filename)

        self.conf_threshold = 0.25 
        self.model_w = 320 
        self.model_h = 320
        self.ball_radius = 0.035
        self.radius_tolerance = 0.02
        self.ransac_iterations = 80
        self.min_inliers = 10 # Réduit un peu pour faciliter la detection

        # IA Init (CPU FORCE)
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.get_logger().info("💻 Mode CPU activé.")
        except Exception as e:
            self.get_logger().error(f"❌ Erreur Init IA: {e}")
            raise e

        # --- ROS SETUP ---
        self.bridge = CvBridge()
        self.pub_ball = self.create_publisher(PointStamped, '/ball_3d', 10)
        self.pub_marker = self.create_publisher(Marker, '/ball_marker', 10)
        self.pub_debug_img = self.create_publisher(Image, '/ball_debug', 10)

        # ⚠️ QOS PROFILE = INDISPENSABLE POUR LA SIMULATION
        self.sub_rgb = message_filters.Subscriber(
            self, Image, '/Realsense/Image/Color', qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, '/Realsense/Image/Depth', qos_profile=qos_profile_sensor_data)
        self.sub_info = message_filters.Subscriber(
            self, CameraInfo, '/Realsense/CameraInfo', qos_profile=qos_profile_sensor_data)

        # ⚠️ SLOP = 10.0 (TOLÉRANCE MAXIMALE AU LAG)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info], 
            queue_size=100, 
            slop=10.0,  # Accepte jusqu'à 10s de décalage
            allow_headerless=True
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info("✅ Algorithme Prêt (Lag-Resistant).")

    def callback(self, rgb_msg, depth_msg, info_msg):
        # Debug simple pour savoir si on reçoit des données
        # self.get_logger().info("📥 Images reçues !", throttle_duration_sec=2.0)

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"❌ Erreur CV Bridge: {e}")
            return

        cam_h, cam_w = cv_rgb.shape[:2]
        debug_image = cv_rgb.copy()

        # IA INFERENCE
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        detections = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        candidates = detections[detections[:, 4] > self.conf_threshold]

        fx = info_msg.k[0]
        fy = info_msg.k[4]
        cx_cam = info_msg.k[2]
        cy_cam = info_msg.k[5]

        final_ball_3d = None

        for det in candidates:
            # Pour simplifier le debug, on prend le premier candidat valide
            mx, my, mw, mh = det[0:4]
            confidence = det[4]
            
            scale_x = cam_w / self.model_w
            scale_y = cam_h / self.model_h
            cx_box = int(mx * scale_x)
            cy_box = int(my * scale_y)
            
            # Simple lecture de la profondeur au centre de la bbox
            # (Plus robuste que le nuage de points complet si ça lague)
            if 0 <= cy_box < cam_h and 0 <= cx_box < cam_w:
                z_mm = cv_depth[cy_box, cx_box]
                
                # Si le pixel central est vide (0), on cherche autour
                if z_mm == 0:
                     # Petite zone 5x5 autour du centre
                     roi = cv_depth[max(0, cy_box-5):min(cam_h, cy_box+5), 
                                    max(0, cx_box-5):min(cam_w, cx_box+5)]
                     if roi.size > 0:
                         z_mm = np.max(roi) # On prend le max pour éviter les trous

                if 0 < z_mm < 5000: # Max 5m
                    z_m = z_mm / 1000.0
                    x_m = (cx_box - cx_cam) * z_m / fx
                    y_m = (cy_box - cy_cam) * z_m / fy
                    
                    final_ball_3d = (x_m, y_m, z_m)
                    
                    # Dessin
                    cv2.circle(debug_image, (cx_box, cy_box), 10, (0, 255, 0), 2)
                    cv2.putText(debug_image, f"{z_m:.2f}m", (cx_box, cy_box-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break # On arrête à la première balle trouvée

        if final_ball_3d:
            gx, gy, gz = final_ball_3d
            self.publish_result(rgb_msg.header, gx, gy, gz)
        
        # Publication Image Debug
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header = rgb_msg.header
            self.pub_debug_img.publish(debug_msg)
        except Exception:
            pass

    def publish_result(self, header, x, y, z):
        pt = PointStamped()
        pt.header = header
        pt.point.x, pt.point.y, pt.point.z = float(x), float(y), float(z)
        self.pub_ball.publish(pt)
        
        marker = Marker()
        marker.header = header
        marker.ns = "ball_3d"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.pub_marker.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
