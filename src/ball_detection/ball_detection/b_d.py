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

class BallDetectorPointCloud(Node):
    def __init__(self):
        super().__init__('ball_detector_pointcloud')

        # --- CONFIGURATION IA ---
        package_share_directory = get_package_share_directory('ball_detection')
        model_filename = 'yolov5n-int8_edgetpu320.tflite' 
        self.model_path = os.path.join(package_share_directory, 'models', model_filename)

        self.conf_threshold = 0.25 
        self.model_w = 320 
        self.model_h = 320

        # IA Init
        try:
            delegate = tflite.load_delegate('libedgetpu.so.1')
            self.interpreter = tflite.Interpreter(model_path=self.model_path, experimental_delegates=[delegate])
            self.get_logger().info("✅ Mode CORAL activé.")
        except Exception:
            self.get_logger().warn("⚠️ Mode CPU.")
            self.interpreter = tflite.Interpreter(model_path=self.model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # --- ROS SETUP ---
        self.bridge = CvBridge()
        self.pub_ball = self.create_publisher(PointStamped, '/ball_3d', 10)
        self.pub_marker = self.create_publisher(Marker, '/ball_marker', 10)
        self.pub_debug_img = self.create_publisher(Image, '/ball_debug', 10)

        self.sub_rgb = message_filters.Subscriber(self, Image, '/Realsense/Image/Color')
        self.sub_depth = message_filters.Subscriber(self, Image, '/Realsense/Image/Depth')
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/Realsense/CameraInfo')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info("✅ Algorithme Nuage de Points 3D Prêt.")

    def callback(self, rgb_msg, depth_msg, info_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception:
            return

        cam_h, cam_w = cv_rgb.shape[:2]
        debug_image = cv_rgb.copy()

        # --- 1. IA INFERENCE ---
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        detections = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        candidates = detections[detections[:, 4] > self.conf_threshold]

        # Récupération de la matrice intrinsèque K (pour les calculs 3D)
        fx = info_msg.k[0]
        fy = info_msg.k[4]
        cx_cam = info_msg.k[2]
        cy_cam = info_msg.k[5]

        best_z = 999.0 # Plus c'est petit, mieux c'est
        final_ball_3d = None

        for det in candidates:
            score = det[4]
            mx, my, mw, mh = det[0:4]
            
            # Bounding Box Image
            scale_x = cam_w / self.model_w
            scale_y = cam_h / self.model_h
            cx_box = int(mx * scale_x)
            cy_box = int(my * scale_y)
            w = int(mw * scale_x)
            h = int(mh * scale_y)

            # Découpage ROI (Region of Interest)
            x_min = max(0, int(cx_box - w/2))
            x_max = min(cam_w, int(cx_box + w/2))
            y_min = max(0, int(cy_box - h/2))
            y_max = min(cam_h, int(cy_box + h/2))

            # Crop de la profondeur
            depth_roi = cv_depth[y_min:y_max, x_min:x_max]
            
            # --- 2. TRAITEMENT NUAGE DE POINTS (La partie importante) ---
            
            # On ignore les pixels vides (0) et trop loins (> 1.5m)
            valid_mask = (depth_roi > 0) & (depth_roi < 1500)
            valid_depths = depth_roi[valid_mask]

            if len(valid_depths) > 10:
                # A. Trouver le "plan avant" de l'objet (Front Cluster)
                # On trie les pixels. Les plus petits = les plus proches = la surface de la balle
                sorted_depths = np.sort(valid_depths)
                
                # On prend les 20% les plus proches. 
                # C'est la technique anti-mur : le mur est toujours derrière la balle.
                limit_idx = int(len(sorted_depths) * 0.20)
                if limit_idx < 1: limit_idx = 1
                
                # On calcule la distance Z moyenne de cette "face avant"
                closest_cluster = sorted_depths[:limit_idx]
                z_mm = np.mean(closest_cluster)
                z_m = z_mm / 1000.0

                # B. Calcul des Coordonnées X, Y réelles (Déprojection)
                # On utilise le centre de la boite (cx_box, cy_box) et la distance trouvée Z
                # Formule du modèle Sténopé (Pinhole)
                x_m = (cx_box - cx_cam) * z_m / fx
                y_m = (cy_box - cy_cam) * z_m / fy

                # C. Validation
                # On garde la balle la plus proche et la plus confiante
                if z_m < best_z:
                    best_z = z_m
                    final_ball_3d = (x_m, y_m, z_m)
                    
                    # Dessin Vert
                    cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(debug_image, f"{z_m:.2f}m", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Dessin Rouge (Rejeté)
                cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

        # --- 3. PUBLICATION ---
        if final_ball_3d:
            gx, gy, gz = final_ball_3d
            self.publish_result(rgb_msg.header, gx, gy, gz)
            self.get_logger().info(f"📍 Balle 3D: X={gx:.2f} Y={gy:.2f} Z={gz:.2f}")

        # Debug Image
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
        
        # Marker RViz (Sphère 3D)
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
        marker.scale.x = 0.1 # 10cm de diamètre
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