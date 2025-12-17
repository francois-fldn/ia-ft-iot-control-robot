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

        # Paramètres de la balle (AJUSTE SELON TA BALLE RÉELLE)
        self.ball_radius = 0.035  # 3.5cm de rayon (7cm de diamètre)
        self.radius_tolerance = 0.02  # ±2cm de tolérance
        self.ransac_iterations = 80  # Nombre d'itérations RANSAC
        self.min_inliers = 20  # Minimum de points cohérents pour valider

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
        self.get_logger().info("✅ Algorithme RANSAC Sphère 3D Prêt.")

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

        # Récupération de la matrice intrinsèque K
        fx = info_msg.k[0]
        fy = info_msg.k[4]
        cx_cam = info_msg.k[2]
        cy_cam = info_msg.k[5]

        best_score = 0  # Score = inliers * confidence
        final_ball_3d = None

        for det in candidates:
            confidence = det[4]
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

            # --- 2. CONSTRUCTION DU NUAGE DE POINTS 3D ---
            points_3d = []
            for i in range(y_min, y_max):
                for j in range(x_min, x_max):
                    z_mm = cv_depth[i, j]
                    # Filtre: depth valide et < 1.5m
                    if 0 < z_mm < 1500:
                        z_m = z_mm / 1000.0
                        # Déprojection 3D (modèle sténopé)
                        x_m = (j - cx_cam) * z_m / fx
                        y_m = (i - cy_cam) * z_m / fy
                        points_3d.append([x_m, y_m, z_m])
            
            if len(points_3d) < 30:
                # Pas assez de points 3D, on rejette
                cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                cv2.putText(debug_image, "Low pts", (x_min, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue

            points_3d = np.array(points_3d)

            # --- 3. RANSAC : DÉTECTION DE SPHÈRE ---
            best_inliers = 0
            best_center = None
            
            for iteration in range(self.ransac_iterations):
                # Échantillonner 4 points aléatoires
                if len(points_3d) < 4:
                    break
                    
                sample_idx = np.random.choice(len(points_3d), 4, replace=False)
                sample_points = points_3d[sample_idx]
                
                # Hypothèse: le centre de la sphère = moyenne des 4 points
                center_candidate = np.mean(sample_points, axis=0)
                
                # Calculer la distance de TOUS les points au centre
                distances = np.linalg.norm(points_3d - center_candidate, axis=1)
                
                # Compter les "inliers" = points à environ ball_radius du centre
                inliers_mask = np.abs(distances - self.ball_radius) < self.radius_tolerance
                num_inliers = np.sum(inliers_mask)
                
                # Garder le meilleur modèle
                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    # Raffiner le centre avec TOUS les inliers (pas juste les 4)
                    inlier_points = points_3d[inliers_mask]
                    best_center = np.mean(inlier_points, axis=0)
            
            # --- 4. VALIDATION ---
            if best_inliers >= self.min_inliers and best_center is not None:
                x_m, y_m, z_m = best_center
                
                # Vérifications de cohérence
                if z_m > 0.1 and z_m < 2.0:  # Entre 10cm et 2m
                    # Score combiné: inliers * confidence
                    current_score = best_inliers * confidence
                    
                    if current_score > best_score:
                        best_score = current_score
                        final_ball_3d = (x_m, y_m, z_m)
                        
                        # Dessin VERT (Validé)
                        cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        info_text = f"{z_m:.2f}m | {best_inliers}pts | {confidence:.2f}"
                        cv2.putText(debug_image, info_text, (x_min, y_min-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Rejeté: pas assez d'inliers ou pas de sphère trouvée
                cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 165, 255), 1)
                cv2.putText(debug_image, f"Reject ({best_inliers})", (x_min, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # --- 5. PUBLICATION ---
        if final_ball_3d:
            gx, gy, gz = final_ball_3d
            self.publish_result(rgb_msg.header, gx, gy, gz)
            self.get_logger().info(f"🎯 Balle 3D RANSAC: X={gx:.3f} Y={gy:.3f} Z={gz:.3f}m")
        else:
            # Publier un marker invisible pour signaler "pas de balle"
            self.publish_no_detection(rgb_msg.header)

        # Debug Image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header = rgb_msg.header
            self.pub_debug_img.publish(debug_msg)
        except Exception:
            pass

    def publish_result(self, header, x, y, z):
        # Point 3D
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
        # Diamètre = 2 * rayon
        marker.scale.x = self.ball_radius * 2
        marker.scale.y = self.ball_radius * 2
        marker.scale.z = self.ball_radius * 2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.pub_marker.publish(marker)

    def publish_no_detection(self, header):
        """Publie un marker DELETE pour indiquer qu'aucune balle n'est détectée"""
        marker = Marker()
        marker.header = header
        marker.ns = "ball_3d"
        marker.id = 0
        marker.action = Marker.DELETE
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
