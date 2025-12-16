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

class BallDetectorDebug(Node):
    def __init__(self):
        super().__init__('ball_detector_debug')

        # --- 1. CONFIGURATION ---
        package_share_directory = get_package_share_directory('ball_detection')
        
        # Choisissez votre modèle (CPU ou EdgeTPU selon ce qui marche chez vous)
        model_filename = 'yolov5n-int8_edgetpu320.tflite' 
        self.model_path = os.path.join(package_share_directory, 'models', model_filename)

        # SEUIL CRITIQUE : Si vous avez trop de faux positifs, AUGMENTEZ ceci (0.40 -> 0.60)
        self.conf_threshold = 0.40 
        
        self.model_w = 320 
        self.model_h = 320

        # --- 2. INITIALISATION TFLITE ---
        self.get_logger().info(f"Chargement du modèle : {model_filename}")
        try:
            # Tente de charger le délégué TPU
            delegate = tflite.load_delegate('libedgetpu.so.1')
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[delegate]
            )
            self.get_logger().info("✅ Mode CORAL activé.")
        except Exception:
            self.get_logger().warn("⚠️ Coral introuvable, passage en mode CPU (lent).")
            self.interpreter = tflite.Interpreter(model_path=self.model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # --- 3. INITIALISATION ROS ---
        self.bridge = CvBridge()
        
        self.pub_ball = self.create_publisher(PointStamped, '/ball_3d', 10)
        self.pub_marker = self.create_publisher(Marker, '/ball_marker', 10)
        
        # --- NOUVEAU : Image de debug pour voir les carrés ---
        self.pub_debug_img = self.create_publisher(Image, '/ball_debug', 10)

        # Subscribers
        self.sub_rgb = message_filters.Subscriber(self, Image, '/Realsense/Image/Color')
        self.sub_depth = message_filters.Subscriber(self, Image, '/Realsense/Image/Depth')
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/Realsense/CameraInfo')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info], 
            queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info("Node Debug prêt. Vérifiez le topic /ball_debug")

    def callback(self, rgb_msg, depth_msg, info_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            return

        cam_h, cam_w = cv_rgb.shape[:2]
        
        # Copie pour le dessin
        debug_image = cv_rgb.copy()

        # Préparation Inférence
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        # Inférence
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        detections = output_data[0]
        
        # On regarde tout ce qui dépasse 10% de confiance pour comprendre ce qui se passe
        candidates = detections[detections[:, 4] > 0.10] 

        best_conf = 0
        final_x, final_y, final_z = 0, 0, 0
        detected = False

        for det in candidates:
            score = det[4]
            # Coordonnées Modèle
            mx, my, mw, mh = det[0:4]
            
            # Conversion vers Image Caméra
            scale_x = cam_w / self.model_w
            scale_y = cam_h / self.model_h
            cx = int(mx * scale_x)
            cy = int(my * scale_y)
            w = int(mw * scale_x)
            h = int(mh * scale_y)

            # --- DESSIN DE DEBUG ---
            # Si c'est au dessus du seuil valide (VERT), sinon (ROUGE)
            if score > self.conf_threshold:
                color = (0, 255, 0) # Vert
                label = f"BALL: {score:.2f}"
                
                # C'est notre meilleure détection ?
                if score > best_conf:
                    best_conf = score
                    detected = True
                    final_cx, final_cy = cx, cy
            else:
                color = (0, 0, 255) # Rouge (rejeté mais détecté)
                label = f"Ignored: {score:.2f}"

            # Dessiner le rectangle sur l'image de debug
            top_left = (int(cx - w/2), int(cy - h/2))
            bottom_right = (int(cx + w/2), int(cy + h/2))
            cv2.rectangle(debug_image, top_left, bottom_right, color, 2)
            cv2.putText(debug_image, label, (top_left[0], top_left[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Si on a trouvé une balle valide
        if detected:
            # Calcul 3D (Median)
            window = 5
            x_min = max(0, final_cx - window)
            x_max = min(cam_w, final_cx + window)
            y_min = max(0, final_cy - window)
            y_max = min(cam_h, final_cy + window)
            
            depth_crop = cv_depth[y_min:y_max, x_min:x_max]
            valid_depths = depth_crop[depth_crop > 0]

            if len(valid_depths) > 0:
                z_mm = np.median(valid_depths)
                z_m = z_mm * 0.001
                
                fx = info_msg.k[0]
                fy = info_msg.k[4]
                ppx = info_msg.k[2]
                ppy = info_msg.k[5]
                
                x_m = (final_cx - ppx) * z_m / fx
                y_m = (final_cy - ppy) * z_m / fy

                # Publication
                self.publish_result(rgb_msg.header, x_m, y_m, z_m)
                
                # Log dans le terminal pour vous rassurer
                self.get_logger().info(f"⚽ Balle détectée ! Confiance: {best_conf:.2f} | Dist: {z_m:.2f}m")

        # TOUJOURS publier l'image de debug, même si rien n'est détecté
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
        
        # Marker (pour Rviz)
        marker = Marker()
        marker.header = header
        marker.ns = "ball"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.pub_marker.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorDebug()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()