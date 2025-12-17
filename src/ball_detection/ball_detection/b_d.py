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
        
        # Modèle (YOLOv5 Nano)
        model_filename = 'yolov5n-int8_edgetpu320.tflite' 
        self.model_path = os.path.join(package_share_directory, 'models', model_filename)

        # SEUIL : Ajusté pour éviter les faux positifs
        self.conf_threshold = 0.35 
        
        # Dimensions d'entrée du modèle
        self.model_w = 320 
        self.model_h = 320

        # --- 2. INITIALISATION TFLITE ---
        self.get_logger().info(f"Chargement du modèle : {model_filename}")
        try:
            # Tente de charger le délégué TPU (Coral USB)
            delegate = tflite.load_delegate('libedgetpu.so.1')
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[delegate]
            )
            self.get_logger().info("✅ Mode CORAL activé.")
        except Exception:
            self.get_logger().warn("⚠️ Coral introuvable, passage en mode CPU.")
            self.interpreter = tflite.Interpreter(model_path=self.model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # --- 3. INITIALISATION ROS ---
        self.bridge = CvBridge()
        
        self.pub_ball = self.create_publisher(PointStamped, '/ball_3d', 10)
        self.pub_marker = self.create_publisher(Marker, '/ball_marker', 10)
        self.pub_debug_img = self.create_publisher(Image, '/ball_debug', 10)

        # Synchronization des topics (RGB + Depth + Info)
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
            # Conversion ROS -> OpenCV
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Erreur conversion image: {e}")
            return

        cam_h, cam_w = cv_rgb.shape[:2]
        debug_image = cv_rgb.copy()

        # --- PRÉPARATION INFÉRENCE ---
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        # --- INFÉRENCE ---
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        detections = output_data[0]
        
        # Filtrage grossier pour l'analyse
        candidates = detections[detections[:, 4] > 0.10] 

        best_conf = 0
        final_cx, final_cy = 0, 0
        final_w, final_h = 0, 0
        detected = False

        # --- ANALYSE DES DÉTECTIONS ---
        for det in candidates:
            score = det[4]
            mx, my, mw, mh = det[0:4]
            
            # Remise à l'échelle (Model -> Camera)
            scale_x = cam_w / self.model_w
            scale_y = cam_h / self.model_h
            cx = int(mx * scale_x)
            cy = int(my * scale_y)
            w = int(mw * scale_x)
            h = int(mh * scale_y)

            # Dessin Debug
            top_left = (int(cx - w/2), int(cy - h/2))
            bottom_right = (int(cx + w/2), int(cy + h/2))

            if score > self.conf_threshold:
                # C'est une bonne détection
                color = (0, 255, 0) # Vert
                label = f"{score:.2f}"
                
                # On garde la meilleure balle (la plus confiante)
                if score > best_conf:
                    best_conf = score
                    detected = True
                    final_cx, final_cy = cx, cy
                    final_w, final_h = w, h
            else:
                color = (0, 0, 255) # Rouge (rejeté)
                label = f"X {score:.2f}"

            cv2.rectangle(debug_image, top_left, bottom_right, color, 2)
            cv2.putText(debug_image, label, (top_left[0], top_left[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- CALCUL 3D INTELLIGENT ---
        if detected:
            # 1. Définir une zone d'intérêt (ROI) plus petite que la boite
            # On prend 50% de la largeur/hauteur centrale pour éviter les bords (le mur)
            roi_ratio = 0.50 
            roi_w_half = int((final_w * roi_ratio) / 2)
            roi_h_half = int((final_h * roi_ratio) / 2)

            # Coordonnées du crop (sécurisées pour ne pas sortir de l'image)
            x_min = max(0, final_cx - roi_w_half)
            x_max = min(cam_w, final_cx + roi_w_half)
            y_min = max(0, final_cy - roi_h_half)
            y_max = min(cam_h, final_cy + roi_h_half)
            
            # On découpe la profondeur dans cette zone
            depth_crop = cv_depth[y_min:y_max, x_min:x_max]
            
            # --- CŒUR DU PROBLÈME : FILTRAGE ROBUSTE ---
            # On ne garde que les pixels > 0 (valides) et < 3.5m (on ignore le mur lointain)
            valid_depths = depth_crop[(depth_crop > 0) & (depth_crop < 3500)]

            if len(valid_depths) > 10:
                # TRI : La balle est l'objet le plus proche dans la boite.
                # On trie les valeurs : [plus_proche, ..., plus_loin]
                valid_depths = np.sort(valid_depths)
                
                n_pixels = len(valid_depths)
                
                # On ignore les 5% les plus proches (bruit "poivre" ou erreur capteur)
                # On s'arrête à 20% (pour être sûr de taper la balle et pas le bord qui fuit vers le mur)
                idx_start = int(n_pixels * 0.05)
                idx_end = int(n_pixels * 0.20)
                
                # Sécurité si peu de pixels
                if idx_end <= idx_start:
                    idx_end = idx_start + 1

                # Moyenne sur les pixels les plus proches (le "devant" de la balle)
                z_mm = np.mean(valid_depths[idx_start:idx_end])
                z_m = z_mm * 0.001
                
                # Projection 2D -> 3D (Modèle Pinhole)
                fx = info_msg.k[0]
                fy = info_msg.k[4]
                ppx = info_msg.k[2]
                ppy = info_msg.k[5]
                
                # On utilise le centre de la boite (final_cx) pour le X/Y
                x_m = (final_cx - ppx) * z_m / fx
                y_m = (final_cy - ppy) * z_m / fy

                # Publication et Log
                self.publish_result(rgb_msg.header, x_m, y_m, z_m)
                
                # Affichage sur l'image de debug
                label_dist = f"{z_m:.2f}m"
                cv2.putText(debug_image, label_dist, (final_cx, final_cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                self.get_logger().info(f"⚽ Balle OK | Conf: {best_conf:.2f} | Z: {z_m:.3f}m | (Pixels utilisés: {idx_end-idx_start})")
            #else:
                #self.get_logger().warn("Balle vue mais pas de profondeur valide (trop près ou reflets ?)")

        # Publication Image Debug
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
        
        # Marker pour RViz
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
        marker.color.r = 1.0 # Jaune
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.pub_marker.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()