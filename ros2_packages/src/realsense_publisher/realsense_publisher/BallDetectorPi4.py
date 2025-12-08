import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import message_filters # Pour la synchro RGB/Depth
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class BallTrackerNode(Node):
    def __init__(self):
        super().__init__('ball_tracker_node')

        # --- CONFIGURATION UTILISATEUR ---
        self.model_path = 'best_int8_edgetpu.tflite' # Votre modèle compilé
        self.conf_threshold = 0.5                     # Seuil de confiance
        
        # Dimensions
        self.cam_w = 640  # Résolution Caméra
        self.cam_h = 480
        self.model_w = 320 # Résolution Modèle
        self.model_h = 320
        
        # Facteurs d'échelle (Pré-calculés pour économiser le CPU)
        # Scale X = 640 / 320 = 2.0
        # Scale Y = 480 / 320 = 1.5
        self.scale_x = self.cam_w / self.model_w
        self.scale_y = self.cam_h / self.model_h

        # --- 1. INITIALISATION EDGE TPU ---
        self.get_logger().info("Chargement du TPU...")
        try:
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            self.get_logger().error(f"Erreur TPU: {e}. Vérifiez libedgetpu.")
            return

        # --- 2. INITIALISATION ROS ---
        self.bridge = CvBridge()
        self.pub_ball = self.create_publisher(PointStamped, '/ball_3d', 10)

        # Subscribers synchronisés
        # On utilise ApproximateTimeSynchronizer car les timestamps USB varient légèrement
        self.sub_rgb = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info], 
            queue_size=10, 
            slop=0.1 # Tolérance de 100ms
        )
        self.ts.registerCallback(self.listener_callback)
        
        self.get_logger().info("Node prêt. En attente d'images...")

    def listener_callback(self, rgb_msg, depth_msg, info_msg):
        # --- A. PRÉPARATION DES DONNÉES ---
        try:
            # Conversion ROS -> OpenCV
            # Depth en 'passthrough' garde le format uint16 (mm) d'origine
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().warn(f"Erreur conversion image: {e}")
            return

        # Récupération des intrinsèques (Camera Info)
        # On le fait à chaque frame ou on pourrait le stocker une fois dans __init__
        fx = info_msg.k[0] # Focale X
        fy = info_msg.k[4] # Focale Y
        cx = info_msg.k[2] # Centre optique X
        cy = info_msg.k[5] # Centre optique Y

        # --- B. PRÉ-TRAITEMENT (Resize 640x480 -> 320x320) ---
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        img_rgb_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb_input, axis=0) # Ajout batch dimension

        # --- C. INFÉRENCE CORAL ---
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # --- D. PARSING & REMAPPING ---
        # Note: Adaptez les indices [:, 4] selon votre export YOLOv5 (parfois c'est [:, 5] pour le score)
        # Ici on suppose une sortie raw [batch, anchors, 5+classes]
        
        # Calcul du score : Confiance Objet * Probabilité Classe
        # Si vous n'avez qu'une classe (balle), objectness suffit souvent
        scores = output_data[0, :, 4] 
        best_idx = np.argmax(scores)
        confidence = scores[best_idx]

        if confidence > self.conf_threshold:
            # Récupération box (Format Modèle 320x320)
            box = output_data[0, best_idx, :4] # x, y, w, h
            model_x, model_y = box[0], box[1]

            # Si le modèle sort des valeurs normalisées (0-1), décommentez :
            # model_x *= self.model_w
            # model_y *= self.model_h

            # Transformation vers l'image Caméra (640x480)
            # C'est ici qu'on gère la distorsion
            cam_x = int(model_x * self.scale_x)
            cam_y = int(model_y * self.scale_y)

            # Vérification limites image
            cam_x = np.clip(cam_x, 0, self.cam_w - 1)
            cam_y = np.clip(cam_y, 0, self.cam_h - 1)

            # --- E. LECTURE PROFONDEUR & CALCUL 3D ---
            # On prend une petite fenêtre 3x3 autour du centre pour éviter les pixels vides (0)
            d_region = cv_depth[
                max(0, cam_y-1):min(self.cam_h, cam_y+2),
                max(0, cam_x-1):min(self.cam_w, cam_x+2)
            ]
            
            # Filtrer les zéros (bruit)
            valid_depths = d_region[d_region > 0]

            if len(valid_depths) > 0:
                # Médiane pour robustesse
                z_mm = np.median(valid_depths)
                z_m = z_mm / 1000.0 # Conversion mm -> mètres

                # Formule Pinhole Inverse
                x_m = (cam_x - cx) * z_m / fx
                y_m = (cam_y - cy) * z_m / fy

                # --- F. PUBLICATION ROS ---
                msg = PointStamped()
                msg.header = rgb_msg.header # Important: garder le timestamp et frame_id originaux
                msg.point.x = x_m
                msg.point.y = y_m
                msg.point.z = z_m

                self.pub_ball.publish(msg)
                # self.get_logger().info(f"Balle: X={x_m:.2f} Y={y_m:.2f} Z={z_m:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()