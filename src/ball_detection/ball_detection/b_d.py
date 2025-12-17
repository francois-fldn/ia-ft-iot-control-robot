import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from ament_index_python.packages import get_package_share_directory
import os

class BallDetectorSmart(Node):
    def __init__(self):
        super().__init__('ball_detector_smart')

        # --- CONFIGURATION ---
        package_share_directory = get_package_share_directory('ball_detection')
        model_filename = 'yolov5n-int8_edgetpu320.tflite' 
        self.model_path = os.path.join(package_share_directory, 'models', model_filename)

        # On garde un seuil bas pour capter les vraies balles
        self.conf_threshold = 0.25 
        self.model_w = 320 
        self.model_h = 320

        # --- IA SETUP ---
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
        
        self.sub_rgb = message_filters.Subscriber(self, Image, '/Realsense/Image/Color')
        self.sub_depth = message_filters.Subscriber(self, Image, '/Realsense/Image/Depth')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info("✅ Mode FILTRÉ prêt. En attente...")

    def callback(self, rgb_msg, depth_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception:
            return

        cam_h, cam_w = cv_rgb.shape[:2]

        # Inférence
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        detections = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        candidates = detections[detections[:, 4] > self.conf_threshold]

        for det in candidates:
            score = det[4]
            mx, my, mw, mh = det[0:4]
            
            # Conversion
            scale_x = cam_w / self.model_w
            scale_y = cam_h / self.model_h
            cx, cy = int(mx * scale_x), int(my * scale_y)
            w, h = int(mw * scale_x), int(mh * scale_y)
            
            # --- FILTRES ANTI-FANTÔME (La partie importante) ---
            
            # 1. Filtre de Taille relative
            # Si la boite fait plus de 40% de l'image, c'est probablement le mur entier détecté par erreur
            area_img = cam_w * cam_h
            area_box = w * h
            ratio_area = area_box / area_img
            
            if ratio_area > 0.30:
                self.get_logger().warn(f"🚫 REJETÉ (Trop gros): {ratio_area*100:.1f}% de l'écran")
                continue # On passe au suivant

            # 2. Filtre de Forme (Ratio Aspect)
            # Une balle est carrée dans la boite (ratio ~1.0). Si ratio > 2.0 ou < 0.5, c'est une barre ou un truc plat.
            if w > 0 and h > 0:
                aspect_ratio = w / h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    self.get_logger().warn(f"🚫 REJETÉ (Forme bizarre): Ratio {aspect_ratio:.2f}")
                    continue

            # 3. Filtre de Bordure
            # Si le centre est collé au bord de l'image, c'est souvent un artefact
            if cx < 20 or cx > (cam_w - 20) or cy < 20 or cy > (cam_h - 20):
                 self.get_logger().warn(f"🚫 REJETÉ (Bordure)")
                 continue

            # --- Si on arrive ici, c'est probablement une vraie balle ---
            
            # Analyse Profondeur
            x_min, x_max = max(0, int(cx - w/2)), min(cam_w, int(cx + w/2))
            y_min, y_max = max(0, int(cy - h/2)), min(cam_h, int(cy + h/2))
            
            depth_roi = cv_depth[y_min:y_max, x_min:x_max]
            valid_pixels = depth_roi[(depth_roi > 0) & (depth_roi < 3000)] # Max 3m
            
            if len(valid_pixels) > 0:
                coverage = (len(valid_pixels) / depth_roi.size) * 100
                median_z = np.median(valid_pixels) / 1000.0
                
                # LOG VALIDÉ
                self.get_logger().info(
                    f"✅ BALLE VALIDÉE | Dist: {median_z:.2f}m | Conf: {score:.2f} | Taille: {w}x{h}"
                )
            else:
                 self.get_logger().info(f"⚠️ Balle vue (Conf {score:.2f}) mais profondeur vide/trop loin")

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorSmart()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()