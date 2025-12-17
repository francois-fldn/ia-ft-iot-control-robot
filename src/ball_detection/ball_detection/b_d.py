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

class BallDetectorHeadless(Node):
    def __init__(self):
        super().__init__('ball_detector_headless')

        # --- CONFIGURATION ---
        package_share_directory = get_package_share_directory('ball_detection')
        model_filename = 'yolov5n-int8_edgetpu320.tflite' 
        self.model_path = os.path.join(package_share_directory, 'models', model_filename)

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
        
        # Synchro large (0.5s) pour accepter les lags réseau/wifi
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info("✅ Mode TEXTE (Headless) prêt. Regardez les logs ci-dessous...")
        self.get_logger().info("-------------------------------------------------------------")

    def callback(self, rgb_msg, depth_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception:
            return

        cam_h, cam_w = cv_rgb.shape[:2]

        # Inférence IA
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        detections = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        candidates = detections[detections[:, 4] > self.conf_threshold]

        if len(candidates) == 0:
            # Petit point pour dire que ça tourne, sans spammer
            # print(".", end="", flush=True) 
            pass

        for det in candidates:
            score = det[4]
            mx, my, mw, mh = det[0:4]
            
            # Conversion coordonnées
            scale_x = cam_w / self.model_w
            scale_y = cam_h / self.model_h
            cx, cy = int(mx * scale_x), int(my * scale_y)
            w, h = int(mw * scale_x), int(mh * scale_y)

            # Découpage de la zone de profondeur (ROI)
            x_min = max(0, int(cx - w/2))
            x_max = min(cam_w, int(cx + w/2))
            y_min = max(0, int(cy - h/2))
            y_max = min(cam_h, int(cy + h/2))

            depth_roi = cv_depth[y_min:y_max, x_min:x_max]
            
            # --- ANALYSE STATISTIQUE (C'est ici que ça se joue) ---
            total_pixels = depth_roi.size
            if total_pixels == 0: continue

            # On compte les pixels valides (> 0) et pertinents (< 2m)
            valid_pixels = depth_roi[(depth_roi > 0) & (depth_roi < 2000)]
            count_valid = len(valid_pixels)
            
            # Ratio de remplissage (Coverage)
            coverage_percent = (count_valid / total_pixels) * 100
            
            dist_msg = ""
            status_icon = ""

            if count_valid > 0:
                median_dist_mm = np.median(valid_pixels)
                median_dist_m = median_dist_mm / 1000.0
                dist_msg = f"{median_dist_m:.2f}m"
            else:
                dist_msg = "N/A"

            # --- DIAGNOSTIC AUTOMATIQUE ---
            if coverage_percent < 10.0:
                status_icon = "🔴 FANTÔME"
                advice = "MATIÈRE INVISIBLE AUX IR (Trop noir/brillant)"
            elif coverage_percent < 40.0:
                status_icon = "🟠 FRAGILE"
                advice = "Détection partielle (Reflets ?)"
            else:
                status_icon = "🟢 SOLIDE"
                advice = "Balle bien vue"

            # Affichage console formaté
            log_msg = (
                f"\n--- BALL DETECTED ({score:.2f}) ---\n"
                f"   📍 Distance  : {dist_msg}\n"
                f"   📊 Couverture: {coverage_percent:.1f}% des pixels valides\n"
                f"   🩺 Diagnostic: {status_icon} -> {advice}\n"
                f"-----------------------------------"
            )
            self.get_logger().info(log_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorHeadless()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()