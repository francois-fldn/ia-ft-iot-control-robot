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

class BallDetectorEdgeTPU(Node):
    def __init__(self):
        super().__init__('ball_detector_edgetpu')

        # --- 1. CONFIGURATION DU MODÈLE ---
        package_share_directory = get_package_share_directory('ball_detection')
        
        # IMPORTANT : Mettez ici le nom EXACT de votre modèle compilé pour le TPU
        # Il contient souvent "_edgetpu" dans le nom.
        model_filename = 'yolov5n-int8_edgetpu320.tflite' 
        self.model_path = os.path.join(package_share_directory, 'models', model_filename)

        self.conf_threshold = 0.40 # Seuil de confiance
        self.model_w = 320 
        self.model_h = 320

        # --- 2. INITIALISATION EDGE TPU ---
        self.get_logger().info(f"Chargement du modèle EdgeTPU : {model_filename}")
        try:
            # Chargement de la librairie partagée du Coral
            # Si vous êtes sur PC et que cela échoue, vérifiez LD_LIBRARY_PATH
            delegate = tflite.load_delegate('libedgetpu.so.1')
            
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[delegate]
            )
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.get_logger().info("🚀 SUCCÈS : Accélérateur Coral EdgeTPU actif !")
            
        except Exception as e:
            self.get_logger().error(f"❌ ÉCHEC CRITIQUE TPU : {e}")
            self.get_logger().error("Vérifiez que la Coral est branchée et que libedgetpu1-std est installé.")
            return # On arrête le nœud si le TPU ne marche pas

        # --- 3. INITIALISATION ROS ---
        self.bridge = CvBridge()
        
        self.pub_ball = self.create_publisher(PointStamped, '/ball_3d', 10)
        self.pub_marker = self.create_publisher(Marker, '/ball_marker', 10)

        # Subscribers Synchronisés (RGB + Depth + Info)
        self.sub_rgb = message_filters.Subscriber(self, Image, '/Realsense/Image/Color')
        self.sub_depth = message_filters.Subscriber(self, Image, '/Realsense/Image/Depth')
        self.sub_info = message_filters.Subscriber(self, CameraInfo, '/Realsense/CameraInfo')

        # Synchronisation
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.callback)
        
        self.get_logger().info("Node prêt. En attente de flux...")

    def callback(self, rgb_msg, depth_msg, info_msg):
        # A. Conversion Images
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().warn(f"Erreur cv_bridge: {e}")
            return

        cam_h, cam_w = cv_rgb.shape[:2]

        # B. Préparation Inférence
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        # C. Inférence RAPIDE (TPU)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke() # C'est ici que le Coral travaille
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # D. Traitement des résultats
        # YOLOv5 sort [1, N, 85] (xywh + conf + classes)
        detections = output_data[0]
        
        # Filtre de confiance
        strong_detections = detections[detections[:, 4] > self.conf_threshold]

        if len(strong_detections) > 0:
            best_idx = np.argmax(strong_detections[:, 4])
            detection = strong_detections[best_idx]
            
            # Coordonnées Modèle (0-320)
            mx, my, mw, mh = detection[0:4]
            
            # Mise à l'échelle vers Caméra
            scale_x = cam_w / self.model_w
            scale_y = cam_h / self.model_h
            
            cx_img = int(mx * scale_x)
            cy_img = int(my * scale_y)
            
            cx_img = np.clip(cx_img, 0, cam_w - 1)
            cy_img = np.clip(cy_img, 0, cam_h - 1)

            # E. Calcul 3D (Median Filter)
            # Fenêtre de 10x10 pixels
            window = 5
            x_min = max(0, cx_img - window)
            x_max = min(cam_w, cx_img + window)
            y_min = max(0, cy_img - window)
            y_max = min(cam_h, cy_img + window)

            depth_crop = cv_depth[y_min:y_max, x_min:x_max]
            valid_depths = depth_crop[depth_crop > 0]

            if len(valid_depths) > 0:
                z_mm = np.median(valid_depths)
                z_m = z_mm * 0.001 

                # Projection 3D
                fx = info_msg.k[0]
                fy = info_msg.k[4]
                ppx = info_msg.k[2]
                ppy = info_msg.k[5]

                x_m = (cx_img - ppx) * z_m / fx
                y_m = (cy_img - ppy) * z_m / fy

                # F. Publication
                self.publish_result(rgb_msg.header, x_m, y_m, z_m)

    def publish_result(self, header, x, y, z):
        # Topic de coordonnées
        pt = PointStamped()
        pt.header = header
        pt.point.x = float(x)
        pt.point.y = float(y)
        pt.point.z = float(z)
        self.pub_ball.publish(pt)

        # Marker Rviz
        marker = Marker()
        marker.header = header
        marker.ns = "yolo_ball"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0 # Vert pour dire "TPU OK"
        marker.color.b = 0.0
        self.pub_marker.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorEdgeTPU()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()