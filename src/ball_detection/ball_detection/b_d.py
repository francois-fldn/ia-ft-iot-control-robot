import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo  # <-- Ajout de CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from ament_index_python.packages import get_package_share_directory
import os
import math

class BallDetectorHybrid(Node):
    def __init__(self):
        super().__init__('ball_detector_hybrid')

        # --- IA SETUP ---
        try:
            package_share_directory = get_package_share_directory('ball_detection')
            model_filename = 'yolov5n-int8_320.tflite' 
            self.model_path = os.path.join(package_share_directory, 'models', model_filename)
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.get_logger().info(" IA Chargée (Mode CameraInfo).")
        except Exception as e:
            self.get_logger().error(f" Erreur Init IA: {e}")
            raise e

        self.conf_threshold = 0.80
        self.model_w = 320; self.model_h = 320
        self.latest_depth_img = None 
        
        # Stockage des infos caméra (fx, fy, cx, cy)
        self.camera_intrinsics = None 

        # --- ROS SETUP ---
        self.bridge = CvBridge()
        self.pub_ball = self.create_publisher(PointStamped, 'ball_3d', 10)
        self.pub_marker = self.create_publisher(Marker, 'ball_marker', 10)
        self.pub_debug_img = self.create_publisher(Image, 'ball_debug', 10)

        # Subscribers
        self.sub_rgb = self.create_subscription(Image, 'rgb_camera/image', self.callback_rgb, 10)
        self.sub_depth = self.create_subscription(Image, 'depth_camera/image', self.callback_depth, 10)
        
        # NOUVEAU : On écoute les caractéristiques de la caméra
        self.sub_info = self.create_subscription(CameraInfo, 'rgb_camera/camera_info', self.callback_info, 10)
        
        self.get_logger().info(" Prêt : Utilisation dynamique de CameraInfo.")

    def callback_info(self, msg):
        # On ne stocke ça qu'une fois (ou à chaque update), c'est très léger
        # La matrice K contient : [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        if self.camera_intrinsics is None:
            self.get_logger().info(f" CameraInfo reçu : {msg.width}x{msg.height}")
        self.camera_intrinsics = msg

    def callback_depth(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError: pass

    def callback_rgb(self, rgb_msg):
        try: cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except: return

        debug_image = cv_rgb.copy()
        cam_h, cam_w = cv_rgb.shape[:2]
        
        # Flag pour savoir si une balle a été détectée
        ball_found = False

        # --- IA Inference ---
        img_resized = cv2.resize(cv_rgb, (self.model_w, self.model_h))
        input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)
        
        input_d = self.input_details[0]
        if input_d['dtype'] == np.int8 or input_d['dtype'] == np.uint8:
            scale, zero = input_d['quantization']
            input_data = (input_data / 255.0 / scale + zero).astype(input_d['dtype'])
        else:
            input_data = (input_data.astype(np.float32) / 255.0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_d = self.output_details[0]
        raw = self.interpreter.get_tensor(output_d['index'])[0]

        if output_d['dtype'] == np.int8 or output_d['dtype'] == np.uint8:
            scale, zero = output_d['quantization']
            detections = (raw.astype(np.float32) - zero) * scale
        else: detections = raw

        candidates = detections[detections[:, 4] > self.conf_threshold]

        for det in candidates:
            cx_norm, cy_norm, w_norm, h_norm = det[0], det[1], det[2], det[3]
            
            if w_norm > 1.0: 
                scale_x = cam_w/320.0; scale_y = cam_h/320.0
                cx = int(cx_norm*scale_x); cy = int(cy_norm*scale_y)
                w = int(w_norm*scale_x); h = int(h_norm*scale_y)
            else:
                cx = int(cx_norm*cam_w); cy = int(cy_norm*cam_h)
                w = int(w_norm*cam_w); h = int(h_norm*cam_h)

            x_tl = int(cx - w/2); y_tl = int(cy - h/2)
            cv2.rectangle(debug_image, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 255, 0), 2)

            if self.latest_depth_img is not None:
                d_h, d_w = self.latest_depth_img.shape[:2]
                u_depth = int(cx * (d_w / cam_w))
                v_depth = int(cy * (d_h / cam_h))

                if 0 <= u_depth < d_w and 0 <= v_depth < d_h:
                    dist_z = self.latest_depth_img[v_depth, u_depth]
                    
                    if 0.1 < dist_z < 10.0 and not math.isnan(dist_z):
                        
                        # --- UTILISATION DE CAMERA INFO ---
                        if self.camera_intrinsics is not None:
                            # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                            fx = self.camera_intrinsics.k[0]
                            cx_opt = self.camera_intrinsics.k[2] # Centre optique X
                            fy = self.camera_intrinsics.k[4]
                            cy_opt = self.camera_intrinsics.k[5] # Centre optique Y
                        
                        
                        # Formule Optique : X = (pixel - centre) * Z / focale
                        raw_x = (u_depth - cx_opt) * dist_z / fx 
                        raw_y = (v_depth - cy_opt) * dist_z / fy 
                        raw_z = dist_z                           

                        label = f"Z:{raw_z:.2f}m"
                        cv2.putText(debug_image, label, (x_tl, y_tl-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        self.publish_result(raw_x, raw_y, raw_z)
                        ball_found = True
                        break 
        
        # Si aucune balle n'a été détectée, publier des coordonnées aberrantes
        if not ball_found:
            self.publish_no_ball()

        try: self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
        except: pass

    def publish_result(self, cam_x, cam_y, cam_z):
        pt = PointStamped()
        pt.header.frame_id = "base_footprint" 
        pt.header.stamp = self.get_clock().now().to_msg()
        
        # --- INVERSION DES AXES (Manuelle) ---
        pt.point.x = float(cam_z)       # Profondeur -> Devant
        pt.point.y = -float(cam_x)      # Droite -> Gauche
        pt.point.z = float(cam_y)       # Bas -> Haut (Attention au signe ici)
        
        self.pub_ball.publish(pt)
        
        marker = Marker()
        marker.header = pt.header
        marker.ns = "ball_visual"; marker.id = 0; marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position = pt.point
        marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0
        self.pub_marker.publish(marker)
    
    def publish_no_ball(self):
        """Publie des coordonnées aberrantes quand aucune balle n'est détectée"""
        pt = PointStamped()
        pt.header.frame_id = "base_footprint"
        pt.header.stamp = self.get_clock().now().to_msg()
        
        # Coordonnées aberrantes pour indiquer l'absence de balle
        pt.point.x = 1000.0
        pt.point.y = 1000.0
        pt.point.z = 1000.0
        
        self.pub_ball.publish(pt)

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorHybrid()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
