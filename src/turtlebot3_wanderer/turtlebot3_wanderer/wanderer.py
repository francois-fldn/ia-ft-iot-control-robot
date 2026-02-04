import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8
from datetime import datetime
import logging
import math
from rclpy.qos import qos_profile_sensor_data

STATE_STOP = 0
STATE_ROTATE = 1
STATE_WANDERER = 2
STATE_LOCK_IN = 3
STATE_GO = 4

K_ROT = 0.5   # Gain pour la rotation (plus grand = correction plus agressive)
K_LIN = 0.3   # Gain pour l'avancée
MAX_ROT_SPEED = 1.0  # Vitesse max rad/s
MAX_LIN_SPEED = 0.22 # Vitesse max m/s (Turtlebot limit)
DIST_STOP = 0.20     # S'arrêter à 20cm de la balle

class SearchBallBehavior(Node):
    def __init__(self):
        super().__init__('search_ball_behavior')
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.point_subscriber = self.create_subscription(PointStamped, '/coordinator/point', self.ball_callback, 10)
        self.state_publisher = self.create_subscription(UInt8, '/coordinator/state', self.state_callback, 10)

        self.state = STATE_STOP
        self.obstacle_ahead = False

        self.log_state = -1

        self.ball_position = None
        self.aligned_with_ball = False

        self.rotating = False
        self.omega = None

        self.near_ball = False
        self.deplacing = False
        self.vitesse = None

        self.print_log = False

        self.logger = logging.getLogger("EXPLORATOR")
        logging.basicConfig(
            level = logging.INFO,
            format = "[EXPLORATOR] [%(levelname)s] %(message)s"
            )

        self.timer = self.create_timer(0.1, self.control_loop)
    
    # def change_state(self, new_state):
    #     self.prev_state = self.state
    #     self.state = new_state

    def state_to_str(self,state_int):
        states = ['STATE_STOP', 'STATE_ROTATE', 'STATE_WANDERER', 'STATE_LOCK_IN', 'STATE_GO']
        return states[state_int]

    def state_callback(self,msg):
        prev_state = self.state
        self.state = msg.data
        if (prev_state != self.state): self.logger.info(f'Reçu état {self.state_to_str(self.state)}')
        self.print_log = True


    def ball_callback(self,msg):
        self.ball_position = (msg.point.x, msg.point.y, msg.point.z)
        # print(self.ball_position)

    def scan_callback(self,msg):
        # checker entre -30deg et +30deg
        front_ranges = msg.ranges[0:30] + msg.ranges[330:359]
        # avoir la distance minimale en retirant les erreurs
        min_dist = min([r for r in front_ranges if r > 0.1])

        self.obstacle_ahead = min_dist <= 0.3
    
    def control_loop(self):
        msg = Twist()
        
        prev_log = self.log_state

        if (self.obstacle_ahead):
            self.log_state = 2
            msg.linear.x = 0.0
            msg.angular.z = 0.5
            if (self.state == STATE_LOCK_IN):
                msg.angular.z = 0.0

        elif (self.state == STATE_STOP):
            self.log_state = 0
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        elif (self.state == STATE_ROTATE):
            self.log_state = 1
            msg.linear.x = 0.0
            msg.angular.z = 0.5
        
        elif (self.state == STATE_WANDERER):
            self.log_state = 3
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        
        elif (self.state == STATE_LOCK_IN):
            self.log_state = 4

            if self.ball_position is None:
                msg.linear.x = 0.0
                msg.angular.z = 0.0

            else:
                x_ball = self.ball_position[0]
                y_ball = self.ball_position[1]

                error_angle = math.atan2(y_ball, x_ball)
                error_dist = math.sqrt(x_ball**2 + y_ball**2)

                if error_dist <= DIST_STOP:
                    self.near_ball = True
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
                    self.get_logger().info("Arrivé à la balle (STOP) !")
                
                else:
                    self.near_ball = False 
                    
                    cmd_rot = K_ROT * error_angle
                    msg.angular.z = max(min(cmd_rot, MAX_ROT_SPEED), -MAX_ROT_SPEED)

                    cmd_lin = K_LIN * error_dist
                    
                    correction_seuil = 0.5 
                    speed_factor = max(0.0, 1.0 - (abs(error_angle) / correction_seuil))

                    final_lin = cmd_lin * speed_factor
                    
                    msg.linear.x = max(min(final_lin, MAX_LIN_SPEED), 0.0)


        elif (self.state == STATE_GO):
            self.log_state = 6
            msg.linear.x = 0.3
        
        if prev_log != self.log_state : 
            self.logger.info(f'Applique la vitesse x = {msg.linear.x}, z = {msg.angular.z}')
        
        self.cmd_publisher.publish(msg)

def main():
    rclpy.init()
    node = SearchBallBehavior()
    rclpy.spin(node)
    rclpy.shutdown()
