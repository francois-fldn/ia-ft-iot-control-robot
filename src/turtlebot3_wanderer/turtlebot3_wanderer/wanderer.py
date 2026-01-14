import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8
import random
from datetime import datetime
import logging
import math

STATE_STOP = 0
STATE_ROTATE = 1
STATE_WANDERER = 2
STATE_LOCK_IN = 3

class SearchBallBehavior(Node):
    def __init__(self):
        super().__init__('search_ball_behavior')
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.point_subscriber = self.create_subscription(PointStamped, '/coordinator/point', self.ball_callback, 10)
        self.state_publisher = self.create_subscription(UInt8, '/coordinator/state', self.state_callback, 10)

        self.state = STATE_STOP
        self.obstacle_ahead = False

        self.log_state = -1

        self.ball_position = None
        self.aligned_with_ball = False
        # self.initiated_rotation = False
        self.rotating = False
        self.omega = 0.0

        self.near_ball = False
        self.deplacing = False
        self.vitesse = 0.0

        self.logger = logging.getLogger("EXPLORATOR")
        logging.basicConfig(
            level = logging.INFO,
            format = "[EXPLORATOR] [%(levelname)s] %(message)s"
            )

        self.timer = self.create_timer(0.1, self.control_loop)
    
    def state_to_str(self,state_int):
        states = ['STATE_STOP', 'STATE_ROTATE', 'STATE_WANDERER', 'STATE_LOCK_IN']
        return states[state_int]

    def state_callback(self,msg):
        prev_state = self.state
        self.state = msg.data
        if (prev_state != self.state): self.logger.info(f'Reçu état {self.state_to_str(self.state)}')

    def ball_callback(self,msg):
        self.ball_position = (msg.point.x, msg.point.y, msg.point.z)
        # print(self.ball_position)

    def scan_callback(self,msg):
        # checker entre -30deg et +30deg
        front_ranges = msg.ranges[0:30] + msg.ranges[330:359]
        # avoir la distance minimale en retirant les erreurs
        min_dist = min([r for r in front_ranges if r > 0.1])

        self.obstacle_ahead = min_dist < 0.4
    
    def control_loop(self):
        msg = Twist()
        
        prev_log = self.log_state

        if (self.state == STATE_STOP):
            self.log_state = 0
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        elif (self.state == STATE_ROTATE):
            self.log_state = 1
            msg.linear.x = 0.0
            msg.angular.z = 0.5
        
        elif (self.state == STATE_WANDERER):
            if self.obstacle_ahead:
                self.log_state = 2
                msg.linear.x = 0.0
                msg.angular.z = -0.5
            else:
                self.log_state = 3
                msg.linear.x = 0.2
                msg.angular.z = random.uniform(-0.5, 0.5)
        
        elif (self.state == STATE_LOCK_IN) :
            # premiere etape, s'aligner avec la balle

            if (not(self.aligned_with_ball)):
                if (self.rotating): msg.angular.z = self.omega
                else: 
                    self.log_state = 4
                    self.omega = (math.atan2(self.ball_position[1], self.ball_position[0]))/2
                    msg.angular.z = self.omega
                    self.rotating = True
                    self.start_timer = datetime.now()
                delta = datetime.now() - self.start_timer
                if (delta.seconds >= 2):
                    self.aligned_with_ball = True
                    self.rotating = False
                    self.near_ball = False 

            if (self.aligned_with_ball and not(self.near_ball)):
                msg.angular.z = 0.0
                if (self.deplacing): msg.linear.x = self.vitesse
                else:
                    # logger.info(f"Deplacing vers {round(self.ball_position[0],2), round(self.ball_position[1],2)}")
                    self.start_timer = datetime.now()
                    self.log_state = 5
                    distance = math.sqrt(self.ball_position[0]**2 + self.ball_position[1]**2)
                    self.vitesse = distance / 10
                    msg.linear.x = self.vitesse
                    self.deplacing = True
                delta = datetime.now() - self.start_timer
                if (delta.seconds >= 11):
                    self.near_ball = True
                    self.deplacing = False
                    self.aligned_with_ball = False

        if prev_log != self.log_state : self.logger.info(f'Applique la vitesse x = {msg.linear.x}, z = {msg.angular.z}')
        self.cmd_publisher.publish(msg)

def main():
    rclpy.init()
    node = SearchBallBehavior()
    rclpy.spin(node)
    rclpy.shutdown()