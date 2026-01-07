import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8
import random
from datetime import datetime

STATE_STOP = 0
STATE_ROTATE = 1
STATE_WANDERER = 2
STATE_LOCK_IN = 3

class SearchBallBehavior(Node):
    def __init__(self):
        super().__init__('search_ball_behavior')
        self.state_publisher = self.create_subscription(UInt8, '/turtlebot3_state', self.state_callback, 10)
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = STATE_STOP
        self.obstacle_ahead = False

        self.log_state = -1

        self.timer = self.create_timer(0.1, self.control_loop)
    
    def state_to_str(self,state_int):
        states = ['STATE_STOP', 'STATE_ROTATE', 'STATE_WANDERER', 'STATE_LOCK_IN']
        return states[state_int]

    def state_callback(self,msg):
        self.state = msg.data
        print(f'[INFO] {datetime.now()} Reçu état {self.state_to_str(self.state)}')

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
        
        if prev_log != self.log_state : print(f'[INFO] Applique la vitesse x = {msg.linear.x}, z = {msg.angular.z}')
        self.cmd_publisher.publish(msg)

def main():
    rclpy.init()
    node = SearchBallBehavior()
    rclpy.spin(node)
    rclpy.shutdown()