import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
import random

class SearchBallBehavior(Node):
    def __init__(self):
        super().__init__('search_ball_behavior')
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.obstacle_ahead = False

        self.timer = self.create_timer(0.1, self.control_loop)
    
    def scan_callback(self,msg):
        # checker entre -30deg et +30deg
        front_ranges = msg.ranges[0:30] + msg.ranges[330:359]
        # avoir la distance minimale en retirant les erreurs
        min_dist = min([r for r in front_ranges if r > 0.1])

        self.obstacle_ahead = min_dist < 0.4
    
    def control_loop(self):
        msg = Twist()

        if self.obstacle_ahead:
            msg.linear.x = 0.0
            msg.angular.z = -0.5
        else:
            msg.linear.x = 0.2
            msg.angular.z = random.uniform(-0.5, 0.5)
        
        self.cmd_publisher.publish(msg)

def main():
    rclpy.init()
    node = SearchBallBehavior()
    rclpy.spin(node)
    rclpy.shutdown()