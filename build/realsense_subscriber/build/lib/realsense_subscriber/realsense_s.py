import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import pyrealsense2 as rs
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
# pas sur qu'il y ait tout qui serve

class Listener(Node):
    def __init__(self):
        super().__init__('pyRealsenseSub')
        self.subscriptionImage = self.create_subscription(Image, 'RealsenseImage', self.imageCallback, 10)
        self.subscriptionPointCloud = self.create_subscription(PointCloud2, 'RealsensePointCloud', self.pointCloudCallback, 10)
    
    def imageCallback(self, msg):
        print("TODO: transfo le message en Image traitable")
    
    def pointCloudCallback(self, msg):
        print("TODO: transfo le nuage de points en ")

def main(args=None):
    rclpy.init(args=args)
    node = Listener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()