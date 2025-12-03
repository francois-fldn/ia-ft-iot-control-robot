import rclpy # the ros2 python client library
from rclpy.node import Node # A base class from rclpy that represents a ROS2 node
from sensor_msgs.msg import PointCloud2, PointField
# from std_msgs.msg import PointCloud2
import pyrealsense2 as rs
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class PointCloudPublisher(Node):
	def __init__(self):
		# publisher config
		super().__init__('pointcloudpublisher')
		self.point_cloud_publisher_ = self.create_publisher(PointCloud2, 'RealsensePointCloud', 10)
		self.color_publisher_ = self.create_publisher(Image, 'RealsenseColor', 10)
		self.timer = self.create_timer(1/10, self.timer_callback)
		
		# realsense pipeline config
		self.pc = rs.pointcloud()
		self.points = rs.points()
		self.pipe = rs.pipeline()
		self.config = rs.config()
		self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
		self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

		self.pipe.start(self.config)

		self.decimate = rs.decimation_filter(8)
		self.align = rs.align(rs.stream.color)
		
		self.i = 0

	def timer_callback(self):
		
		frames = self.pipe.wait_for_frames()
	
		aligned_frames = self.align.process(frames)
		depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()
		
		if (not depth_frame) or (not color_frame): return
		
		points = self.pc.calculate(depth_frame)
		# decimated = decimate.process(aligned_frames).as_frameset()
		
		verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
		
		header = Header()
		header.frame_id = "camera_depth_optical_frame"
		pc_msg = pc2.create_cloud_xyz32(header, verts)
		
		color_image = np.asanyarray(color_frame.get_data())
		
		bridge = CvBridge()
		ros_image_msg = bridge.cv2_to_imgmsg(color_image, encoding="rgb8")
	
		self.point_cloud_publisher_.publish(pc_msg)	
		self.color_publisher_.publish(ros_image_msg)
		
		
		
		'''
		color_image = np.asanyarray(color_frame.get_data()).reshape(-1,3)
		
		r = color_image_flat[:,0].astype(np.uint32)
		g = color_image_flat[:,1].astype(np.uint32)
		b = color_image_flat[:,2].astype(np.uint32)
		rgb = np.left_shift(r, 16) | np.left_shift(g, 8) | b
		rgb.dtype = np.float32
		
		dtype_cloud = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)]
		cloud_data = np.zeros(verts.shape[0], dtype=dtype_cloud)
		cloud_data['x'] = verts[:, 0]
		cloud_data['y'] = verts[:, 1]
		cloud_data['z'] = verts[:, 2]
		cloud_data['rgb'] = rgb
		
		header = Header()
		header.frame_id = "frame_RS_D435i"  # Remplace par le nom de ta frame TF
		# header.stamp = ... (idéalement, utilise l'horloge ROS ou le timestamp realsense)

		
		point_cloud_msg = pc2.create_cloud(
			 header,
			 fields=[
				  PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
				  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
				  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
			 ],
			 points=cloud_data
		)
		
		self.publisher_.publish(point_cloud_msg)
		'''

		self.i+=1

def main(args=None):
	rclpy.init(args=args) # Initialize the ROS2 Python system
	node = PointCloudPublisher() # Create an instance of the Listener node
	rclpy.spin(node) # Keep the node running, listening for messages
	node.destroy_node() # Cleanup when the node is stopped
	rclpy.shutdown() # It cleans up all ROS2 resources used by the node

if __name__ == '__main__':
	main()
