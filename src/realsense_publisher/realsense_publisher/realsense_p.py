import rclpy # the ros2 python client library
from rclpy.node import Node # A base class from rclpy that represents a ROS2 node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import PointCloud2
import pyrealsense2 as rs
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo


class PointCloudPublisher(Node):
	def __init__(self):
		# publisher config
		super().__init__('pointcloudpublisher')
		self.depth_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Depth', 10)
		self.color_image_publisher_ = self.create_publisher(Image, 'Realsense/Image/Color', 10)
		# self.point_cloud_publisher_ = self.create_publisher(PointCloud2, 'Realsense/PointCloud', 10)
		self.camera_info_publisher_ = self.create_publisher(CameraInfo, 'Realsense/CameraInfo', 10)
		self.timer = self.create_timer(1/10, self.timer_callback)
		
		# realsense pipeline config
		self.pc = rs.pointcloud()
		self.points = rs.points()
		self.pipe = rs.pipeline()
		self.config = rs.config()
		self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
		self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

		profile = self.pipe.start(self.config)
		
		stream_profile = profile.get_stream(rs.stream.depth)
		self.intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()

		self.decimate = rs.decimation_filter(8)
		self.align = rs.align(rs.stream.color)

		depth_sensor = profile.get_device().first_depth_sensor()
		depth_scale = depth_sensor.get_depth_scale()
		# print("Depth Scale is: " , depth_scale)
		
		self.i = 0

	def timer_callback(self):
		
		frames = self.pipe.wait_for_frames()
	
		aligned_frames = self.align.process(frames)
		depth_frame = aligned_frames.get_depth_frame()
		color_frame = aligned_frames.get_color_frame()
		
		if (not depth_frame) or (not color_frame): return
		
		now = self.get_clock().now().to_msg()

		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		# Convertir les array numpy en image
		bridge = CvBridge()
		ros_depth_image_msg = bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
		ros_color_image_msg = bridge.cv2_to_imgmsg(color_image, encoding="rgb8")
		
		# On recup le message Camera Info
		camera_info_msg = self.get_camera_info_msg(self.intrinsics)

		# Création du header du message
		header = Header()
		header.frame_id = "camera_depth_optical_frame"
		header.stamp = self.get_clock().now().to_msg()

		# Ajout du header au message camera info
		camera_info_msg.header = header
		self.camera_info_publisher_.publish(camera_info_msg)

		# Ajout du header au message color image
		ros_color_image_msg.header = header
		self.color_image_publisher_.publish(ros_color_image_msg)
		
		# Ajout du header au message depth image
		ros_depth_image_msg.header = header
		self.depth_image_publisher_.publish(ros_depth_image_msg)

		# Code pour le nuage de points

		# points = self.pc.calculate(depth_frame)
		# decimated = decimate.process(aligned_frames).as_frameset()
		
		# verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
		# pc_msg = pc2.create_cloud_xyz32(header, verts)
	
		# self.point_cloud_publisher_.publish(pc_msg)	
		
		
		
		''' jsplu a quoi ca sert jsuis sah
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

	def get_camera_info_msg(intrinsics):
		camera_info_msg = CameraInfo()
		camera_info_msg.width = intrinsics.width
		camera_info_msg.height = intrinsics.height

		camera_info_msg.k = [
			intrinsics.fx, 0.0, intrinsics.ppx,
			0.0, intrinsics.fy, intrinsics.ppy,
			0.0, 0.0, 1.0
		]

		camera_info_msg.distortion_model = intrinsics.model
		camera_info_msg.d = intrinsics.coeffs
		return camera_info_msg


def main(args=None):
	rclpy.init(args=args) # Initialize the ROS2 Python system
	node = PointCloudPublisher() # Create an instance of the Listener node
	rclpy.spin(node) # Keep the node running, listening for messages
	node.destroy_node() # Cleanup when the node is stopped
	rclpy.shutdown() # It cleans up all ROS2 resources used by the node

if __name__ == '__main__':
	main()
