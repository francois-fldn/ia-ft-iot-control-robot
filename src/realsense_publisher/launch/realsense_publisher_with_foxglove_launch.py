from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo
def generate_launch_description():
	return LaunchDescription([
		Node(
			package='realsense_publisher',
			executable='realsense_publisher',
			name='realsense_p',
			output='screen'
		),
		Node(
			package='foxglove_bridge',
			executable='foxglove_bridge',
			name='foxglove_bridge',
			output='screen',
			parameters=[{
				"port": 8765, # WebSocket port (default: 8765)
				"address": "0.0.0.0" # Allow external connections
			}]
		),
		LogInfo(
			msg="Please open Foxglove Studio and connect to ws://0.0.0.0:8765"
		)
	])
