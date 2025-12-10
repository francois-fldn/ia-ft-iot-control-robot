import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/francois/00-projets/ia-ft-iot-control-robot/ros2_packages/install/realsense_subscriber'
