from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'realsense_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('lib', package_name), glob(package_name + '/yolov5/*.tflite')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='francois',
    maintainer_email='francois@todo.todo',
    description='ROS package to publish on the topic point_cloud le nuage de points genere par la realsense D435i',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        		'realsense_publisher = realsense_publisher.realsense_p:main',
                'ball_detector = realsense_publisher.BallDetector3D:main',
        ],
    },
)
