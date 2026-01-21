#!/bin/bash

colcon build --symlink-install

if [ -f "install/setup.bash" ]; then
    source install/setup.bash
fi

export TURTLEBOT3_MODEL=burger
export PROJECT_MODEL=turtlebot3_burger_d435i

ros2 launch turtlebot3_gazebo projects_house_world.launch.py