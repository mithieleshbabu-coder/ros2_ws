from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    path = '/home/mithi/ros2_ws/src/my_robot_description/urdf/robot.urdf'
    with open(path) as f:
        desc = f.read()
    return LaunchDescription([
        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             parameters=[{'robot_description': desc}]),
        Node(package='joint_state_publisher_gui',
             executable='joint_state_publisher_gui'),
        Node(package='rviz2', executable='rviz2'),
    ])
