"""
ROS 2 Launch file for Multi-Agent ET-DAG-Deep-POMCP Pursuit-Evasion System.
Spawns two autonomous pursuer hunter nodes within namespaced topics.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'arena_width',
            default_value='8.0',
            description='Arena width in meters'
        ),
        DeclareLaunchArgument(
            'arena_height',
            default_value='6.0',
            description='Arena height in meters'
        ),
        DeclareLaunchArgument(
            'control_freq',
            default_value='20.0',
            description='Control frequency in Hz'
        ),

        # Hunter 0 Node
        Node(
            package='deep_pomcp_ros2',
            executable='pomcp_node',
            name='pursuer_agent_0',
            namespace='hunter_0',
            parameters=[{
                'agent_id': 0,
                'arena_width_m': LaunchConfiguration('arena_width'),
                'arena_height_m': LaunchConfiguration('arena_height'),
                'control_freq_hz': LaunchConfiguration('control_freq'),
            }],
            output='screen'
        ),

        # Hunter 1 Node
        Node(
            package='deep_pomcp_ros2',
            executable='pomcp_node',
            name='pursuer_agent_1',
            namespace='hunter_1',
            parameters=[{
                'agent_id': 1,
                'arena_width_m': LaunchConfiguration('arena_width'),
                'arena_height_m': LaunchConfiguration('arena_height'),
                'control_freq_hz': LaunchConfiguration('control_freq'),
            }],
            output='screen'
        ),
    ])

