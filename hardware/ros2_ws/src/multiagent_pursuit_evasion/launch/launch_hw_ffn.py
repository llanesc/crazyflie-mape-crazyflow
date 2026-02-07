import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

package_name = 'multiagent_pursuit_evasion'


def launch_mape_node(context):
    """Launch the MAPE executor node with conditional arguments."""
    require_accel = context.launch_configurations.get('require_accel', 'false')

    args = ['-p', 'ffn']
    if require_accel.lower() == 'true':
        args.append('--require-accel')

    return [
        Node(
            package='multiagent_pursuit_evasion',
            executable='main_executor',
            name='mape_executor',
            output='screen',
            arguments=args,
            additional_env={
                'SCIPY_ARRAY_API': '1',
            },
        ),
    ]


def generate_launch_description():

    crazyflies_yaml_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'crazyflies_hw.yaml')

    motion_capture_yaml_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'motion_capture.yaml')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'require_accel',
                default_value='false',
                description='Require acceleration data from blue agents'
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        os.path.join(
                            get_package_share_directory('crazyflie'), 'launch'
                        ),
                        '/launch.py',
                    ]
                ),
                launch_arguments={
                    'crazyflies_yaml_file': crazyflies_yaml_path,
                    'motion_capture_yaml_file': motion_capture_yaml_path,
                    'gui': 'False',
                    'rviz': 'True',
                    'mocap': 'True',
                    'teleop': 'False',
                    'backend': 'cpp',
                }.items(),
            ),
            OpaqueFunction(function=launch_mape_node),
        ]
    )
