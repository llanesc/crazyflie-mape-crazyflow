import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

package_name = 'multiagent_pursuit_evasion'


def generate_launch_description():

    crazyflies_yaml_path = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'crazyflies_sim.yaml')

    return LaunchDescription(
        [
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
                    'gui': 'False',
                    'rviz': 'True',
                    'mocap': 'False',
                    'teleop': 'False',
                    'backend': 'cpp',
                }.items(),
            ),
            Node(
                package='multiagent_pursuit_evasion',
                executable='main_executor',
                name='mape_executor',
                output='screen',
                arguments=['-p', 'ffn'],
                additional_env={
                    'PYTHONPATH': '/home/llanesc/multiagent_pursuit_evasion/crazyflie-mape-crazyflow/env_hardware/lib/python3.12/site-packages:' + os.environ.get('PYTHONPATH', ''),
                    'SCIPY_ARRAY_API': '1',
                },
            ),
        ]
    )
