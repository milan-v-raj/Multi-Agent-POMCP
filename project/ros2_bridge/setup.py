import os
from glob import glob
from setuptools import setup

package_name = 'deep_pomcp_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Milan V Raj',
    maintainer_email='milanvraj@gmail.com',
    description='ROS 2 Node for Multi-Agent PointNet ET-DAG-Deep-POMCP Pursuit-Evasion',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pomcp_node = deep_pomcp_ros2.pomcp_ros2_node:main',
        ],
    },
)

