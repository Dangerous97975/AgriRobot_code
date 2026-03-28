import os
from glob import glob
from setuptools import find_packages, setup


package_name = 'delta_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ced',
    maintainer_email='wesley222666@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'delta_robot_api = delta_robot.DeltaRobot_API:main',
            'trajectory_plan = delta_robot.Trajectory_plan:main',
            'weeding_operation = delta_robot.weeding_operation:main',
            'delta_center_tf_publisher = delta_robot.Delta_center_tf_publisher:main',
        ],
    },
)
