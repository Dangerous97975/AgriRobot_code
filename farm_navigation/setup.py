import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'farm_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share/', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share/', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share/', package_name, 'maps'), glob('maps/*.yaml')),
        (os.path.join('share/', package_name, 'maps'), glob('maps/*.pgm')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ced',
    maintainer_email='ced@todo.todo',
    description='Navigation line detector for agricultural robots',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation_line_detector = farm_navigation.navigation_line_detector_node:main',
            'pointcloud_pub_test = farm_navigation.pointcloud_pub_test:main',
            'ccpp_generator = farm_navigation.ccpp_generator:main',
            'visual_nav_controller = farm_navigation.visual_nav_controller:main',
            'visual_nav_line_detector = farm_navigation.vis_nav_line_detector_v2:main',
            'visual_nav_line_detector_ridge = farm_navigation.vis_nav_line_detector_ridge:main',
            'vis_nav_line_detector_ridge = farm_navigation.vis_nav_line_detector_ridge:main',
            'performance_monitor = farm_navigation.performance_monitor:main',
            'in_out_row_detector = farm_navigation.in_out_row_detector:main',
            'rotation_tracker = farm_navigation.rotation_tracker:main',
            'nav_fsm_controller = farm_navigation.nav_fsm_controller:main',
        ],
    },
)
