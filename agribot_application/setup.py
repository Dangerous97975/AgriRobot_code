from setuptools import find_packages, setup

package_name = 'agribot_application'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ced',
    maintainer_email='ced@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'get_robot_pose = agribot_application.get_robot_pose:main',
            'trajectory_comparator = agribot_application.trajectory_comparator:main',
            'vis_nav_accuracy_checker = agribot_application.vis_nav_accuracy_checker:main',
            'icp_distance_detector = agribot_application.icp_distance_detector:main',
        ],
    },
)
