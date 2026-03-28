from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():

    # TODO: "~/ros2_ws/src/plant_detection/model/no_augmentation/best.engine"
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.expanduser(
            "~/ros2_ws/src/plant_detection/model/best/best.engine"),
        description='YOLO模型路徑'
    )
    # thinning_radius_arg = DeclareLaunchArgument(
    #     'thinning_radius',
    #     default_value='150',
    #     description='疏苗判斷半徑'
    # )
    # conf_threshold_arg = DeclareLaunchArgument(
    #     'conf_threshold_arg',
    #     default_value='0.5',
    #     description='置信度閾值'
    # )
    iou_arg = DeclareLaunchArgument(
        'iou',
        default_value='0.5',
        description='IoU閾值.'
    )

    plant_detection_node = Node(
        package='plant_detection',
        executable='plant_detector',
        parameters=[
            {"model_path": LaunchConfiguration('model_path'),
            #  "thinning_radius": LaunchConfiguration('thinning_radius'),
            #  "conf_threshold": LaunchConfiguration('conf_threshold_arg'),
             "iou": LaunchConfiguration('iou')
             },
        ],
        output='screen'
    )

    launch_description = LaunchDescription([
        model_path_arg,
        # thinning_radius_arg,
        # conf_threshold_arg,
        iou_arg,
        plant_detection_node
    ])

    return launch_description
