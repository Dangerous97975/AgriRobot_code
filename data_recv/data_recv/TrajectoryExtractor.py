import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import numpy as np


class TrajectoryExtractor(Node):
    def __init__(self):
        super().__init__('trajectory_extractor')
        self.get_logger().info('Trajectory receiver node started')
        self.subscription = self.create_subscription(
            Path,
            '/mapPath',
            self.path_callback,
            10)
        self.trajectory = []

    def path_callback(self, msg):
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            qz = pose.pose.orientation.z
            qw = pose.pose.orientation.w
            self.trajectory.append([x, y, qz, qw])

        # 保存到文件
        np.savetxt('trajectory.csv', np.array(self.trajectory),
                   delimiter=',',
                   header='x,y, qz,qw')
        self.get_logger().info('Trajectory saved with %d points' % len(self.trajectory))


def main(args=None):
    rclpy.init(args=args)
    extractor = TrajectoryExtractor()
    rclpy.spin(extractor)
    extractor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
