import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.srv import SetKinematicsPose



class EndEffPositionClient(Node):

    def __init__(self):
        super().__init__('test_node_1234')
        self.cli = self.create_client(SetKinematicsPose, '/goal_joint_space_path_to_kinematics_position')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetKinematicsPose.Request()

    def send_request(self, x_pose, pathTime):
        
        pose = Pose()

        point = Point()
        point.x = x_pose
        point.y = 0.0
        point.z = 0.22
        pose.position = point

        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.w = 0.0
        quaternion.z = 0.0
        pose.orientation = quaternion
        
        kinematics_pose = KinematicsPose()
        kinematics_pose.pose = pose
        kinematics_pose.max_accelerations_scaling_factor = 0.0
        kinematics_pose.max_velocity_scaling_factor = 0.0
        kinematics_pose.tolerance = 0.0

        self.req.planning_group = ""
        self.req.end_effector_name = "gripper"
        self.req.kinematics_pose = kinematics_pose
        self.req.path_time = pathTime

        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    minimal_client = EndEffPositionClient()
    minimal_client.send_request(0.15)

    # while rclpy.ok():
    #     rclpy.spin_once(minimal_client)
    #     if minimal_client.future.done():
    #         try:
    #             response = minimal_client.future.result()
    #         except Exception as e:
    #             minimal_client.get_logger().info(
    #                 'Service call failed %r' % (e,))
    #         else:
    #             minimal_client.get_logger().info(
    #                 'Result of add_two_ints: for %d + %d = %d' %
    #                 (minimal_client.req.a, minimal_client.req.b, response.sum))
    #         break

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()