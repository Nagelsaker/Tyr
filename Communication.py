import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.srv import SetKinematicsPose



class SetPositionClient(Node):

    def __init__(self):
        super().__init__('test_node_1234')
        self.cli = self.create_client(SetKinematicsPose, '/goal_joint_space_path_to_kinematics_position')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetKinematicsPose.Request()

    def send_request(self, goalPose={"position": {"x" : 0.1, "y" : 0.0, "z" : 0.22},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}},
                                pathTime=1.5):
        
        pose = Pose()

        point = Point()
        point.x = goalPose["position"]["x"]
        point.y = goalPose["position"]["y"]
        point.z = goalPose["position"]["z"]
        pose.position = point

        quaternion = Quaternion()
        quaternion.x = goalPose["orientation"]["x"]
        quaternion.y = goalPose["orientation"]["y"]
        quaternion.w = goalPose["orientation"]["z"]
        quaternion.z = goalPose["orientation"]["w"]
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




class PoseSubscriber(Node):

    def __init__(self):
        super().__init__('kinematics_pose')
        self.subscription = self.create_subscription(
            KinematicsPose,
            '/kinematics_pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.pose={"position": {"x" : 0.1, "y" : 0.0, "z" : 0.0},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}}

    def listener_callback(self, msg):
        self.pose["position"]["x"] = msg.pose.position.x
        self.pose["position"]["y"] = msg.pose.position.y
        self.pose["position"]["z"] = msg.pose.position.z
        self.pose["orientation"]["x"] = msg.pose.orientation.x
        self.pose["orientation"]["y"] = msg.pose.orientation.y
        self.pose["orientation"]["z"] = msg.pose.orientation.z
        self.pose["orientation"]["w"] = msg.pose.orientation.w
    
    def getPose(self):
        return self.pose


def main(args=None):
    rclpy.init(args=args)

    poseSubscriber = PoseSubscriber()

    rclpy.spin_once(poseSubscriber)

    poseSubscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()