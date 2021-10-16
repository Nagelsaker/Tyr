import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from open_manipulator_msgs.msg import KinematicsPose, JointPosition
from open_manipulator_msgs.srv import SetKinematicsPose, SetJointPosition



class SetPositionClient(Node):

    def __init__(self):
        super().__init__('set_position_node')
        self.cli = self.create_client(SetKinematicsPose, '/goal_joint_space_path_to_kinematics_position')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetKinematicsPose.Request()

    def sendRequest(self, goalPose={"position": {"x" : 0.1, "y" : 0.0, "z" : 0.22},
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


class SetOrientationClient(Node):

    def __init__(self):
        super().__init__('set_orientation_node')
        self.cli = self.create_client(SetKinematicsPose, '/goal_joint_space_path_to_kinematics_orientation')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetKinematicsPose.Request()

    def sendRequest(self, goalPose={"position": {"x" : 0.1, "y" : 0.0, "z" : 0.22},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}},
                                pathTime=1.5):
        
        pose = Pose()

        point = Point()
        # point.x = goalPose["position"]["x"]
        # point.y = goalPose["position"]["y"]
        # point.z = goalPose["position"]["z"]
        pose.position = point

        quaternion = Quaternion()
        quaternion.x = goalPose["orientation"]["x"]
        quaternion.y = goalPose["orientation"]["y"]
        quaternion.z = goalPose["orientation"]["z"]
        quaternion.w = goalPose["orientation"]["w"]
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


class SetJointPositionClient(Node):

    def __init__(self):
        super().__init__('set_joint_position_node')
        self.cli = self.create_client(SetJointPosition, '/goal_joint_space_path')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetJointPosition.Request()

    def sendRequest(self, position, pathTime=1.5):
        '''
        Assuming joint = joint4 for now
        '''

        minVal = -1.80
        maxVal = 2.10

        if not minVal <= position["joint4"] <= maxVal:
            if position["joint4"] < minVal:
                position["joint4"] = minVal
            else:
                position["joint4"] = maxVal

        jointPosition = JointPosition()
        jointPosition.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        jointPosition.position =  [position[key] for key in position]
        jointPosition.max_accelerations_scaling_factor = 0.0
        jointPosition.max_velocity_scaling_factor = 0.0

        self.req.planning_group = ""
        self.req.joint_position = jointPosition
        self.req.path_time = pathTime

        self.future = self.cli.call_async(self.req)

class SetGripperDistanceClient(Node):

    def __init__(self):
        super().__init__('set_gripper_node')
        self.cli = self.create_client(SetJointPosition, '/goal_tool_control')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetJointPosition.Request()


    def sendRequest(self, position, pathTime=0.4):
        '''
        In:
            position: (Float) Valid numbers between -0.010 and 0.010 (m)
            pathTime: (Float)
        '''
        
        minVal = -0.010
        maxVal = 0.010

        if not minVal <= position <= maxVal:
            if position < minVal:
                position = minVal
            else:
                position = maxVal

        jointPosition = JointPosition()
        jointPosition.joint_name = ["gripper"]
        jointPosition.position =  [position]
        jointPosition.max_accelerations_scaling_factor = 0.0
        jointPosition.max_velocity_scaling_factor = 0.0

        self.req.planning_group = ""
        self.req.joint_position = jointPosition
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
        '''
        In:
            msg: 
        '''
        self.pose["position"]["x"] = msg.pose.position.x
        self.pose["position"]["y"] = msg.pose.position.y
        self.pose["position"]["z"] = msg.pose.position.z
        self.pose["orientation"]["x"] = msg.pose.orientation.x
        self.pose["orientation"]["y"] = msg.pose.orientation.y
        self.pose["orientation"]["z"] = msg.pose.orientation.z
        self.pose["orientation"]["w"] = msg.pose.orientation.w
    
    def getPose(self):
        return self.pose




class JointPositionSubscriber(Node):

    def __init__(self):
        super().__init__('joint_states')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.joints = {}

    def listener_callback(self, msg):
        '''
        In:
            msg: 
        '''
        self.joints["joint1"] = msg.position[0]
        self.joints["joint2"] = msg.position[1]
        self.joints["joint3"] = msg.position[2]
        self.joints["joint4"] = msg.position[3]
        self.joints["gripper"] = msg.position[4]

    def getPositions(self):
        return self.joints


def main(args=None):
    rclpy.init(args=args)

    poseSubscriber = PoseSubscriber()

    rclpy.spin_once(poseSubscriber)

    poseSubscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()