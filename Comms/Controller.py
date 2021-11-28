'''
* Talks with the manipulator
* Sends requests to manipulator
* Gets pose info
* Does not request illegal poses
* Calculates distance to goal object, in (x,y,z)
'''

import copy
import rclpy
import numpy as np
from Utility.utils import euler_from_quaternion, quaternion_from_euler
from Comms.Communication import *

class Controller():
    '''
    Controller class

    This class establishes a connection with a Robotis OpenManipulator,
    calculates the manipulators polar coordinates and requests position
    and orientation changes.

    Attributes
    ----------
        obstacles: (5x1) Array(Obstacle)
        pathTime: Float
        imgWidth: Int
        imgHeight: Int

        K_p_psi: Float
            Proportionality control value for the horizontal turning velocity
        K_p_r: Float
            Proportionality control value for the horizontal radial velocity
        K_p_z: Float
            Proportionality control value for the vertical velocity
        K_p_theta: Float
            Proportionality control value for the end-effectors tilt velocity.
            Default tilting rule, where end-effector orientation is altered
            by only moving the closest revolute joint to the end-effector.
        K_p_phi: Float
            Proportionality control value for the end-effectors tilt velocity.
            Used in alternative tilting rule, where end-effector orientation
            is altered by moving all revolute joints, but maintaining the
            end-effector position.
        pose: Dict(Dict(Float))
            Contains both position and orientation of end effector
        jointPositions: Dict(Float)
            Contains angular motor positions of all joints, and linear position
            of gripper.
        desiredPose: Dict(Dict(Float))
            The desired calculated pose
    '''
    def __init__(self, imgWidth, imgHeight, Kp, pathTime, obstacles=None):
        '''
        In:
            imgWidth: Int
            imgHeight: Int
            Kp: (5x1) Array(Float)
            pathTime: Float
            obstacles: (5x1) Array(Obstacle)
        '''
        rclpy.init()
        self.obstacles = obstacles
        self.pathTime = pathTime
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

        self.K_p_psi = Kp[0]
        self.K_p_r = Kp[1]
        self.K_p_z = Kp[2]
        self.K_p_theta = Kp[3]
        self.K_p_phi = Kp[4]

        self.poseSubscriber = PoseSubscriber()
        self.positionClient = SetPositionClient()
        self.orientationClient = SetOrientationClient()
        self.gripperDistanceClient = SetGripperDistanceClient()
        self.jointPositionClient = SetJointPositionClient()
        self.jointSubscriber =  JointPositionSubscriber()

        self.pose = copy.deepcopy(self.poseSubscriber.getPose())
        self.jointPositions = self.jointSubscriber.getPositions()
        self.desiredPose = self.pose

    def requestPose(self, desiredPose, time=None):
        '''
        Method which sends a pose request to the OpenManipulator.

        In:
            desiredPose: Dict(Dict(Float))
            time: Float
                The time the manipulator uses to move to its new pose
        '''
        if time is None: time = self.pathTime
        self.positionClient.sendRequest(desiredPose, time)

    def requestOrientation(self, desiredOrientation, time=None):
        '''
        Method which sends an orientation request to the OpenManipulator.

        In:
            desiredOrientation: Dict(Dict(Float))
            time: Float
                The time the manipulator uses to move to its new orientation
        '''
        if time is None: time = self.pathTime
        self.orientationClient.sendRequest(desiredOrientation, time)

    def requestJointPositions(self, desiredPositions, time=None):
        '''
        Method which sends joint position requests to the OpenManipulator.

        In:
            desiredPositions: Dict(Float)
            time: Float
                The time the manipulator uses to move to its joints to new positions
        '''
        if time is None: time = self.pathTime
        self.jointPositionClient.sendRequest(desiredPositions, time)

    def requestGripperDistance(self, desiredPosition):
        '''
        Method which sends a gripper position request to the OpenManipulator.

        In:
            desiredPosition: (Float)
                Valid numbers between -0.010 and 0.010 (m)
        '''
        self.gripperDistanceClient.sendRequest(desiredPosition)

    def incrementRadius(self, direction, precision):
        '''
        Method which calculates the manipulators new horizontal radius.

        Calculates the manipulators polar coordinates, adjusts the radius according
        to the K_p_r value, recalculates the cartesian coordinates. Then it checks whether
        the new pose does not collide with obstacles, and is inside the manipulators
        workspace. If these requirements are fulfilled, the pose is sent as a request
        to the OpenManipulator, which has its own safety checks for new poses.

        In:
            direction: String
                either "forward" or "backward"
            precision: Bool
                whether the operator wants precise slow movements, or normal paced movements
        '''
        x = self.pose["position"]["x"]
        y = self.pose["position"]["y"]
        z = self.pose["position"]["z"]
        r = np.sqrt(x**2 + y**2)
        psi = np.arctan2(y, x)

        if precision == True:
            r_delta = self.K_p_r / 5
        else:
            r_delta = self.K_p_r

        if direction == "forward":
            r_delta *= 1
        elif direction == "backward":
            r_delta *= -1

        r_new = r + r_delta
        x_new = r_new * np.cos(psi)
        y_new = r_new * np.sin(psi)

        newPose = copy.deepcopy(self.pose)
        newPose["position"]["x"] = x_new
        newPose["position"]["y"] = y_new
        newPose["position"]["z"] = z

        point = np.array([x_new, y_new, z])

        if self.isPointInWorkspace(point):
            self.requestPose(newPose)
            self.pose = copy.deepcopy(newPose)
        else:
            print("Obstacle Alert!")

    def incrementHeight(self, depth, range):
        '''
        Method which calculates the manipulators new height.

        Calculates the manipulators height according to the K_p_z value and measured
        hand depth. Then it checks whether the new pose does not collide with obstacles,
        and is inside the manipulators workspace. If these requirements are fulfilled,
        the pose is sent as a request to the OpenManipulator, which has its own
        safety checks for new poses.

        In:
            depth: Float
                measured hand depth
            range: (2x1) Array(Float)
                min and max expected depth values
        '''
        x = self.pose["position"]["x"]
        y = self.pose["position"]["y"]
        z = self.pose["position"]["z"]
        
        if not (range[0] < depth < range[1]):
            if depth < range[0]:
                depth = range[0]
            else:
                depth = range[1]

        z_delta = (-1)*(depth - range[0] - (range[1]-range[0])/2) / (range[1] - range[0]) * self.K_p_z
        z_new = z + z_delta

        newPose = copy.deepcopy(self.pose)
        newPose["position"]["x"] = x
        newPose["position"]["y"] = y
        newPose["position"]["z"] = z_new

        point = np.array([x, y, z_new])

        if self.isPointInWorkspace(point):
            self.requestPose(newPose)
            self.pose = copy.deepcopy(newPose)
        else:
            print("Obstacle Alert!")

    def turnHorizontally(self, direction, precision):
        '''
        Method which calculates the manipulators new horizontal angular position.

        Calculates the manipulators polar coordinates, adjusts the angle psi according
        to the K_p_psi value, recalculates the cartesian coordinates. Then it checks whether
        the new pose does not collide with obstacles, and is inside the manipulators
        workspace. If these requirements are fulfilled, the pose is sent as a request
        to the OpenManipulator, which has its own safety checks for new poses.
        
        In:
            direction: String
                either "left" or "right"
            precision: Bool
                whether the operator wants precise slow movements, or normal paced movements
        '''

        x = self.pose["position"]["x"]
        y = self.pose["position"]["y"]
        z = self.pose["position"]["z"]

        r = np.sqrt(x**2 + y**2)
        psi = np.arctan2(y, x)

        if precision == True:
            psi_delta = self.K_p_psi / 5
        else:
            psi_delta = self.K_p_psi
        
        if direction == "left":
            psi_delta *= 1
        elif direction == "right":
            psi_delta *= -1

        psi_new = psi + psi_delta
        x_new = r * np.cos(psi_new)
        y_new = r * np.sin(psi_new)

        newPose = copy.deepcopy(self.pose)
        newPose["position"]["x"] = x_new
        newPose["position"]["y"] = y_new
        newPose["position"]["z"] = z

        point = np.array([x_new, y_new, z])

        if self.isPointInWorkspace(point):
            self.requestPose(newPose)
            self.pose = copy.deepcopy(newPose)
        else:
            print("Obstacle Alert!")

    def incrementTilt(self, direction):
        '''
        Method which calculates the end-effectors new orientation.

        Default tilting rule, where end-effector orientation is altered
        by only moving the closest revolute joint to the end-effector.

        In:
            direction: String
                either "up" or "down"
        '''
        gain = self.K_p_theta
        if direction == "up":
            gain *= -1
        elif direction == "down":
            gain *= 1

        jointPositions = self.getJointPositions()
        jointPositions["joint4"] += gain
        self.requestJointPositions(jointPositions)

    def incrementOrientation(self, direction):
        '''
        Method which calculates the end-effectors new orientation.

        Alternative tilting rule, where end-effector orientation
        is altered by moving all revolute joints, but maintaining the
        end-effector position.

        In:
            direction: String
                either "up" or "down"
        '''
        gain = self.K_p_phi
        if direction == "up":
            gain *= -1
        elif direction == "down":
            gain *= 1
        
        newPose = copy.deepcopy(self.pose)
        orientation = newPose["orientation"]
        quaternion = np.array([orientation["x"], orientation["y"], orientation["z"], orientation["w"]])

        euler = euler_from_quaternion(quaternion)
        euler[1] += gain # Adjust y angle

        newQuaternion = quaternion_from_euler(euler)
        orientation["x"] = newQuaternion[0] 
        orientation["y"] = newQuaternion[1] 
        orientation["z"] = newQuaternion[2] 
        orientation["w"] = newQuaternion[3]

        newPose["orientation"] = copy.deepcopy(orientation)
        self.requestOrientation(newPose)

    def incrementGripper(self, direction):
        '''
        Method which increments the gripper distance by either -0.10 or 0.10 m.

        In:
            direction: String
                either "close" or "open"
        '''
        if direction == "close":
            self.requestGripperDistance(-0.010)
        elif direction == "open":
            self.requestGripperDistance(0.010)

    def isPointInWorkspace(self, point):
        '''
        Method which calculates whether the given point is valid.

        Checks if the point collides with any of the known obstacles, and is within
        the manipulators workspace.

        In:
            point: (2x1) Array(Float)
        '''
        if self.obstacles is None:
            return True

        for obstacle in self.obstacles:
            if obstacle.collidesWith(point):
                return False

        return True

    def getPose(self):
        '''
        Method which returns a copy of the pose
        '''
        return copy.deepcopy(self.pose)

    def getJointPositions(self):
        '''
        Method which returns a copy of the joint positions
        '''
        return copy.deepcopy(self.jointPositions)

    def updateRobotPose(self, updateX=False, updateY=False, updateZ=False):
        '''
        Method which updates the manipulators pose.

        Only updates necessary coordinates, as numerical errors will cause the
        manipulator to drift if all of its coordinates are updated each time
        a new pose is requested.

        In:
            updateX: Bool
                whether the x-position should be updated
            updateY: Bool
                whether the y-position should be updated
            updateZ: Bool
                whether the z-position should be updated
        '''
        rclpy.spin_once(self.poseSubscriber) # Update pose
        newPose = self.poseSubscriber.getPose()
        if updateX:
            self.pose["position"]["x"] = copy.deepcopy(newPose["position"]["x"])
        if updateY:
            self.pose["position"]["y"] = copy.deepcopy(newPose["position"]["y"])
        if updateZ:
            self.pose["position"]["z"] = copy.deepcopy(newPose["position"]["z"])

        self.pose["orientation"] = copy.deepcopy(newPose["orientation"])

        rclpy.spin_once(self.jointSubscriber) # Update positions
        self.jointPositions = self.jointSubscriber.getPositions()
        
    
    def endController(self):
        '''
        Method which disconnects from the OpenManipulator in an orderly fashion.
        '''
        self.poseSubscriber.destroy_node()
        rclpy.shutdown()


class Obstacle():
    '''
    Obstacle class

    Obstacles can be represented as either rectangles or cylinders, and are defined
    by ranges in the cartesian or polar coordinates.

    '''
    def __init__(self, xRange=None, yRange=None, zRange=None, radiusRange=None):
        '''
        An obstacle is assumed to be a spatial rectangle defined by 3 arrays,
        representing the space it occupies.
        In:
            xRange: (2x1) Array(Float)
                cartesian range describing the rectangle
            yRange: (2x1) Array(Float)
                cartesian range describing the rectangle
            zRange: (2x1) Array(Float)
                cartesian range describing the rectangle
            radiusRange: (2x1) Array(Float)
                polar coordinates range describing a cylinder, or donut
                intersecting the horizontal plane perpendicularly
                with a z range of [-99, 99] m
        '''
        self.xRange = xRange
        self.yRange = yRange
        self.zRange = zRange
        self.radiusRange = radiusRange

    def collidesWith(self, point):
        '''
        Method which checks whether the point collides with the obstacle.

        In:
            point: (2x1) Array(Float)
        Out:
            Bool
        '''
        if not (self.xRange is None and self.yRange is None and self.zRange is None):
            collision = np.array([False, False, False])
            if self.xRange is not None:
                if self.xRange[0] < point[0] < self.xRange[1]:
                    collision[0] = True
            else:
                    collision[0] = True
            if self.yRange is not None:
                if self.yRange[0] < point[1] < self.yRange[1]:
                    collision[1] = True
            else:
                    collision[1] = True
            if self.zRange is not None:
                if self.zRange[0] < point[2] < self.zRange[1]:
                    collision[2] = True
            else:
                    collision[2] = True
            
            if np.all(collision):
                return True
                
        if self.radiusRange is not None:
            if self.radiusRange[0] < np.sqrt(point[0]**2 + point[1]**2) < self.radiusRange[1]:
                return True

        # Point does not collide
        return False