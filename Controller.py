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
from utils import euler_from_quaternion, quaternion_from_euler
from Communication import SetPositionClient, SetGripperDistanceClient, PoseSubscriber, SetOrientationClient, JointPositionSubscriber, SetJointPositionClient

class Controller():
    def __init__(self, imgWidth, imgHeight, Kp, pathTime, obstacles=None):
        rclpy.init()
        self.obstacles = obstacles
        self.pathTime = pathTime
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

        self.K_p_a = Kp[0]
        self.K_p_r = Kp[1]
        self.K_p_z = Kp[2]
        self.K_p_t = Kp[3]
        self.K_p_o = Kp[4]

        self.goal = np.array([-99, -99, -99])

        self.poseSubscriber = PoseSubscriber()
        self.positionClient = SetPositionClient()
        self.orientationClient = SetOrientationClient()
        self.gripperDistanceClient = SetGripperDistanceClient()
        self.jointPositionClient = SetJointPositionClient()
        self.jointSubscriber =  JointPositionSubscriber()

        self.pose = self.poseSubscriber.getPose()
        self.jointPositions = self.jointSubscriber.getPositions()
        self.desiredPose = self.pose

    def requestPose(self, desiredPose, time=None):
        if time is None: time = self.pathTime
        self.positionClient.sendRequest(desiredPose, time)
        # print(f"Sim: Setting position {desiredPose}")

    def requestOrientation(self, desiredOrientation, time=None):
        if time is None: time = self.pathTime
        self.orientationClient.sendRequest(desiredOrientation, time)

    def requestJointPositions(self, desiredPositions, time=None):
        if time is None: time = self.pathTime
        self.jointPositionClient.sendRequest(desiredPositions, time)

    def requestGripperDistance(self, desiredPosition):
            self.gripperDistanceClient.sendRequest(desiredPosition)

    def setGoal(self, point):
        pass

    def incrementRadius(self, direction, precision):
        x = self.pose["position"]["x"]
        y = self.pose["position"]["y"]
        z = self.pose["position"]["z"]
        r = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)

        if precision == True:
            r_delta = self.K_p_r / 5
        else:
            r_delta = self.K_p_r

        if direction == "forward":
            r_delta *= 1
        elif direction == "backward":
            r_delta *= -1

        r_new = r + r_delta
        x_new = r_new * np.cos(alpha)
        y_new = r_new * np.sin(alpha)

        newPose = copy.deepcopy(self.pose)
        newPose["position"]["x"] = np.around(x_new, 4)
        newPose["position"]["y"] = np.around(y_new, 4)
        newPose["position"]["z"] = np.around(z, 4)

        point = np.array([x_new, y_new, z])

        if self.isPointInWorkspace(point):
            self.requestPose(newPose)
        else:
            print("Obstacle Alert!")

    def incrementHeight(self, depth, range):
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
        newPose["position"]["x"] = np.around(x, 4)
        newPose["position"]["y"] = np.around(y, 4)
        newPose["position"]["z"] = np.around(z_new, 4)

        point = np.array([x, y, z_new])

        if self.isPointInWorkspace(point):
            self.requestPose(newPose)
        else:
            print("Obstacle Alert!")

    def turnHorizontally(self, direction, precision):

        x = self.pose["position"]["x"]
        y = self.pose["position"]["y"]
        z = self.pose["position"]["z"]

        r = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)

        if precision == True:
            alpha_delta = self.K_p_a / 5
        else:
            alpha_delta = self.K_p_a
        
        if direction == "left":
            alpha_delta *= 1
        elif direction == "right":
            alpha_delta *= -1

        alpha_new = alpha + alpha_delta
        x_new = r * np.cos(alpha_new)
        y_new = r * np.sin(alpha_new)

        newPose = copy.deepcopy(self.pose)
        newPose["position"]["x"] = np.around(x_new, 4)
        newPose["position"]["y"] = np.around(y_new, 4)
        newPose["position"]["z"] = np.around(z, 4)

        point = np.array([x_new, y_new, z])

        if self.isPointInWorkspace(point):
            self.requestPose(newPose)
        else:
            print("Obstacle Alert!")

    def incrementTilt(self, direction):
        gain = self.K_p_t
        if direction == "up":
            gain *= -1
        elif direction == "down":
            gain *= 1

        jointPositions = self.getJointPositions()
        jointPositions["joint4"] += gain
        self.requestJointPositions(jointPositions)

    def incrementOrientation(self, direction):
        gain = self.K_p_o
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
        if direction == "close":
            self.requestGripperDistance(-0.010)
        elif direction == "open":
            self.requestGripperDistance(0.010)

    def isPointInWorkspace(self, point):
        if self.obstacles is None:
            return True

        for obstacle in self.obstacles:
            if obstacle.collidesWith(point):
                return False

        return True

    def getDistanceToGoal(self):
        pass

    def getPose(self):
        return copy.deepcopy(self.pose)

    def getJointPositions(self):
        return copy.deepcopy(self.jointPositions)

    def updateRobotPose(self, updateX=False, updateY=False, updateZ=False):
        rclpy.spin_once(self.poseSubscriber) # Update pose
        newPose = self.poseSubscriber.getPose()
        if updateX:
            self.pose["position"]["x"] = newPose["position"]["x"]
        if updateY:
            self.pose["position"]["y"] = newPose["position"]["y"]
        if updateZ:
            self.pose["position"]["z"] = newPose["position"]["z"]

        self.pose["orientation"] = copy.deepcopy(newPose["orientation"])

        rclpy.spin_once(self.jointSubscriber) # Update positions
        self.jointPositions = self.jointSubscriber.getPositions()
    
    def endController(self):
        self.poseSubscriber.destroy_node()
        rclpy.shutdown()


class Obstacle():
    '''
    Obstacle class
    '''
    def __init__(self, xRange=None, yRange=None, zRange=None, radiusRange=None):
        '''
        An obstacle is assumed to be a spatial rectangle defined by 3 arrays,
        representing the space it occupies.
        In:
            xRange: (2x1 Array(Float))
            yRange: (2x1 Array(Float))
            zRange: (2x1 Array(Float))
        '''
        self.xRange = xRange
        self.yRange = yRange
        self.zRange = zRange
        self.radiusRange = radiusRange

    def collidesWith(self, point):
        '''
        In:
            Point: (2x1 Array(Float))
        
        Out:
            (Bool): False means point does not collide with obstacle
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