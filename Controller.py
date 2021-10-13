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
from Communication import SetPositionClient, PoseSubscriber

class Controller():
    def __init__(self, imgWidth, imgHeight, depthRange, obstacles=None, pathTime=0.2):
        rclpy.init()
        self.obstacles = obstacles
        self.pathTime = pathTime
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.depthRange = depthRange

        self.goal = np.array([-99, -99, -99])

        self.poseSubscriber = PoseSubscriber()
        self.positionClient = SetPositionClient()

        self.pose = self.poseSubscriber.getPose()
        self.desiredPose = self.pose

    def requestPose(self, desiredPose, time=None):
        if time is None: time = self.pathTime
        self.positionClient.sendRequest(desiredPose, time)
        # print(f"Sim: Setting position {desiredPose}")
        pass

    def setGoal(self, point):
        pass

    def incrementRadiusAndZ(self, palmPos, depth):
        x = self.pose["position"]["x"]
        y = self.pose["position"]["y"]
        z = self.pose["position"]["z"]
        r = np.sqrt(x**2 + y**2)
        rho = np.sqrt(x**2 + y**2 + z**2)
        alpha = np.arctan2(y, x)

        K_p_r = 0.07 # TODO: Tune
        K_p_z = 0.13 # TODO: Tune

        # In/Out movement
        r_delta = (palmPos[1] - 1/2)*K_p_r
        r_new = r + r_delta
        
        # print(f"D: {depth}")
        if not (self.depthRange[0] < depth < self.depthRange[1]):
            if depth < self.depthRange[0]:
                depth = self.depthRange[0]
            else:
                depth = self.depthRange[1]

        z_delta = (-1)*(depth - self.depthRange[0] - (self.depthRange[1]-self.depthRange[0])/2) / (self.depthRange[1] - self.depthRange[0]) * K_p_z
        z_new = z + z_delta

        # print(f"Moving in directions: r={r_delta/np.abs(r_delta)}\t\tz={z_delta/np.abs(z_delta)} \r", end="")

        beta_new = np.arctan2(z_new, r_new)
        x_new = r_new * np.cos(alpha)
        y_new = r_new * np.sin(alpha)
        z_new = rho * np.sin(beta_new)

        newPose = copy.deepcopy(self.pose)
        newPose["position"]["x"] = np.around(x_new, 4)
        newPose["position"]["y"] = np.around(y_new, 4)
        newPose["position"]["z"] = np.around(z_new, 4)

        point = np.array([x_new, y_new, z_new])

        if self.isPointInWorkspace(point):
            self.requestPose(newPose)
        else:
            print("Obstacle Alert!")

    def incrementXY(self, palmPos):
        # Moves downwards in z direction as well. TODO: Possibly recalibrate
        # Turns only to the left
        K_p_a = 0.5

        x = self.pose["position"]["x"]
        y = self.pose["position"]["y"]
        z = self.pose["position"]["z"]

        r = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)

        alpha_delta = (palmPos[0] - 1/2) * K_p_a

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

    def incrementGripper(self, palmPos):
        pass

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

    def updateRobotPose(self):
        rclpy.spin_once(self.poseSubscriber) # Update pose
        self.pose = self.poseSubscriber.getPose()
    
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