import copy
import numpy as np


class SimpleController():
    def __init__(self, imgHeight, imgWidth, safeZoneSize=[0.07, 0.07], angIncr=5, posIncr=0.01, scale=0.0002, angScale=0.0008):
        sizeX = safeZoneSize[0]
        sizeY = safeZoneSize[1]
        self.safeZoneX = np.array([0.5-sizeX, 0.5+sizeX]) * imgWidth
        self.safeZoneY = np.array([0.5-sizeY, 0.5+sizeY]) * imgHeight
        # self.safeZoneZ = np.array([0.30, 0.40])
        self.safeZoneZ = np.array([0.0, 10])
        self.minX = 0
        self.maxX = imgWidth
        self.minY = 0
        self.maxY = imgHeight
        self.minZ = 0.25
        self.maxZ = 0.45

        self.angIncrRad = angIncr * np.pi / 180 # In radians
        self.posIncr = posIncr
        self.scale = scale
        self.angScale = angScale

    def computeNewPose(self, pose, handDepth, handPos, prevPose):
        x = pose["position"]["x"]
        y = pose["position"]["y"]
        z = pose["position"]["z"]
        r = np.sqrt(x**2 + y**2)
        rho = np.sqrt(x**2 + y**2 + z**2)
        alpha = np.arctan2(y, x)
        beta = np.arctan2(z, r)

        # In/Out movement
        r_delta = 0
        if not (self.safeZoneY[0] < handPos[1] < self.safeZoneY[1]):
            if handPos[1] < self.safeZoneY[0]:
                r_delta = -(self.safeZoneY[0] - handPos[1])*self.scale #+ self.posIncr
            else:
                r_delta = -(self.safeZoneY[1] - handPos[1])*self.scale #- self.posIncr
        

        r_new = r + r_delta
        
        # Rotation:
        alpha_delta = 0
        if not (self.safeZoneX[0] < handPos[0] < self.safeZoneX[1]):
            if handPos[0] < self.safeZoneX[0]:
                alpha_delta = -(self.safeZoneX[0] - handPos[0])*self.angScale #+ self.angIncrRad
            else:
                alpha_delta = -(self.safeZoneX[1] - handPos[0])*self.angScale #- self.angIncrRad

        # # Up/Down
        beta_delta = 0
        handDepth = self.maxZ if handDepth > self.maxZ else (self.minZ if handDepth < self.minZ else handDepth)
        if not (self.safeZoneZ[0] < handDepth < self.safeZoneZ[1]):
            if handDepth < self.safeZoneZ[0]:
                beta_delta = -(handDepth - self.safeZoneZ[0])*10 #+ self.angIncrRad
            else:
                beta_delta = -(handDepth - self.safeZoneZ[1])*10 #- self.angIncrRad
        else:
            z = prevPose["position"]["z"]
            beta_delta = 0

        alpha_new = alpha + alpha_delta
        beta_new = beta + beta_delta
        rho_new = r_new / np.cos(beta_new)
        x_new = r_new * np.cos(alpha_new)
        y_new = r_new * np.sin(alpha_new)
        z_new = rho * np.sin(beta_new)

        newPose = copy.deepcopy(pose)
        newPose["position"]["x"] = np.around(x_new, 4)
        newPose["position"]["y"] = np.around(y_new, 4)
        newPose["position"]["z"] = np.around(z_new, 4)
        return newPose