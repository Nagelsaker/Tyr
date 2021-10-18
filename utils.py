import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_from_quaternion(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler("xyz")
    return euler[0], euler[1], euler[2] # in radians

def quaternion_from_euler(euler):
    roll_x, pitch_y, yaw_z = euler
    r = R.from_euler('xyz', euler, degrees=False)
    quaternion = r.as_quat()
    return quaternion

def toHomogeneous(arr):
    '''
    In:
        arr:    (nxm Array) m vectors with n dimensions
    '''
    if len(arr.shape()) > 1:
        h, w = arr.shape
        temp = np.ones([h+1,w]).astype(np.float)
        temp[:h,:w] = arr
        arr = temp
    else:
        h = len(arr)
        temp = np.ones(h+1).astype(np.float)
        temp[:h] = arr
        arr = temp
    return arr


def transformationMatrix(rot, trans):
    '''
    In:
        rot:    (3x3/4x4 Array(Float))
        trans:  (3x1 Array(Float))
    
    Out:
        H:  (4x4 Array(Float))
    '''
    if rot.shape == (3,3):
        temp = np.zeros([4,4]).astype(np.float)
        temp[:3,:3] = rot
        temp[-1,-1] = 1
        H = temp
    elif rot.shape == (4,4):
        H = rot
    else:
        raise ValueError("Error: Rotation matrix must be either 3x3 or 4x4")
    
    H[:3,3] = trans

    return H


def xRotToMat(ang):
    '''
    In:
        ang:    (float) Angle in radians

    Out:
        R:  (3x3 Array(Float))
    '''

    R = np.diag([1, 0 ,0]).astype(np.float)
    R[1,1] = np.cos(ang)
    R[1,2] = -np.sin(ang)
    R[2,1] = np.sin(ang)
    R[2,2] = np.cos(ang)

    return R


def yRotToMat(ang):
    '''
    In:
        ang:    (float) Angle in radians

    Out:
        R:  (3x3 Array(Float))
    '''

    R = np.diag([0, 1 ,0]).astype(np.float)
    R[0,0] = np.cos(ang)
    R[0,2] = np.sin(ang)
    R[2,0] = -np.sin(ang)
    R[2,2] = np.cos(ang)

    return R


def zRotToMat(ang):
    '''
    In:
        ang:    (float) Angle in radians

    Out:
        R:  (3x3 Array(Float))
    '''

    R = np.diag([0, 0 ,1]).astype(np.float)
    R[0,0] = np.cos(ang)
    R[0,1] = -np.sin(ang)
    R[1,0] = np.sin(ang)
    R[1,1] = np.cos(ang)

    return R



def generateWorkspace(imageHeight, imageWidth, layerSize):
    height, width = imageHeight, imageWidth
    layerSizeH, layerSizeW = layerSize
    turnColor = 0 # Red
    moveColor = 2 # Blue
    miscColor = 1 # Green
    intensity = 125
    checkerSize = 1 # Can only be 1 TODO: Fix

    workspaceSections = {} # Key: [[yMin, yMax], [xMin, xMax]]

    workspace = np.zeros((height, width, 3))

    # Turn left
    layerIndices = np.zeros((height, width), dtype=bool)
    layerIndices[int(height*(1-layerSizeH)/2):int(height*(1+layerSizeH)/2), 0:int(width*(1-layerSizeW)/2)] = True
    workspace[layerIndices, turnColor] = intensity
    workspaceSections["TurnLeft"] = {}
    workspaceSections["TurnLeft"]["YRange"] = [int(height*(1-layerSizeH)/2), int(height*(1+layerSizeH)/2)]
    workspaceSections["TurnLeft"]["XRange"] = [0,                            int(width*(1-layerSizeW)/2)]

    # Turn right
    layerIndices = np.zeros((height, width), dtype=bool)
    layerIndices[int(height*(1-layerSizeH)/2):int(height*(1+layerSizeH)/2), int(width*(1+layerSizeW)/2):] = True
    workspace[layerIndices, turnColor] = intensity
    workspaceSections["TurnRight"] = {}
    workspaceSections["TurnRight"]["YRange"] = [int(height*(1-layerSizeH)/2), int(height*(1+layerSizeH)/2)]
    workspaceSections["TurnRight"]["XRange"] = [int(width*(1+layerSizeW)/2),  width]

    # Move Forward
    layerIndices = np.zeros((height, width), dtype=bool)
    layerIndices[0:int(height*(1-layerSizeH)/2), int(width*(1-layerSizeW)/2):int(width*(1+layerSizeW)/2)] = True
    workspace[layerIndices, moveColor] = intensity
    workspaceSections["MoveForward"] = {}
    workspaceSections["MoveForward"]["YRange"] = [0,                           int(height*(1-layerSizeH)/2)]
    workspaceSections["MoveForward"]["XRange"] = [int(width*(1-layerSizeW)/2), int(width*(1+layerSizeW)/2)]

    # Move Backward
    layerIndices = np.zeros((height, width), dtype=bool)
    layerIndices[int(height*(1+layerSizeH)/2):, int(width*(1-layerSizeW)/2):int(width*(1+layerSizeW)/2)] = True
    workspace[layerIndices, moveColor] = intensity
    workspaceSections["MoveBackward"] = {}
    workspaceSections["MoveBackward"]["YRange"] = [int(height*(1+layerSizeH)/2), height]
    workspaceSections["MoveBackward"]["XRange"] = [int(width*(1-layerSizeW)/2),  int(width*(1+layerSizeW)/2)]

    # Misc Section
    layerIndices = np.zeros((height, width), dtype=bool)
    layerIndices[int(height*(1-layerSizeH)/2):int(height*(1+layerSizeH)/2), int(width*(1-layerSizeW)/2):int(width*(1+layerSizeW)/2)] = True
    workspace[layerIndices, miscColor] = intensity
    workspaceSections["Misc"] = {}
    workspaceSections["Misc"]["YRange"] = [int(height*(1-layerSizeH)/2), int(height*(1+layerSizeH)/2)]
    workspaceSections["Misc"]["XRange"] = [int(width*(1-layerSizeW)/2),  int(width*(1+layerSizeW)/2)]

    # Turn Left & Move Forward
    layerIndices = np.zeros((height, width), dtype=bool)
    checkerIndices = np.zeros((int(height*(1-layerSizeH)/2), int(width*(1-layerSizeW)/2)), dtype=bool)
    checkerIndices[::(2*checkerSize), ::(2*checkerSize)] = np.ones((checkerSize), dtype=bool)
    checkerIndices[checkerSize::(2*checkerSize), checkerSize::(2*checkerSize)] = True

    blueIndices = np.copy(layerIndices)
    redIndices = np.copy(layerIndices)
    blueIndices[0:int(height*(1-layerSizeH)/2), 0:int(width*(1-layerSizeW)/2)] = checkerIndices
    redIndices[0:int(height*(1-layerSizeH)/2), 0:int(width*(1-layerSizeW)/2)] = np.invert(checkerIndices)

    workspace[blueIndices, moveColor] = intensity
    workspace[redIndices, turnColor] = intensity
    workspaceSections["LeftForward"] = {}
    workspaceSections["LeftForward"]["YRange"] = [0, int(height*(1-layerSizeH)/2)]
    workspaceSections["LeftForward"]["XRange"] = [0, int(width*(1-layerSizeW)/2)]

    # Turn Left & Move Backward
    layerIndices = np.zeros((height, width), dtype=bool)
    checkerIndices = np.zeros((int(height*(1-layerSizeH)/2), int(width*(1-layerSizeW)/2)), dtype=bool)
    checkerIndices[::(2*checkerSize), ::(2*checkerSize)] = True
    checkerIndices[checkerSize::(2*checkerSize), checkerSize::(2*checkerSize)] = True

    blueIndices = np.copy(layerIndices)
    redIndices = np.copy(layerIndices)
    blueIndices[int(height*(1+layerSizeH)/2):, 0:int(width*(1-layerSizeW)/2)] = checkerIndices
    redIndices[int(height*(1+layerSizeH)/2):, 0:int(width*(1-layerSizeW)/2)] = np.invert(checkerIndices)

    workspace[blueIndices, moveColor] = intensity
    workspace[redIndices, turnColor] = intensity
    workspaceSections["LeftBackward"] = {}
    workspaceSections["LeftBackward"]["YRange"] = [int(height*(1+layerSizeH)/2), height]
    workspaceSections["LeftBackward"]["XRange"] = [0, int(width*(1-layerSizeW)/2)]

    # Turn Right & Move Backward
    layerIndices = np.zeros((height, width), dtype=bool)
    checkerIndices = np.zeros((int(height*(1-layerSizeH)/2), int(width*(1-layerSizeW)/2)), dtype=bool)
    checkerIndices[::(2*checkerSize), ::(2*checkerSize)] = True
    checkerIndices[checkerSize::(2*checkerSize), checkerSize::(2*checkerSize)] = True

    blueIndices = np.copy(layerIndices)
    redIndices = np.copy(layerIndices)
    blueIndices[int(height*(1+layerSizeH)/2):, int(width*(1+layerSizeW)/2):] = checkerIndices
    redIndices[int(height*(1+layerSizeH)/2):, int(width*(1+layerSizeW)/2):] = np.invert(checkerIndices)

    workspace[blueIndices, moveColor] = intensity
    workspace[redIndices, turnColor] = intensity
    workspaceSections["RightBackward"] = {}
    workspaceSections["RightBackward"]["YRange"] = [int(height*(1+layerSizeH)/2), height]
    workspaceSections["RightBackward"]["XRange"] = [int(width*(1+layerSizeW)/2), width]

    # Turn Right & Move Forward
    layerIndices = np.zeros((height, width), dtype=bool)
    checkerIndices = np.zeros((int(height*(1-layerSizeH)/2), int(width*(1-layerSizeW)/2)), dtype=bool)
    checkerIndices[::(2*checkerSize), ::(2*checkerSize)] = True
    checkerIndices[checkerSize::(2*checkerSize), checkerSize::(2*checkerSize)] = True

    blueIndices = np.copy(layerIndices)
    redIndices = np.copy(layerIndices)
    blueIndices[0:int(height*(1-layerSizeH)/2), int(width*(1+layerSizeW)/2):] = checkerIndices
    redIndices[0:int(height*(1-layerSizeH)/2), int(width*(1+layerSizeW)/2):] = np.invert(checkerIndices)

    workspace[blueIndices, moveColor] = intensity
    workspace[redIndices, turnColor] = intensity
    workspaceSections["RightForward"] = {}
    workspaceSections["RightForward"]["YRange"] = [0, int(height*(1-layerSizeH)/2)]
    workspaceSections["RightForward"]["XRange"] = [int(width*(1+layerSizeW)/2), width]
                                         
    return workspace, workspaceSections