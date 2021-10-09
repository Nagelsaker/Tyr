import numpy as np


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