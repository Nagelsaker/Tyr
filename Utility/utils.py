import json
import cv2
import os
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R

def euler_from_quaternion(quaternion):
    '''
    Function that converts  from quaternion to euler representation (rad)

    In:
        quaternion: (4x1) Array(Float)
    Out:
        (3x1) Array(Float)
    '''
    r = R.from_quat(quaternion)
    euler = r.as_euler("xyz")
    return euler # in radians

def quaternion_from_euler(euler):
    '''
    Function that converts euler angles (rad) to quaternion

    In:
        euler: (3x1) Array(Float)
    Out:
        (4x1) Array(Float)
    '''
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
        ang:    Float
            Angle in radians
    Out:
        R:  (3x3) Array(Float)
    '''

    R = np.diag([0, 0 ,1]).astype(np.float)
    R[0,0] = np.cos(ang)
    R[0,1] = -np.sin(ang)
    R[1,0] = np.sin(ang)
    R[1,1] = np.cos(ang)

    return R

def loadWorkspace():
    '''
    Load operator workspace stored under "data/"

    Out:
        workspaceOverlay: (1920 x 1080 x 3) Array(Float)
        workSpaceSections: Dict: (1920 x 1080) Array(Bool)
    '''
    workspaceSections = loadDictFromJSON("data/ws_section")
    workspaceOverlay = cv2.imread("data/ws_overlay.jpg")
    return workspaceOverlay, workspaceSections

def generateWorkspace(imageHeight, imageWidth, r1, r2, offset, bias=0):
    '''
    Function that generates operator workspace

    In:
        imageHeight: Float
        imageWidth: Float
        r1: Float
            inner radius of turn/misc section
        r2: Float
            outer radius of turn/misc section
        offset: Float
            turn/misc disk offset in y direction
        bias: Float
            larger bias increases the turn sections
    Out:
        workspaceOverlay: (1920 x 1080 x 3) Array(Float)
        workSpaceSections: Dict: (1920 x 1080) Array(Bool)
    '''
    height, width = imageHeight, imageWidth
    turnColor = 0 # Red
    moveColor = 2 # Blue
    miscColor = 1 # Green
    intensity = 125

    workspaceSections = {} # Key: [imageHeight x imageWidth Array(Bool)]
    workspaceSections["MoveBackward"] = np.zeros([width, height]).astype(bool)
    workspaceSections["MoveForward"] = np.zeros([width, height]).astype(bool)
    workspaceSections["TurnLeft"] = np.zeros([width, height]).astype(bool)
    workspaceSections["TurnRight"] = np.zeros([width, height]).astype(bool)
    workspaceSections["Misc"] = np.zeros([width, height]).astype(bool)

    workspaceOverlay = np.zeros((height, width, 3))

    a1 = (height+offset) / (width/2)
    a2 = (-height-offset) / (width/2)
    b = bias # bias

    # Transformation from image coordinates to workspace coordinates
    H_im_w = np.array([[1, 0, -width/2],
                    [0, 1, -height - offset],
                    [0, 0, 1]])

    X_im = np.array([[x, y, 1] for x in range(width) for y in range(height)]).T
    X_w = H_im_w @ X_im

    for idx in range(X_w.shape[1]):
        x_w, y_w, _ = X_w[:,idx]
        x, y, _ = X_im[:,idx]

        if x_w**2 + y_w**2 < r1**2:
            # Point is located in MoveBackward section
            workspaceSections["MoveBackward"][x,y] = True
            workspaceOverlay[y, x, moveColor] = intensity
        elif x_w**2 + y_w**2 < r2**2:
            # Point is located in either TurnLeft, TurnRight or Misc section
            if y_w > a1*(x_w-b):
                # Point is located in TurnLeft section
                workspaceSections["TurnLeft"][x,y] = True
                workspaceOverlay[y, x, turnColor] = intensity
            elif y_w > a2*(x_w+b):
                # Point is located in TurnRight section
                workspaceSections["TurnRight"][x,y] = True
                workspaceOverlay[y, x, turnColor] = intensity
            else:
                # Point is lmpHandsocated in Misc section
                workspaceSections["Misc"][x,y] = True
            workspaceOverlay[y, x, miscColor] = intensity
        else:
            # Point is located in MoveForward section
            workspaceSections["MoveForward"][x,y] = True
            workspaceOverlay[y, x, moveColor] = intensity

    # Save Workspace
    saveDictAsJSON(workspaceSections, "data/ws_section")
    cv2.imwrite("data/ws_overlay.jpg", workspaceOverlay)

    return workspaceOverlay, workspaceSections


def saveDictAsJSON(dict, fname):
    '''
    Function that saves a dictionary to a JSON file

    In:
        dict: Dict
        fname: String
    '''
    finalDict = {k:v.tolist() for k,v in dict.items()}
    with open(f"{fname}.json", "w") as fp:
        json.dump(finalDict, fp)

def loadDictFromJSON(fname):
    '''
    Function that loads a dictionary from a JSON file

    In:
        fname: String
    Out:
        data: Dict
    '''
    data = None
    with open(f"{fname}.json") as fp:
        data = json.load(fp)
        data = {k:np.array(v) for k,v in data.items()}
    return data


def drawLandmarks(results, image, workspaceOverlay, thread=None):
    '''
    Function that draws a 2D skeleton on top of a detected human hand in an image

    In:
        results: mpHands.Hands.process
        image: (1920x1080x3) Array(Float)
        workspaceOverlay: (1920 x 1080 x 3) Array(Float)
        thread: QThread
    Out:
        (1920x1080x3) Array(Float)
    '''
    # Draw the hand annotations on the image.
    image = image.astype("uint8")
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    workspaceOverlay = workspaceOverlay.astype("uint8")
    workspaceOverlay.flags.writeable = True
    workspaceOverlay = cv2.cvtColor(workspaceOverlay, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(workspaceOverlay, 0.5, image, 0.5, 0, image)

    mpDrawing = mp.solutions.drawing_utils
    mpDrawingStyles = mp.solutions.drawing_styles
    mpHands = mp.solutions.hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(
                image,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS,
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style())
    
    if thread is not None:
        # Convert the image for Qimage, only accepts RGB Format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def visualize(results, image, workspaceOverlay):
    '''
    Function that draws a 2D skeleton on top of a detected human hand in an image
    and displays it

    In:
        results: mpHands.Hands.process
        image: (1920x1080x3) Array(Float)
        workspaceOverlay: (1920 x 1080 x 3) Array(Float)
    '''
    image = drawLandmarks(results, image, workspaceOverlay)

    # Show stream
    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        return

def generateFilename(pathToDataset, type, name=None, removeSuffix=False):
    nrOfFiles = 0
    for filename in os.listdir(pathToDataset):
        if filename.endswith(f".{type}"):
            if name is not None:
                if not name in filename: continue
            nrOfFiles += 1
        else:
            continue
    
    generatedFname = f"{pathToDataset}{nrOfFiles+1}_"
    generatedFname += name if name is not None else ""
    if not removeSuffix: generatedFname += f".{type}"
    return generatedFname
if __name__ == "__main__":
    workspaceOverlay, workspaceSections = generateWorkspace(1080, 1920, 375, 800, 100, 30)