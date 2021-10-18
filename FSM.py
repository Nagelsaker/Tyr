import numpy as np
from HandTracking import HandTracking
from OperatorPanel import visualize
from utils import generateWorkspace
from Controller import Controller, Obstacle
from HandModel import *

def fsm():
    depthRange = [0.50, 0.85]
    pathTime = 0.6 # 0.2
    imgWidth = 1920
    imgHeight = 1080

    # Obstacles
    floor = Obstacle(zRange=[-99, 0.037])
    ceiling = Obstacle(zRange=[0.456, 99])
    innerCylinder = Obstacle(radiusRange=[0, 0.06])
    outerCylinder = Obstacle(radiusRange=[0.386, 99])
    motor = Obstacle(xRange=[0.175, 99], yRange=[-0.12, 0.05], zRange=[-99, 0.095])
    obstacles = np.array([floor, ceiling, innerCylinder, outerCylinder, motor])

    # Operator workspace
    workspaceOverlay, workspaceSections = generateWorkspace(imgHeight, imgWidth, layerSize=[0.4, 0.3])

    handTracker = HandTracking()
    hm = HandModel("left", workspaceSections)
    # Kp=[1, 0.20, 0.26, 0.30]
    # Kp=[K_p_a, K_p_r, K_p_z, K_p_t]
    controller = Controller(imgWidth, imgHeight, Kp=[0.20, 0.05, 0.07, 0.15], pathTime=pathTime, obstacles=obstacles)
    
    handTracker.startStream()

    try:
        while True: # Tracking loop
            handPoints, image, results = handTracker.getLiveLandamarks()
            visualize(results, image, workspaceOverlay) # Temp TODO Switch with an OperatorPanel object
            hm.addMeasurement(handPoints)

            controller.updateRobotPose()
            currentGesture = hm.getCurrentGesture()
            wsLoc = hm.getWorkspaceLocation(imgHeight, imgWidth)

            # FSM
            usePrecision = (currentGesture==PRECISION)

            if wsLoc == WS_TURN_LEFT:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.turnHorizontally(direction="left", precision=usePrecision)
                pass
            elif wsLoc == WS_TURN_RIGHT:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.turnHorizontally(direction="right", precision=usePrecision)
                pass
            elif wsLoc == WS_MOVE_FORWARD:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.incrementRadius(direction="forward", precision=usePrecision)
                pass
            elif wsLoc == WS_MOVE_BACKWARD:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.incrementRadius(direction="backward", precision=usePrecision)
                pass
            elif wsLoc == WS_LEFT_FORWARD:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.turnHorizontally(direction="left", precision=usePrecision)
                controller.incrementRadius(direction="forward", precision=usePrecision)
                pass
            elif wsLoc == WS_LEFT_BACKWARD:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.turnHorizontally(direction="right", precision=usePrecision)
                controller.incrementRadius(direction="backward", precision=usePrecision)
                pass
            elif wsLoc == WS_RIGHT_FORWARD:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.turnHorizontally(direction="right", precision=usePrecision)
                controller.incrementRadius(direction="forward", precision=usePrecision)
                pass
            elif wsLoc == WS_MISC and currentGesture == GRIP:
                controller.incrementGripper(direction="close")
                pass
            elif wsLoc == WS_MISC and currentGesture == UNGRIP:
                controller.incrementGripper(direction="open")
                pass
            elif wsLoc == WS_MISC and currentGesture == MOVE_HEIGHT:
                controller.updateRobotPose(updateZ=True)
                depth,_ = hm.getHandDepth()
                controller.incrementHeight(depth=depth, range=depthRange)
            elif wsLoc == WS_MISC and currentGesture == TILT_UP:
                controller.incrementTilt(direction="up")
            elif wsLoc == WS_MISC and currentGesture == TILT_DOWN:
                controller.incrementTilt(direction="down")

    except KeyboardInterrupt:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        controller.endController()


if __name__ == "__main__":
    fsm()