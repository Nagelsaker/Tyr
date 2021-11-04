import numpy as np
from Hand.HandTracking import HandTracking
from Utility.utils import generateWorkspace, drawLandmarks, loadWorkspace, visualize
from Comms.Controller import Controller, Obstacle
from PyQt5.QtGui import QImage
from Hand.HandModel import *

def fsm(thread=None):
    depthRange = [0.60, 0.85]
    pathTime = 0.6 # 0.2
    imgWidth = 1920
    imgHeight = 1080
    camSN = "836612072676"

    # Obstacles
    floor = Obstacle(zRange=[-99, 0.035])
    ceiling = Obstacle(zRange=[0.456, 99])
    innerCylinder = Obstacle(radiusRange=[0, 0.06])
    outerCylinder = Obstacle(radiusRange=[0.386, 99])
    motor = Obstacle(xRange=[0.185, 99], yRange=[-0.12, 0.05], zRange=[-99, 0.095])
    obstacles = np.array([floor, ceiling, innerCylinder, outerCylinder, motor])

    # Operator workspace
    # workspaceOverlay, workspaceSections = generateWorkspace(imgHeight, imgWidth, 500, 950, 250)
    workspaceOverlay, workspaceSections = loadWorkspace()

    handTracker = HandTracking(camSN)
    hm = HandModel("left", workspaceSections)
    # Kp=[1, 0.20, 0.26, 0.30]
    # Kp=[K_p_a, K_p_r, K_p_z, K_p_t]
    controller = Controller(imgWidth, imgHeight, Kp=[0.20, 0.05, 0.07, 0.15, np.deg2rad(10)], pathTime=pathTime, obstacles=obstacles)
    controller.updateRobotPose(updateX=True, updateY=True, updateZ=True)
    handTracker.startStream()

    try:
        while True: # Tracking loop
            if thread is not None:
                if thread.isInterruptionRequested():
                    raise Exception

            handPoints, image, results = handTracker.getLiveLandamarks()
            if thread is not None:
                imgLM =  drawLandmarks(results, image, workspaceOverlay, thread)
                h, w, ch = imgLM.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(imgLM.data, w, h, bytesPerLine, QImage.Format_RGB888).scaled(thread.w, thread.h)
                thread.changePixmap.emit(convertToQtFormat)
            # visualize(results, image, workspaceOverlay) # Temp TODO Switch with an OperatorPanel object
            hm.addMeasurement(handPoints)

            controller.updateRobotPose()
            currentGesture = hm.getCurrentGesture()
            wsLoc = hm.getWorkspaceLocation(imgHeight, imgWidth)

            # FSM
            usePrecision = (currentGesture==PRECISION)

            if wsLoc == WS_TURN_LEFT:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.turnHorizontally(direction="left", precision=usePrecision)
                # controller.updateRobotPose(updateX=True, updateY=True)
            elif wsLoc == WS_TURN_RIGHT:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.turnHorizontally(direction="right", precision=usePrecision)
                # controller.updateRobotPose(updateX=True, updateY=True)
            elif wsLoc == WS_MOVE_FORWARD:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.incrementRadius(direction="forward", precision=usePrecision)
                # controller.updateRobotPose(updateX=True, updateY=True)
            elif wsLoc == WS_MOVE_BACKWARD:
                controller.updateRobotPose(updateX=True, updateY=True)
                controller.incrementRadius(direction="backward", precision=usePrecision)
                # controller.updateRobotPose(updateX=True, updateY=True)
            elif wsLoc == WS_MISC and currentGesture == GRIP:
                controller.incrementGripper(direction="close")
            elif wsLoc == WS_MISC and currentGesture == UNGRIP:
                controller.incrementGripper(direction="open")
            elif wsLoc == WS_MISC and currentGesture == MOVE_HEIGHT:
                controller.updateRobotPose(updateZ=True)
                depth,_ = hm.getHandDepth()
                controller.incrementHeight(depth=depth, range=depthRange)
                # controller.updateRobotPose(updateZ=True)
            elif wsLoc == WS_MISC and currentGesture == TILT_UP:
                # controller.incrementTilt(direction="up")
                controller.incrementOrientation(direction="up")
            elif wsLoc == WS_MISC and currentGesture == TILT_DOWN:
                # controller.incrementTilt(direction="down")
                controller.incrementOrientation(direction="down")

    except Exception:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        controller.endController()
        if thread is not None:
            thread.quit()


if __name__ == "__main__":
    fsm()