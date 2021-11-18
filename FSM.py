import json
import numpy as np
from Hand.HandTracking import HandTracking
from Utility.utils import generateWorkspace, drawLandmarks, loadWorkspace, visualize
from Comms.Controller import Controller, Obstacle
from PyQt5.QtGui import QImage
from Hand.HandModel import *


class FSM():
    def __init__(self):
        f = open("data/settings.json")
        settings = json.load(f)
        self.depthRange = settings["depthRange"] # [0.60, 0.85]
        self.pathTime = settings["pathTime"] # 0.2
        self.imgWidth = settings["imgWidth"]
        self.imgHeight = settings["imgHeight"]
        self.camSN = settings["camSN"]
        # Kp=[K_p_psi, K_p_r, K_p_z, K_p_theta, K_p_phi]
        self.Kp_default = settings["Kp_default"]
        wristAngle_threshold = settings["wristAngle_threshold"]
        thumbAngle_threshold = settings["thumbAngle_threshold"]
        fingerAngle_threshold = settings["fingerAngle_threshold"]

        # Obstacles
        floor = Obstacle(zRange=settings["floor"]["zRange"])
        ceiling = Obstacle(zRange=settings["ceiling"]["zRange"])
        innerCylinder = Obstacle(radiusRange=settings["innerCylinder"]["radiusRange"])
        outerCylinder = Obstacle(radiusRange=settings["outerCylinder"]["radiusRange"])
        motor = Obstacle(settings["motor"]["xRange"], settings["motor"]["yRange"], settings["motor"]["zRange"])
        self.obstacles = np.array([floor, ceiling, innerCylinder, outerCylinder, motor])

        # Operator workspace
        self.workspaceOverlay, self.workspaceSections = loadWorkspace()

        self.handTracker = HandTracking(self.camSN)
        self.hm = HandModel("left", self.workspaceSections, wristAngle_threshold, thumbAngle_threshold, fingerAngle_threshold)
        self.controller = Controller(self.imgWidth, self.imgHeight, Kp=self.Kp_default, pathTime=self.pathTime, obstacles=self.obstacles)

    def setWristThreshold(self, threshold):
        self.hm.setWristThreshold(threshold)

    def setFingerThreshold(self, threshold):
        self.hm.setFingerThreshold(threshold)

    def setThumbThreshold(self, threshold):
        self.hm.setThumbThreshold(threshold)

    def run(self, thread=None):
        self.controller.updateRobotPose(updateX=True, updateY=True, updateZ=True)
        self.handTracker.startStream()

        try:
            while True: # Tracking loop
                handPoints, image, results = self.handTracker.getLiveLandamarks()

                if thread is not None:
                    if thread.isInterruptionRequested():
                        raise Exception

                    imgLM =  drawLandmarks(results, image, self.workspaceOverlay, thread)
                    h, w, ch = imgLM.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(imgLM.data, w, h, bytesPerLine, QImage.Format_RGB888).scaled(thread.w, thread.h)
                    thread.changePixmap.emit(convertToQtFormat)
                # visualize(results, image, self.workspaceOverlay) # Temp TODO Switch with an OperatorPanel object
                
                self.hm.addMeasurement(handPoints)
                
                self.controller.updateRobotPose()
                currentGesture = self.hm.getCurrentGesture()
                wsLoc = self.hm.getWorkspaceLocation(self.imgHeight, self.imgWidth)

                # Send Current gesture to GUI application
                thread.activateGesture.emit(currentGesture)

                # FSM
                usePrecision = (currentGesture==PRECISION)

                if wsLoc == WS_TURN_LEFT and currentGesture != STOP:
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.turnHorizontally(direction="left", precision=usePrecision)
                elif wsLoc == WS_TURN_RIGHT and currentGesture != STOP:
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.turnHorizontally(direction="right", precision=usePrecision)
                elif wsLoc == WS_MOVE_FORWARD and currentGesture != STOP:
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.incrementRadius(direction="forward", precision=usePrecision)
                elif wsLoc == WS_MOVE_BACKWARD and currentGesture != STOP:
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.incrementRadius(direction="backward", precision=usePrecision)
                elif wsLoc == WS_MISC and currentGesture == GRIP:
                    # thread.activateGesture.emit(1)
                    self.controller.incrementGripper(direction="close")
                elif wsLoc == WS_MISC and currentGesture == UNGRIP:
                    # thread.activateGesture.emit(0)
                    self.controller.incrementGripper(direction="open")
                elif wsLoc == WS_MISC and currentGesture == MOVE_HEIGHT:
                    # thread.activateGesture.emit(4)
                    self.controller.updateRobotPose(updateZ=True)
                    depth,_ = self.hm.getHandDepth()
                    self.controller.incrementHeight(depth=depth, range=self.depthRange)
                elif wsLoc == WS_MISC and currentGesture == TILT_UP:
                    # thread.activateGesture.emit(2)
                    self.controller.incrementOrientation(direction="up")
                elif wsLoc == WS_MISC and currentGesture == TILT_DOWN:
                    # thread.activateGesture.emit(2)
                    self.controller.incrementOrientation(direction="down")
                # elif currentGesture == STOP:
                    # thread.activateGesture.emit(5)
                # else:
                    # thread.activateGesture.emit(-1)

        except Exception as e:
            print(f"{str(e)}")
            # print("\nExiting..")
            self.handTracker.endStream() # Remember to end the stream
            self.controller.endController()
            if thread is not None:
                thread.quit()


if __name__ == "__main__":
    fsm = FSM()
    fsm.run()