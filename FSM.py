import json
import csv
import time
import numpy as np
from datetime import date
from Hand.HandTracking import HandTracking
from Utility.utils import drawLandmarks, loadWorkspace, generateFilename
from Comms.Controller import Controller, Obstacle
from PyQt5.QtGui import QImage
from Hand.HandModel import *



class FSM():
    '''
    A class representing a finite state machine

    The FSM swicthes between different control states for the Robotis
    OpenManipulator, based on the input signals from an operator.
    For further information on the inpu signals, see the documentation
    on HandModel and HandTracking.

    Attributes
    ----------
        depthRange : 2x1 Array(Float)
            the range from min to max, which yields max to min manipulator movement
        pathTime: Float
            time it takes for the manipulator to move to desired position
        imgWidth: int
            image width
        imgHeight: int
            image height
        camSN: str
            serial number for the camera recording operators hand
        Kp_default: (5x1) Array(Float)
            controller parameters for the velocity controllers
        handTracker: HandTracking
            hand tracking object
        hm: HandModel
            hand model object
        controller: Controller
            controller object
        obstacles: (5x1) Array(Obstacle)
        imgLM: Array
            current workspace image with hand landmarks drawn
    '''
    STATE = ST_STOP

    def __init__(self):
        f = open("settings.json")
        settings = json.load(f)
        # Long range: [0.60, 0.85], Short range: [0.30, 0.59]
        self.depthRange = settings["depthRange"]
        self.pathTime = settings["pathTime"] # 0.2
        self.imgWidth = settings["imgWidth"]
        self.imgHeight = settings["imgHeight"]
        self.camSN = settings["camSN"]
        useDepth = settings["useDepth"] == 1
        # Kp=[K_p_beta, K_p_r, K_p_z, K_p_theta, K_p_phi]
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
        self.hm = HandModel("left", self.workspaceSections, wristAngle_threshold, thumbAngle_threshold, fingerAngle_threshold, useDepth)
        self.controller = Controller(self.imgWidth, self.imgHeight, Kp=self.Kp_default, pathTime=self.pathTime, obstacles=self.obstacles)
        self.imgLM = None

        # Setup logs
        self.pathToLog = settings["pathToLog"]
        self.writeLogs = settings["writeLogs"] == 1
        self.pathToLogPose = ""
        self.pathToLogFSMState = ""
        self.pathToLogHandPoints = ""
        self.pathToLogTimeSteps = ""
        self.curDate = date.today().strftime("%Y_%m_%d")
        self.curTime = 0
        self.startTime = time.time()
        if self.writeLogs:
            self.pathToLogPose = generateFilename(self.pathToLog, "csv", f"{self.curDate}_RobotPoseLogger")
            self.pathToLogFSMState = generateFilename(self.pathToLog, "csv", f"{self.curDate}_FSMStateLogger")
            self.pathToLogHandPoints = generateFilename(self.pathToLog, "csv", f"{self.curDate}_HandPointsLogger")
            self.pathToLogTimeSteps = generateFilename(self.pathToLog, "csv", f"{self.curDate}_TimeStepLogger")

    def setWristThreshold(self, threshold):
        '''
        In:
            threshold: (2x1) Array(float)
        '''
        self.hm.setWristThreshold(threshold)

    def setFingerThreshold(self, threshold):
        '''
        In:
            threshold: (float)
        '''
        self.hm.setFingerThreshold(threshold)

    def setThumbThreshold(self, threshold):
        '''
        In:
            threshold: (2x1) Array(float)
        '''
        self.hm.setThumbThreshold(threshold)

    def getCurrentImage(self):
        '''
        Out:
            (2x1) (1080x1920x3) Array(float)
        '''
        return self.imgLM
    
    def run(self, thread=None):
        '''
        In:
            thread: (QThread) object
        '''
        self.controller.updateRobotPose(updateX=True, updateY=True, updateZ=True)
        self.handTracker.startStream()

        try:
            while True: # Tracking loop
                handPoints, image, results = self.handTracker.getLiveLandamarks()

                if thread is not None:
                    if thread.isInterruptionRequested():
                        raise Exception

                    self.imgLM =  drawLandmarks(results, image, self.workspaceOverlay, thread)
                    h, w, ch = self.imgLM.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(self.imgLM.data, w, h, bytesPerLine, QImage.Format_RGB888).scaled(thread.w, thread.h)
                    thread.changePixmap.emit(convertToQtFormat)
                
                self.hm.addMeasurement(handPoints)
                depth,_ = self.hm.getHandDepth()
                
                self.controller.updateRobotPose()
                currentGesture = self.hm.getCurrentGesture()
                wsLoc = self.hm.getWorkspaceLocation(self.imgHeight, self.imgWidth)

                if len(handPoints) != 0 and thread is not None:
                    # Send Current gesture to GUI application
                    thread.activateGesture.emit(currentGesture)
                    thread.updateSkeleton.emit(handPoints[0])
                    thread.setDepthValue.emit(depth*100)

                # FSM
                usePrecision = (currentGesture==PRECISION)
                

                if wsLoc == WS_TURN_LEFT and currentGesture != STOP:
                    self.STATE = ST_TURN_LEFT
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.turnHorizontally(direction="left", precision=usePrecision)
                elif wsLoc == WS_TURN_RIGHT and currentGesture != STOP:
                    self.STATE = ST_TURN_RIGHT
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.turnHorizontally(direction="right", precision=usePrecision)
                elif wsLoc == WS_MOVE_FORWARD and currentGesture != STOP:
                    self.STATE = ST_MOVE_FORWARD
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.incrementRadius(direction="forward", precision=usePrecision)
                elif wsLoc == WS_MOVE_BACKWARD and currentGesture != STOP:
                    self.STATE = ST_MOVE_BACKWARD
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.incrementRadius(direction="backward", precision=usePrecision)
                elif wsLoc == WS_MISC and currentGesture == GRIP:
                    self.STATE = ST_GRIP
                    self.controller.incrementGripper(direction="close")
                elif wsLoc == WS_MISC and currentGesture == UNGRIP:
                    self.STATE = ST_UNGRIP
                    self.controller.incrementGripper(direction="open")
                elif wsLoc == WS_MISC and currentGesture == MOVE_HEIGHT:
                    self.STATE = ST_HEIGHT
                    self.controller.updateRobotPose(updateZ=True)
                    self.controller.incrementHeight(depth=self.hm.getHandDepthSensor(), range=self.depthRange)
                elif wsLoc == WS_MISC and currentGesture == TILT_UP:
                    self.STATE = ST_TILT_UP
                    self.controller.incrementOrientation(direction="up")
                elif wsLoc == WS_MISC and currentGesture == TILT_DOWN:
                    self.STATE = ST_TILT_DOWN
                    self.controller.incrementOrientation(direction="down")
                else:
                    self.STATE = ST_STOP
                
                # Logging
                if self.writeLogs:
                    pose = self.controller.getPose()
                    self.log(pose, self.STATE, handPoints)


        except Exception as e:
            print(f"{str(e)}")
            self.handTracker.endStream() # Remember to end the stream
            self.controller.endController()
            if thread is not None:
                thread.quit()

    def log(self, pose=None, state=None, handPoints=None):
        self.logRobotPose(pose)
        self.logFSMState(state)
        self.logHandPoints(handPoints)
        t = time.time() - self.startTime
        self.logTimeSteps(t)

    def logRobotPose(self, pose):
        with open(f"{self.pathToLogPose}", "a") as fp:
            writer = csv.writer(fp)

            if pose is None:
                writer.writerow(["None"])
                return

            formattedPose = []
            for key, value in pose.items():
                formattedPose.append(key)
                for _, val2 in value.items():
                    formattedPose.append(val2)
            writer.writerow(formattedPose)

    def logFSMState(self, state):
        with open(f"{self.pathToLogFSMState}", "a") as fp:
            writer = csv.writer(fp)

            if state is None:
                writer.writerow(["None"])
                return
            
            writer.writerow(str(state))


    def logHandPoints(self, handPoints):
        with open(f"{self.pathToLogHandPoints}", "a") as fp:
            writer = csv.writer(fp)

            if handPoints == {}:
                writer.writerow(["None"])
                return
            
            formattedHP = []
            for i in range(len(handPoints[0])):
                formattedHP.append(i)
                for _, val in handPoints[0][i].items():
                    formattedHP.append(val)
            
            writer.writerow(formattedHP)
    
    def logTimeSteps(self, t):
        with open(f"{self.pathToLogTimeSteps}", "a") as fp:
            writer = csv.writer(fp)

            if t is None:
                writer.writerow(["None"])
                return
            
            writer.writerow([str(t)])


if __name__ == "__main__":
    fsm = FSM()
    fsm.run()