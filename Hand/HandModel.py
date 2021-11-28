from Utility.constants import *
import Utility.utils as utils
import numpy as np

class HandModel():
    '''
    HandModel class

    Models a human left hand. Is easily expanded to include right hands and multiple hands.
    For now, this functionality is not needed.

    Note: the last value from the windows are always used, as the system is stable
    without averaging over all entries. The window structure remains, if future work
    should deem it necessary to implement averaging.

    Attributes
    ----------
        type: Str
            Currently only supports 'left hand' type
        workspace: Dict: (1920x1080) Array(Bool)
            dictionary over all workspace sections: misc, turn left/right
            and move forward/backward
        wristAngle_threshold: (2x1) Array(Float)
        thumbAngle_threshold: (2x1) Array(Float)
        fingerAngle_threshold: (Float)
        useDepth: Bool
            if True, use measured depth to acquire angles and hand position
            if False, use synthetic data
        fingerAngles: Dict
        slidingWindow: (10x1) Array
            Can be used to choose current gesture from an average
            Contains Landmarks
        depthWindow: (10x2) Array
            Contains palm depth (median) from measured landmark depths and variance
        windowSize: Int
            size of depthWindow and slidingWindow
        openFingers: (5x1) Array(Int)
            array of integers representing closed (0) and open (1) fingers
        gesture: Int
            current detected gesture
        acceptedDepthVar: Float
            If the variance in depth is too large, the measurement should not be trusted
    '''
    def __init__(self, type, workspace, wristAngle_threshold, thumbAngle_threshold, fingerAngle_threshold, useDepth):
        self.type = type
        self.workspace = workspace
        self.wristAngle_threshold = wristAngle_threshold # Default for long [-15, 15]
        self.thumbAngle_threshold = thumbAngle_threshold # Default for long [-15, -15]
        self.fingerAngle_threshold = fingerAngle_threshold # Default for long 25
        self.useDepth = useDepth

        self.fingerAngles = {}
        self.slidingWindow = []
        self.depthWindow = []
        self.windowSize = 10
        self.openFingers = np.ones(5)
        self.gesture = -1
        self.acceptedDepthVar = 0.03

    def setWristThreshold(self, threshold):
        '''
        In:
            threshold: (2x1) Array(float)
        '''
        self.wristAngle_threshold = threshold

    def setFingerThreshold(self, threshold):
        '''
        In:
            threshold: (float)
        '''
        self.fingerAngle_threshold = threshold

    def setThumbThreshold(self, threshold):
        '''
        In:
            threshold: (2x1) Array(float)
        '''
        self.thumbAngle_threshold = threshold

    def addMeasurement(self, landmarks):
        '''
        In:
            landmarks: Dict: (List(Dict: Float))
        '''
        if landmarks != {}:
            self.slidingWindow.append(landmarks)
            if len(self.slidingWindow) > self.windowSize:
                self.slidingWindow.pop(0)
            
            self.calculateFingerAngles()
            self.estimateGesture()

    def getPalmLocation(self):
        '''
        Function which estimates the palm location in the image frame.

        Estimates the palm location by averaging over the positions of points
        0, 1, 5, 9, 13, and 17. Visit MediaPipe Hand section for further
        information on landmark positions

        Out:
            (3x1) Array(Float)
        '''
        try:
            latestLandmarks = self.slidingWindow[-1]
        except Exception:
            print("Error getting palm location: No hand detected yet!")
            return [1/2, 1/2, 0]
        x = np.array([latestLandmarks[0][0]['X'], latestLandmarks[0][1]['X'], latestLandmarks[0][5]['X'],
                    latestLandmarks[0][9]['X'], latestLandmarks[0][13]['X'], latestLandmarks[0][17]['X']]).mean()
        y = np.array([latestLandmarks[0][0]['Y'], latestLandmarks[0][1]['Y'], latestLandmarks[0][5]['Y'],
                    latestLandmarks[0][9]['Y'], latestLandmarks[0][13]['Y'], latestLandmarks[0][17]['Y']]).mean()
        z = np.array([latestLandmarks[0][0]['Z'], latestLandmarks[0][1]['Z'], latestLandmarks[0][5]['Z'],
                    latestLandmarks[0][9]['Z'], latestLandmarks[0][13]['Z'], latestLandmarks[0][17]['Z']]).mean()
        return np.array([x, y, z])

    def getHandDepth(self, overrideDepthSetting=False):
        '''
        Function which estimates the measured hand depth.

        The estimate is the median of the measured depth at the image
        position of all 21 landmarks. The median reduces the effects from outliers,
        which may occure when a finger is misdetected, causing its depth to be that
        of the background.

        In:
            overrideDepthSetting: Bool
                if True allways use measured depth over synthetic
        Out:
            estDepth: Float
            var: Float
        '''
        try:
            latestLandmarks = self.slidingWindow[-1]
        except Exception:
            print("Error getting hand depth: No hand detected yet!")
            return 0, 0
        depthSensor = "Depth" if self.useDepth else "Z"

        if overrideDepthSetting: depthSensor = "Depth"

        depths = np.array([round(latestLandmarks[0][i][depthSensor], 3) for i in range(len(latestLandmarks[0]))])
        var = np.var(depths)
        estDepth = np.median(depths) # Reduces the effect from outliers

        if var < self.acceptedDepthVar or len(self.depthWindow) == 0:
            self.depthWindow.append([estDepth, var])
            if len(self.depthWindow) > self.windowSize:
                self.depthWindow.pop(0)
        else:
            estDepth = np.mean(np.array(self.depthWindow)[:,0])
            var = np.var(np.array(self.depthWindow)[:,0])
        # print(f"Estimator :\t{estDepth:.3f}\t\t Mean: {mean:.3f}\t\tVar: \t{var:.3f} \r", end="")
        return estDepth, var

    def getHandDepthSensor(self):
        '''
        Function which returns the depth of the hand

        Out:
            depth: Float
        '''
        depth,_ = self.getHandDepth(overrideDepthSetting=True)
        return depth

    def getWorkspaceLocation(self, imgHeight, imgWidth):
        '''
        Returns the workspace section in which the hand is located

        Workspace sections could be any of the following:
        WS_MOVE_FORWARD
        WS_MOVE_BACKWARD
        WS_TURN_LEFT
        WS_TURN_RIGHT
        WS_MISC

        Out: Int
        '''
        palmPos = np.array([int(self.getPalmLocation()[0]*imgWidth), int(self.getPalmLocation()[1]*imgHeight)])

        if self.workspace["MoveBackward"][palmPos[0]][palmPos[1]] == True:
            return WS_MOVE_BACKWARD
        elif self.workspace["TurnLeft"][palmPos[0]][palmPos[1]] == True:
            return WS_TURN_LEFT
        elif self.workspace["TurnRight"][palmPos[0]][palmPos[1]] == True:
            return WS_TURN_RIGHT
        elif self.workspace["Misc"][palmPos[0]][palmPos[1]] == True:
            return WS_MISC
        elif self.workspace["MoveForward"][palmPos[0]][palmPos[1]] == True:
            return WS_MOVE_FORWARD
        

    def getFingerAngles(self):
        '''
        Function which returns finger angles

        Out:
            (Dict = "0"-"4": (1x2) Array(Float))
        '''
        return self.fingerAngles

    def getCurrentGesture(self):
        '''
        Function which returns the current gesture
        '''
        return self.gesture

    def calculateFingerAngles(self):
        '''
        Function which calculates theta and beta angles for all links in each finger.

        Theta represents the angle between link (i-1) and (i) in the xy plane,
        while beta represents the angle in the xz plane. Keep in mind that
        new rotations are multiplied from the right according to 'current frames',
        the alternative would be multiplying to the left w.r.t the fixed initial axis.
        '''
        # Calculate transformation from world to wrist point (0)
        # x_ij is x value of a point i in the coordinate system j.
        # X_ij is the homogeneous point i in the coordinate system j
        latestLandmarks = self.slidingWindow[-1]
        x_0_w = latestLandmarks[0][0]['X']
        y_0_w = latestLandmarks[0][0]['Y']
        z_0_w = latestLandmarks[0][0]['Z']
        X_0_w = np.array([x_0_w, y_0_w, z_0_w, 1])
        rho_0 = np.sqrt(x_0_w**2 + y_0_w**2)
        theta_0 = np.arctan2(y_0_w, x_0_w)
        beta_0 = 0

        t_0 = np.array([X_0_w[0], X_0_w[1], X_0_w[2]])
        # A transformation matrix H_ij transforms points in coordinate system i to j.
        H_w_0 = self.calculateTransformation(theta_0, beta_0, t_0)

        for i in range(1, 20, 4):
            angles = []
            joint1 = i
            joint2 = i+1
            joint3 = i+2
            joint4 = i+3

            # Joint 1
            X_1_w = np.array([latestLandmarks[0][joint1]['X'],
                             latestLandmarks[0][joint1]['Y'],
                             latestLandmarks[0][joint1]['Z'],
                             1])
            X_1_0 = H_w_0 @ X_1_w
            rho_1 = np.linalg.norm(X_1_0[:3])
            d_1 = X_1_0[2]
            a_1 = np.sqrt(X_1_0[0]**2 + X_1_0[1]**2)
            theta_1 = np.arctan2(X_1_0[1], X_1_0[0]) # Use arctan2 to get angles from all quadrants
            beta_1 = -np.arctan2(d_1, a_1)
            
            t_1 = np.array([X_1_0[0], X_1_0[1], X_1_0[2]])
            # Angles between link 0 and link 1 does not give any useful information
            # angles.append([theta_1, beta_1])
            H_0_1 = self.calculateTransformation(theta_1, beta_1, t_1)

            # Joint 2
            X_2_w = np.array([latestLandmarks[0][joint2]['X'],
                             latestLandmarks[0][joint2]['Y'],
                             latestLandmarks[0][joint2]['Z'],
                             1])
            X_2_1 = H_0_1 @ H_w_0 @ X_2_w
            rho_2 = np.linalg.norm(X_2_1[:3])
            d_2 = X_2_1[2]
            a_2 = np.sqrt(X_2_1[0]**2 + X_2_1[1]**2)
            theta_2 = np.arctan2(X_2_1[1], X_2_1[0]) # Use arctan2 to get angles from all quadrants
            beta_2 = -np.arctan2(d_2, a_2)
            
            t_2 = np.array([X_2_1[0], X_2_1[1], X_2_1[2]])
            angles.append([theta_2, beta_2])
            H_1_2 = self.calculateTransformation(theta_2, beta_2, t_2)

            # Joint 3
            X_3_w = np.array([latestLandmarks[0][joint3]['X'],
                             latestLandmarks[0][joint3]['Y'],
                             latestLandmarks[0][joint3]['Z'],
                             1])
            X_3_2 = H_1_2 @ H_0_1 @ H_w_0 @ X_3_w
            rho_3 = np.linalg.norm(X_3_2[:3])
            d_3 = X_3_2[2]
            a_3 = np.sqrt(X_3_2[0]**2 + X_3_2[1]**2)
            theta_3 = np.arctan2(X_3_2[1], X_3_2[0]) # Use arctan2 to get angles from all quadrants
            beta_3 = -np.arctan2(d_3, a_3)
            
            t_3 = np.array([X_3_2[0], X_3_2[1], X_3_2[2]])
            angles.append([theta_3, beta_3])
            H_2_3 = self.calculateTransformation(theta_3, beta_3, t_3)

            # Joint 4
            X_4_w = np.array([latestLandmarks[0][joint4]['X'],
                             latestLandmarks[0][joint4]['Y'],
                             latestLandmarks[0][joint4]['Z'],
                             1])
            X_4_3 = H_2_3 @ H_1_2 @ H_0_1 @ H_w_0 @ X_4_w
            rho_4 = np.linalg.norm(X_4_3[:3])
            d_4 = X_4_3[2]
            a_4 = np.sqrt(X_4_3[0]**2 + X_4_3[1]**2)
            theta_4 = np.arctan2(X_4_3[1], X_4_3[0]) # Use arctan2 to get angles from all quadrants
            beta_4 = -np.arctan2(d_4, a_4)
            
            t_4 = np.array([X_4_3[0], X_4_3[1], X_4_3[2]])
            angles.append([theta_4, beta_4])
            H_3_4 = self.calculateTransformation(theta_4, beta_4, t_4)

            self.fingerAngles[int((i-1)/4)] = angles


    def calculateTransformation(self, theta, beta, translation):
        '''
        Calculates transformation matrix from one finger joint (i-1) to joint (i)

        In:
            theta: (Float)
                radians, rotation along z
            beta: (Float)
                radians, rotation along y
            translation: (3x1 Array(Float))
                translation from (i-1) to (i)
                in (i-1) coordinate system
        Out:
            H: (4x4) Array(Float)
        '''
        R_z = utils.zRotToMat(theta)
        R_y = utils.yRotToMat(beta)

        R = R_z @ R_y

        H_i_prev = utils.transformationMatrix(R, translation)

        # Inverse the calculated matrix, so that the transform "works in the general direction wrist -> finger tip"
        H = np.linalg.inv(H_i_prev)

        return H


    def estimateWristAngle(self):
        '''
        Function which estimates the wrist angle

        The angle is defined as the angle in XY plane between the wrist and
        the fingertips.
        '''
        latestLandmarks = self.slidingWindow[-1]
        depthSensor = "Depth" if self.useDepth else "Z"
        wristDepth = latestLandmarks[0][0][depthSensor]
        p0 = np.array([latestLandmarks[0][0]["X"], latestLandmarks[0][0]["Y"], 0])
        p5 = np.array([latestLandmarks[0][5]["X"], latestLandmarks[0][5]["Y"], latestLandmarks[0][5][depthSensor] - wristDepth])
        p9 = np.array([latestLandmarks[0][9]["X"], latestLandmarks[0][9]["Y"], latestLandmarks[0][9][depthSensor] - wristDepth])
        p13 = np.array([latestLandmarks[0][13]["X"], latestLandmarks[0][13]["Y"], latestLandmarks[0][13][depthSensor] - wristDepth])
        p17 = np.array([latestLandmarks[0][17]["X"], latestLandmarks[0][17]["Y"], latestLandmarks[0][17][depthSensor] - wristDepth])
        p_avg = (p5 + p9 + p13 + p17)/4

        dist_xy = np.linalg.norm(p_avg[:2] - p0[:2])
        wristAngle = np.rad2deg(np.arctan2(p_avg[2], dist_xy))
        return wristAngle

    def estimateGesture(self):
        '''
        Function which estimates the current gesture

        The estimate is based on the finger angles, and whether they are
        within given thresholds. Finger statuses are saved to 'openFingers' array.
        Current gesture is saved to 'gesture' and can be any of the following:
        GRIP
        UNGRIP
        PRECISION
        TILT_UP
        TILT_DOWN
        MOVE_HEIGHT
        STOP
        '''
        # Thumb
        finger = self.fingerAngles[0]
        angles = np.array(self.fingerAngles[0][1:])[:,0]
        threshold = np.deg2rad(self.thumbAngle_threshold)
        if np.all(angles > threshold):
            self.openFingers[0] = 1
        else:
            self.openFingers[0] = 0

        for idx in range(1, 5):
            finger = self.fingerAngles[idx]
            # Skipping first finger angle for now
            angles = np.array(finger[1:]).flatten()
            threshold = np.deg2rad(self.fingerAngle_threshold)
            if np.any(np.abs(angles) > threshold):
                self.openFingers[idx] = 0
            else:
                self.openFingers[idx] = 1
        
        wristAngle = self.estimateWristAngle()

        if all(self.openFingers == 0):
            # All fingers closed
            self.gesture = GRIP
        elif all(self.openFingers == 1):
            self.gesture = UNGRIP
        elif all(self.openFingers[:4] == 0) and self.openFingers[4] == 1:
            # Pinky finger open
            self.gesture = PRECISION
        elif self.openFingers[0] == 0 and all(self.openFingers[1:] == 1):
            # Thumb closed
            pass
            if wristAngle < self.wristAngle_threshold[0]:
                self.gesture = TILT_UP
            elif wristAngle > self.wristAngle_threshold[1]:
                self.gesture = TILT_DOWN
            else:
                self.gesture = -1
        elif self.openFingers[0] == 0 and self.openFingers[1] == 1 and all(self.openFingers[2:] == 0):
            # Index finger open
            self.gesture = MOVE_HEIGHT
        elif self.openFingers[0] == 1 and all(self.openFingers[1:] == 0):
            self.gesture = STOP
        else:
            self.gesture = -1