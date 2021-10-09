'''
* Updated regularly with (up to) 21 landmarks
* Should calculate which fingers are straight and closed (probabilities?)
* Should have a sliding window of x (10?) measurements to increase stability.
* If there is a sudden jump in depth, do not save current depth, and use an average
    of the window depths instead.
* Should decide hand gesture based on finger poses. Probabilities?
* If no hand gesture can be determined, choose either the stop gesture, or the previous one.

* TODO: Figure out if the measurements need filtering. PDAF?
* TODO: Grip functionality is a little more complex than what I first imagined. Fix?

'''

import utils
import datetime
import numpy as np


class HandModel():
    '''
    Fingers: [thumb, index, middle, ring, pinky]
    Gestures:
        0:  Stop moving
                [1, 1, 1, 1, 1]
        1:  Grip
                [1, 1, 0, 0, 0]
                * Thumb and index touching
        2:  Release grip
                [1, 1, 0, 0, 0]
                * Thumb and index not touching
        3:  Turn in the xy plane
                [1, 0, 0, 0, 0]
        4:  Move in xz plane
                [0, 0, 0, 0, 0]
        5:  Move to standard pose
                [0, 1, 0, 0, 0]
    '''
    def __init__(self, type):
        self.fingerAngles = {}
        self.slidingWindow = []
        self.windowSize = 10
        self.openFingers = np.ones(5)

    def addMeasurement(self, landmarks):
        if landmarks != {}:
            self.slidingWindow.append(landmarks)
            if len(self.slidingWindow) > self.windowSize:
                self.slidingWindow.pop(0)
            
            self.calculateFingerAngles()
            self.estimateGesture()

    def calculateFingerAngles(self):
        '''
        Calculates theta and beta angles for all links in each finger.
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

        # # Debugging
        # import time
        # time.sleep(0.2)
        # a = np.array(self.fingerAngles[0])*180/np.pi
        # b = np.array(self.fingerAngles[2])*180/np.pi
        # # print(f"Thumb angles: {a[:,0]}")
        # print(f"Middle finger angles: {b[:,0]}")


    def calculateTransformation(self, theta, beta, translation):
        '''
        Calculates transformation matrix from one finger joint (i-1) to joint (i)

        In:
            theta:  (Float) radians, rotation along z
            beta:   (Float) radians, rotation along y
            translation:    (3x1 Array(Float)) translation from (i-1) to (i)
                            in (i-1) coordinate system

        Out:
            H:  (4x4 Array(Float))
        '''
        R_z = utils.zRotToMat(theta)
        R_y = utils.yRotToMat(beta)

        R = R_z @ R_y

        H_i_prev = utils.transformationMatrix(R, translation)

        # Inverse the calculated matrix, so that the transform "works in the direction wrist -> finger tip"
        H = np.linalg.inv(H_i_prev)

        return H

    def estimateGesture(self):
        for idx in range(5):
            finger = self.fingerAngles[idx]
            # Skipping first finger angle for now
            angles = np.array(finger[1:]).flatten()
            #  TODO: Optimize threshold. Different threshold for thumb?
            #  Differ bewteen positive and negative angles?
            threshold = np.deg2rad(25)
            if np.any(np.abs(angles) > threshold):
                self.openFingers[idx] = 0
            else:
                self.openFingers[idx] = 1
        print(f"Open fingers: {self.openFingers} \r", end="")

    def getFingerAngles(self):
        '''
        Out:
            (Dict = "0"-"4": 1x2 Array(Float))
        '''
        return self.fingerAngles

    def log(self, measurement):
        timestamp = datetime.datetime.now()
        # TODO