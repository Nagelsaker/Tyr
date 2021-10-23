import cv2
import glob
import rclpy
import numpy as np
import mediapipe as mp
from pathlib import Path
from Hand.HandModel import HandModel
from Comms.SimpleController import SimpleController
from Comms.RealSenseCam import CameraStream
from Comms.Communication import SetPositionClient, PoseSubscriber



class HandTracking:
    def __init__(self, camSN):
        '''
        Module that tracks hands with the MediaPipe API. Also includes depth information
        of detections, produced by the Intel RealSense Depth Camera D435.
        '''
        self.camSN = camSN
        self.mpDrawing = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpHands = mp.solutions.hands

        self.handPoints = None
        self.image = None
        self.results = None

    def startStream(self):
        '''
        Starts streaming from the stereoscopic sensor
        '''
        self.camStream = CameraStream(self.camSN)
        self.camStream.start()
    
    def endStream(self):
        '''
        Stops the camera sensor stream
        '''
        self.camStream.end()

    def getLiveLandamarks(self, visualize=False):
        '''
        Extracts one image from camera stream and runs hand detection on it.
        
        In:
            visualize: (Bool)
        
        Out:
            handDepth: (Float) Hand depth at wrist point
            estHandPosition: (Array) X and Y position of wrist point in image
            handPoints: (Dict) 21 hand landmarks
            image: (Array) RGB image from depth sensor
        '''
        with self.mpHands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            # images = self.camStream.getAlignedImages(clippingDistanceInMeters=np.array([0.3, 1]))
            images = self.camStream.getAlignedImages(clippingDistanceInMeters=0)
            # images = self.camStream.getAlignedImages()
            if np.all(images == -1):
                return -1 # Empty frames
            
            imgHeight = images.shape[0]
            imgWidth = int(images.shape[1] / 2)

            colorImage = images[:, :imgWidth, :]
            depthImage = images[:, imgWidth:, :]

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(colorImage.astype("uint8"), cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, -1)
            depthImage = cv2.flip(depthImage, -1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            
            # Note that handedness is determined assuming the input image is mirrored,
            # i.e., taken with a front-facing/selfie camera with images flipped horizontally.
            # If it is not the case, please swap the handedness output in the application.
            results = hands.process(image)

            handPoints = {}
            handIdx = 0
            handDepth = -1
            estHandPosition = -1

            if results.multi_hand_landmarks:
                for handLandmarks in results.multi_hand_landmarks:
                    keypoints = []
                    for data_point in handLandmarks.landmark:
                        X = data_point.x
                        Y = data_point.y
                        Z = data_point.z
                        xPx = min([int(X*imgWidth), 1920-1])
                        yPx = min([int(Y*imgHeight), 1080-1])
                        keypoints.append({
                            'X': X,
                            'Y': Y,
                            'Z': Z,
                            'Visibility': data_point.visibility,
                            'Depth' : depthImage[yPx, xPx, 0]
                            })
                    handPoints[handIdx] = keypoints
                    handIdx += 1

            if visualize:
                self.drawLandmarks(results, image)
            
            self.handPoints = handPoints
            self.image = image
            self.results = results

            return handPoints, image, results

    def getCurrentLandmarks(self):
        return self.handPoints, self.image, self.results

    def drawLandmarks(self, results, image):
        '''
        Draws the detected landmarks of a human hand on the video feed,
        and displays it.

        In:
            results: TODO
            image: array
        '''
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mpDrawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mpHands.HAND_CONNECTIONS,
                    self.mpDrawingStyles.get_default_hand_landmarks_style(),
                    self.mpDrawingStyles.get_default_hand_connections_style())
        
        # Show stream
        cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            return


    def trackStaticImgs(self, dataDir, saveAnnotations=False):
        '''
        Fucntion made for debug purposes. Track from stream instead
        '''
        imgFiles = glob.glob(f"{dataDir}/imgs/*.jpg")

        with self.mpHands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
            for idx, file in enumerate(imgFiles):
                image = cv2.imread(file)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                print("Handedness:", results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                imageHeight, imageWidth, _ = image.shape
                annotatedImage = image.copy()
                
                for handLandmarks in results.multi_hand_landmarks:
                    print("handLandmarks:", handLandmarks)
                    print(
                        f"Index finger tip coordinates: (",
                        f"{handLandmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].x * imageWidth},"
                        f"{handLandmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].y * imageHeight})"
                    )

                    self.mpDrawing.draw_landmarks(
                        annotatedImage,
                        handLandmarks,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawingStyles.get_default_hand_landmarks_style(),
                        self.mpDrawingStyles.get_default_hand_connections_style())

                    if saveAnnotations:
                        Path(f"{dataDir}/annotatedImgs").mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(
                            f"{dataDir}/annotatedImgs/" + str(idx) + ".png", annotatedImage)


def depthDemonstration():
    '''
    Demonstration function for controlling the OpenManipulator-X with only depth
    information from the tracking information.
    '''
    rclpy.init()
    handTracker = HandTracking()
    positionClient = SetPositionClient()
    poseSubscriber = PoseSubscriber()
    
    rclpy.spin_once(poseSubscriber) # Update pose
    pose = poseSubscriber.getPose()

    # pose = {"position": {"x" : 0.0, "y" : 0.0, "z" : 0.22},
    #         "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}}

    minDepth = 0.25
    maxDepth = 0.40
    pathTime = 0.2
    maxRobotDepth = 0.30

    minRobotDepth = 0.08
    maxRobotDepth = 0.30

    handTracker.startStream()
    prevHandDepth = minDepth
    try:
        while True: # Tracking loop
            handDepth,_,_,_ = handTracker.getLiveLandamarks()
            if handDepth < minDepth or handDepth > maxDepth:
                handDepth = prevHandDepth
            else:
                prevHandDepth = handDepth

            pos = (maxDepth - handDepth) / (maxDepth - minDepth) * (maxRobotDepth - minRobotDepth) + minRobotDepth
            pose["position"]["z"] = pos
            positionClient.sendRequest(pose, pathTime)
            print(f"Hand depth: {handDepth}, Robot depth: {pos}")
    except KeyboardInterrupt:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        poseSubscriber.destroy_node()
        rclpy.shutdown()



def debug():
    '''
    Debug
    '''
    rclpy.init()
    handTracker = HandTracking()
    positionClient = SetPositionClient()
    poseSubscriber = PoseSubscriber()
    
    rclpy.spin_once(poseSubscriber) # Update pose
    pose = poseSubscriber.getPose()

    # pose = {"position": {"x" : 0.0, "y" : 0.0, "z" : 0.22},
    #         "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}}

    minDepth = 0.25
    maxDepth = 0.40
    pathTime = 0.2
    maxRobotDepth = 0.30

    minRobotDepth = 0.08
    maxRobotDepth = 0.30

    hm = HandModel(type="left")

    handTracker.startStream()
    prevHandDepth = minDepth
    try:
        while True: # Tracking loop
            handPoints, image = handTracker.getLiveLandamarks(visualize=True)
            hm.addMeasurement(handPoints)

    except KeyboardInterrupt:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        poseSubscriber.destroy_node()
        rclpy.shutdown()

def fullControlSimple():
    rclpy.init()
    handTracker = HandTracking()
    positionClient = SetPositionClient()
    poseSubscriber = PoseSubscriber()
    controller = SimpleController(1080, 1920)

    # pose = {"position": {"x" : 0.0, "y" : 0.0, "z" : 0.22},
    #         "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}}

    minDepth = 0.3
    maxDepth = 0.6
    pathTime = 0.25

    handTracker.startStream()
    prevHandDepth = minDepth

    rclpy.spin_once(poseSubscriber) # Update pose
    prevPose = poseSubscriber.getPose()
    z = prevPose["position"]["z"]
    try:
        while True: # Tracking loop
            rclpy.spin_once(poseSubscriber) # Update pose
            currentPose = poseSubscriber.getPose()
            handDepth, handPosition,_,_ = handTracker.getLiveLandamarks()

            if (handDepth == -1 or type(handPosition) != np.ndarray):   continue

            newPose = controller.computeNewPose(currentPose, handDepth, handPosition, prevPose)
            x = newPose["position"]["x"]
            y = newPose["position"]["y"]
            newPose["position"]["z"] = z
            print(f"X: {x}, Y: {y}, Z: {z}")

            positionClient.sendRequest(newPose, pathTime)
            prevPose = newPose
            print(f"Hand depth: {handDepth}")
    except KeyboardInterrupt:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        poseSubscriber.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    # depthDemonstration()
    # main()
    debug()