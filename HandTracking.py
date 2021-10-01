from SimpleController import SimpleController
import cv2
import glob
import rclpy
import numpy as np
import mediapipe as mp
from pathlib import Path
from RealSenseCam import CameraStream
from Communication import SetPositionClient, PoseSubscriber



class HandTracking:
    def __init__(self):
        self.mpDrawing = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpHands = mp.solutions.hands

    def startTrackingFromStream(self):
        self.camStream = CameraStream()
        self.camStream.start()
    
    def endStream(self):
        self.camStream.end()

    def getLiveLandamarks(self, visualize=True):
        with self.mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            handPoints = {}
            handIdx = 0
            handDepth = -1
            estHandPosition = -1

            if results.multi_hand_landmarks:
                for handLandmarks in results.multi_hand_landmarks:
                    keypoints = []
                    for data_point in handLandmarks.landmark:
                        keypoints.append({
                            'X': data_point.x,
                            'Y': data_point.y,
                            'Z': data_point.z,
                            'Visibility': data_point.visibility,
                            })
                    handPoints[handIdx] = keypoints
                    handIdx += 1

                estHandPosition = np.array([
                    (handPoints[0][0]["X"] + (handPoints[0][5]["X"] - handPoints[0][0]["X"])/2 ) * imgWidth,
                    (handPoints[0][0]["Y"] + (handPoints[0][5]["Y"] - handPoints[0][0]["Y"])/2 ) * imgHeight
                    ])
                handDepth = depthImage[int(estHandPosition[1]), int(estHandPosition[0]), 0]
                # print(f"Wrist depth: {handDepth} \tPixels: {estHandPosition}")

            if visualize:
                self.drawLandmarks(results, image)

            return handDepth, estHandPosition, handPoints, image


    def drawLandmarks(self, results, image):
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

            
    def visualizeLiveTracking(self):
        camStream = CameraStream()
        camStream.start()
        
        with self.mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while True: # Stream loop
                images = camStream.getAlignedImages(clippingDistanceInMeters=1)
                if np.all(images == -1):
                    continue # Empty frames
                
                imgHeight = images.shape[0]
                imgWidth = int(images.shape[1] / 2)

                colorImage = images[:, :imgWidth, :]
                depthImage = images[:, imgWidth:, :]

                # Convert the BGR image to RGB.
                image = cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)

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

                cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    def trackStaticImgs(self, dataDir, saveAnnotations=False):
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

    handTracker.startTrackingFromStream()
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
            positionClient.send_request(pose, pathTime)
            print(f"Hand depth: {handDepth}, Robot depth: {pos}")
    except KeyboardInterrupt:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        poseSubscriber.destroy_node()
        rclpy.shutdown()

def main():
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

    handTracker.startTrackingFromStream()
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

            positionClient.send_request(newPose, pathTime)
            prevPose = newPose
            print(f"Hand depth: {handDepth}")
    except KeyboardInterrupt:
        print("\nExiting..")
        handTracker.endStream() # Remember to end the stream
        poseSubscriber.destroy_node()
        rclpy.shutdown()




if __name__ == "__main__":
    depthDemonstration()
    # main()