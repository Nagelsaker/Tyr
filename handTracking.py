import cv2
import glob
from RealSenseCam import cameraStream
import mediapipe as mp
from pathlib import Path



class HandTracking:
    def __init__(self):
        self.mpDrawing = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpHands = mp.solutions.hands

    def trackStream(self):
        camStream = cameraStream()
        camStream.start()
        # TODO


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

if __name__ == "__main__":
    handTracker = HandTracking()
    handTracker.trackStaticImgs("dataset_1", saveAnnotations=True)