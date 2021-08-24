import cv2
import mediapipe as mp
# from mediapipe.python.solutions.hands import HandLandmark



def handtracking():
    mpDrawing = mp.solutions.drawing_utils
    mpDrawingStyles = mp.solutions.drawing_styles
    mpHands = mp.solutions.hands

    # For static images
    # imageFiles = ["imgs/1.jpg", "imgs/2.jpg", "imgs/3.jpg", "imgs/4.jpg", "imgs/5.jpg"]
    imageFiles = ["imgs/6.jpg"]
    with mpHands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(imageFiles):
            # image = cv2.flip(cv2.imread(file),1)
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
                    f"{handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * imageWidth},"
                    f"{handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * imageHeight})"
                )

                mpDrawing.draw_landmarks(
                    annotatedImage,
                    handLandmarks,
                    mpHands.HAND_CONNECTIONS,
                    mpDrawingStyles.get_default_hand_landmarks_style(),
                    mpDrawingStyles.get_default_hand_connections_style())
                # cv2.imwrite(
                #     "C:/Users/sjlexau/git/Tyr/imgs/annotatedImage" + str(idx) + ".png", cv2.flip(annotatedImage, 1))
                cv2.imwrite(
                    "C:/Users/sjlexau/git/Tyr/imgs/annotatedImage" + str(idx) + ".png", annotatedImage)


if __name__ == "__main__":
    handtracking()