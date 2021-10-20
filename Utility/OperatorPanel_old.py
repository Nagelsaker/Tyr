import cv2
import numpy as np
import mediapipe as mp
from skimage import data, color, io, img_as_float

def visualize(results, image, workspaceOverlay):
    '''
    TEMP
    Draws the detected landmarks of a human hand on the video feed,
    and displays it.

    In:
        results: TODO
        image: array
    '''
    # Draw the hand annotations on the image.
    image = image.astype("uint8")
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    workspaceOverlay = workspaceOverlay.astype("uint8")
    workspaceOverlay.flags.writeable = True
    workspaceOverlay = cv2.cvtColor(workspaceOverlay, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(workspaceOverlay, 0.5, image, 0.5, 0, image)

    mpDrawing = mp.solutions.drawing_utils
    mpDrawingStyles = mp.solutions.drawing_styles
    mpHands = mp.solutions.hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(
                image,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS,
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style())


    # Show stream
    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        return