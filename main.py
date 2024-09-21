import cv2
import mediapipe as mp
import math
import numpy as np
from gesture_detector import GestureDetectorLogger

def draw_points(image, points):
    for point in points:
        cv2.circle(image, point, 5, (0, 0, 0), -1)


def check_bounding_box(point, box):
    if point[0] > box[0][0] and point[0] < box[1][0] and point[1] > box[0][1] and point[1] < box[1][1]:
        return True
    return False


wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)
gesture_recognizer = GestureDetectorLogger(video_mode=False)
hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
points = []
CANVAS_BOUNDING_BOX = ((int(wCam - wCam *2 / 3), 0), (wCam, int(hCam * 2 / 3)))
STATES = {"IDLE": 0, "PAINTING": 1}
state = STATES["IDLE"]
while cam.isOpened():
    success, image = cam.read()
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    [gesture, hand_landmarks] = gesture_recognizer.detect(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, CANVAS_BOUNDING_BOX[0], CANVAS_BOUNDING_BOX[1], (0, 255, 0), 2)
    if gesture and gesture.category_name == 'Pointing_Up':
        state = STATES["PAINTING"]
    else:
        state = STATES["IDLE"]
    if state == STATES["PAINTING"]:
        if len(hand_landmarks) > 0:
            coords = hand_landmarks[0][8]
            new_point = (int(coords.x * wCam), int(coords.y * hCam))
            if check_bounding_box(new_point, CANVAS_BOUNDING_BOX):
                points.append(new_point)
    draw_points(image, points)
    cv2.imshow('handDetector', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cam.release()
