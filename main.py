import cv2
import mediapipe as mp
import math
import numpy as np

def draw_points(image, points):
    for point in points:
        cv2.circle(image, point, 5, (0, 0, 0), -1)


def check_bounding_box(point, box):
    if point[0] > box[0][0] and point[0] < box[1][0] and point[1] > box[0][1] and point[1] < box[1][1]:
        return True
    return False

def euclidean_distance(point, ref_point):
    return math.sqrt((point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2)


wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)
hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_landmarks = mp.solutions.hands.HandLandmark
points = []
CANVAS_BOUNDING_BOX = ((int(wCam - wCam *2 / 3), 0), (wCam, int(hCam * 2 / 3)))
STATES = {"IDLE": 0, "PAINTING": 1}
state = STATES["IDLE"]
while cam.isOpened():
    success, image = cam.read()
    image = cv2.flip(image, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.rectangle(image, CANVAS_BOUNDING_BOX[0], CANVAS_BOUNDING_BOX[1], (0, 255, 0), 2)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        WRIST = hand_landmarks[mp_landmarks.WRIST].x, hand_landmarks[mp_landmarks.WRIST].y
        INDEX_FINGER_TIP = hand_landmarks[mp_landmarks.INDEX_FINGER_TIP].x , hand_landmarks[mp_landmarks.INDEX_FINGER_TIP].y 
        MIDDLE_FINGER_TIP = hand_landmarks[mp_landmarks.MIDDLE_FINGER_TIP].x, hand_landmarks[mp_landmarks.MIDDLE_FINGER_TIP].y 
        RING_FINGER_TIP = hand_landmarks[mp_landmarks.RING_FINGER_TIP].x , hand_landmarks[mp_landmarks.RING_FINGER_TIP].y 
        PINKY_TIP = hand_landmarks[mp_landmarks.PINKY_TIP].x, hand_landmarks[mp_landmarks.PINKY_TIP].y 
        THUMB_TIP = hand_landmarks[mp_landmarks.THUMB_TIP].x, hand_landmarks[mp_landmarks.THUMB_TIP].y 
        Finger_points = [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
        distances = [euclidean_distance(point, WRIST) for point in Finger_points]
        max_distance = np.argmax(np.array(distances))
        if len(distances) != 5: 
            continue
        if max_distance == 1 and INDEX_FINGER_TIP[1] < WRIST[1] and distances[1] > (1.5 * distances[2]):
            state = STATES["PAINTING"]
        else:
            state = STATES["IDLE"]
        if state == STATES["PAINTING"]:
            new_point = (int(INDEX_FINGER_TIP[0] * wCam), int(INDEX_FINGER_TIP[1] * hCam))
            if check_bounding_box(new_point, CANVAS_BOUNDING_BOX):
                points.append(new_point)
    draw_points(image, points)
    cv2.imshow('handDetector', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cam.release()
