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

def euclidean_distance(point, ref_point):
    return math.sqrt((point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2)

wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)
gesture_recognizer = GestureDetectorLogger(video_mode=False)
hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands

points = []
CANVAS_BOUNDING_BOX = ((int(wCam - wCam *2 / 3), 0), (wCam, int(hCam * 2 / 3)))
STATES = {"IDLE": 0, "PAINTING": 1}
state = STATES["IDLE"]
while cam.isOpened():
    success, image = cam.read()
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)

    if not results.multi_hand_landmarks:
      continue
   
    hand_landmarks = results.multi_hand_landmarks[0]
    WRIST = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * wCam), int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * hCam)
    INDEX_FINGER_TIP = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * wCam), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * hCam)
    MIDDLE_FINGER_TIP = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * wCam), int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * hCam)
    RING_FINGER_TIP = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * wCam), int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * hCam)
    PINKY_TIP = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * wCam), int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * hCam)
    THUMB_TIP = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * wCam), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * hCam)
    
    cv2.circle(image, center=WRIST, radius=10, color=(255, 0, 0), thickness=-1)
    cv2.circle(image, center=INDEX_FINGER_TIP, radius=3, color=(255, 0, 0), thickness=-1)
    cv2.circle(image, center=MIDDLE_FINGER_TIP, radius=3, color=(255, 255, 0), thickness=-1)
    cv2.circle(image, center=RING_FINGER_TIP, radius=3, color=(255, 0, 0), thickness=-1)
    cv2.circle(image, center=PINKY_TIP, radius=3, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, center=THUMB_TIP, radius=5, color=(0, 0, 0), thickness=-1)
    
    WRIST = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    INDEX_FINGER_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x , hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y 
    MIDDLE_FINGER_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y 
    RING_FINGER_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x , hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y 
    PINKY_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y 
    THUMB_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y 
    
    Finger_points = [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
    distances = [euclidean_distance(point, WRIST) for point in Finger_points]
    max_distance = np.argmax(np.array(distances))

    cv2.rectangle(image, CANVAS_BOUNDING_BOX[0], CANVAS_BOUNDING_BOX[1], (0, 255, 0), 2)
    if len(distances) != 5: 
        continue
    if max_distance == 1 and INDEX_FINGER_TIP[1] < WRIST[1] and distances[1] > (2 * distances[2]):
        print("yes")
        state = STATES["PAINTING"]
    else:
        print("no")
        state = STATES["IDLE"]
    if state == STATES["PAINTING"]:
        new_point = (int(INDEX_FINGER_TIP[0] * wCam), int(INDEX_FINGER_TIP[1] * hCam))
        if check_bounding_box(new_point, CANVAS_BOUNDING_BOX):
            points.append(new_point)
            cv2.circle(image, new_point, 5, (0, 0, 0), -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    draw_points(image, points)
    cv2.imshow('handDetector', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cam.release()
