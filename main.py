import os
import cv2
import mediapipe as mp
import math
import numpy as np
from point import Point

wCam, hCam = 640, 480
BRUSH_COLOR = (0, 0, 0)
BRUSH_SIZE = 2
CANVAS_BOUNDING_BOX = ((int(wCam / 3), 0), (wCam, int(hCam * 2 / 3)))
STATES = {"IDLE": 0, "PAINTING": 1}
POINT_INTERVAL = 1
MIN_DISTANCE = 3
NOTIF_TIME = 60
NOTIF_LOCATION = (CANVAS_BOUNDING_BOX[0][0], CANVAS_BOUNDING_BOX[1][1] + 20)
BRUSH_COLORS = {
    'r': (0, 0, 255),  # Red
    'g': (0, 255, 0),  # Green
    'b': (255, 0, 0),  # Blue
    'l': (0, 0, 0),    # Black
}
BRUSH_SIZES = {
    '1': 2,
    '2': 6,
    '3': 12
}

def draw_notification(image, text):
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.6
    thickness = 1
    cv2.putText(image, text, NOTIF_LOCATION, font, fontScale, (36, 36, 190), thickness, cv2.LINE_AA)


def draw_hints(image):
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.6
    thickness = 1
    texts = [
        ["Press keys:", 'l'],
        ["Red: R", 'r'],
        ["Green: G", 'g'],
        ["Blue: B", 'b'],
        ["Black: L", 'l'],
        ["", "l"],
        ["Small Brush: 1", 'l'],
        ["Medium Brush: 2", 'l'],
        ["Large Brush: 3", 'l'],
        ["", "l"],
        ["Erase: E", 'l'],
        ["Save: S", 'l'],
        ["Quit: Q", 'l']
    ]
    for i, text in enumerate(texts):
        cv2.putText(image, text[0], (20, (i + 1) * 25), font, fontScale, BRUSH_COLORS[text[1]], thickness, cv2.LINE_AA)


def draw_points(image, points):
    for i in range(1, len(points)):
        point_color = points[i].get_color()
        point_size = points[i].get_size()
        if points[i].get_type() == Point.POINT_TYPE['END']:
            cv2.circle(image, points[i].get_coords(), point_size, point_color, -1)
            continue
        cv2.line(image, points[i - 1].get_coords(), points[i].get_coords(), point_color, point_size)


def draw_bounding_box():
    cv2.rectangle(image, CANVAS_BOUNDING_BOX[0], CANVAS_BOUNDING_BOX[1], (0, 255, 0), 2)


def check_bounding_box(point, box):
    if point.get_coords()[0] > box[0][0] and point.get_coords()[0] < box[1][0] and point.get_coords()[1] > box[0][1] and point.get_coords()[1] < box[1][1]:
        return True
    return False


def euclidean_distance(point, ref_point):
    return math.sqrt((point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2)


def check_duplicate_point(last_point, point):
    distance = euclidean_distance(last_point, point)
    if distance < MIN_DISTANCE:
        return False
    return True


def last_filename_index():
    files = os.listdir('.')
    last_index = 0
    for filename in files:
        if filename.startswith('canvas'):
            index = filename[6:-4]
            if index != '':
                last_index = int(index)
            else:
                last_index = 0
    return last_index


def save_paint(points):
    canvas = np.ones((hCam, wCam, 3)) * 255
    draw_points(canvas, points)
    filename = 'canvas' + str(last_filename_index() + 1) + '.jpg'
    cv2.imwrite(filename, canvas[:CANVAS_BOUNDING_BOX[1][1], CANVAS_BOUNDING_BOX[0][0]:])


hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_landmarks = mp.solutions.hands.HandLandmark
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)
points = []
state = STATES["IDLE"]
last_point_time = 0
is_end_point = False
point_is_valid = False
notification_last_shown = 0
is_notificatin_showing = False
while cam.isOpened():
    if is_notificatin_showing:
        if notification_last_shown >= NOTIF_TIME:
            is_notificatin_showing = False
            notification_last_shown = 0
        else:
            notification_last_shown += 1
    last_point_time += 1
    success, image = cam.read()
    image = cv2.flip(image, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(1) & 0xFF
    _key = chr(key)
    if _key in BRUSH_COLORS:
        BRUSH_COLOR = BRUSH_COLORS[_key]
    if _key in BRUSH_SIZES:
        BRUSH_SIZE = BRUSH_SIZES[_key]
    if _key == 'e':
        points = []
    if _key == 's':
        save_paint(points)
        is_notificatin_showing = True
        points = []
    if _key == 'q':
        break
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
            is_end_point = True
        if state == STATES["PAINTING"]:
            if last_point_time >= POINT_INTERVAL:
                last_point_time = 0
                point_is_valid = False
                new_point = Point(
                    INDEX_FINGER_TIP[0] * wCam,
                    INDEX_FINGER_TIP[1] * hCam,
                    type=Point.POINT_TYPE['END'] if is_end_point else Point.POINT_TYPE['LINE'],
                    color= BRUSH_COLOR,
                    size=BRUSH_SIZE
                )
                is_end_point = False
                if check_bounding_box(new_point, CANVAS_BOUNDING_BOX):
                    if (len(points) > 0 and check_duplicate_point(points[-1].get_coords(), new_point.get_coords())) or len(points) == 0:
                        point_is_valid = True
                if point_is_valid:
                    points.append(new_point)
    draw_bounding_box()
    draw_points(image, points)
    draw_hints(image)
    if is_notificatin_showing:
        draw_notification(image, 'Canvas Saved!')
    cv2.imshow('HandPaint', image)
cam.release()
