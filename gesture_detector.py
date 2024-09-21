from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp

class GestureDetectorLogger:

    def __init__(self, video_mode: bool = False):
        self._video_mode = video_mode
        base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)


    def detect(self, image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognition_result = self.recognizer.recognize(image)
        gesture = None
        hand_landmarks = None
        if len(recognition_result.gestures) > 0:
            gesture = recognition_result.gestures[0][0]
        hand_landmarks = recognition_result.hand_landmarks
        return [gesture, hand_landmarks]
