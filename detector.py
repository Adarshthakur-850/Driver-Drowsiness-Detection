import dlib
import cv2
import os

class FaceDetector:
    def __init__(self, predictor_path):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor not found at {predictor_path}")
            
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_faces(self, gray_frame):
        return self.detector(gray_frame, 0)

    def get_landmarks(self, gray_frame, rect):
        return self.predictor(gray_frame, rect)
