import cv2
import dlib
import numpy as np
import requests
import os
import sys
from imutils import face_utils

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from detector import FaceDetector
from eye_aspect_ratio import eye_aspect_ratio

def download_sample_image():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    filename = "sample_face.jpg"
    if not os.path.exists(filename):
        print(f"Downloading sample image from {url}...")
        resp = requests.get(url)
        with open(filename, "wb") as f:
            f.write(resp.content)
        print("Download complete.")
    return filename

def test_simulation():
    print("Initializing Simulation...")
    
    # 1. Setup Detector
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print("FAILED: Predictor file not found.")
        return

    try:
        detector = FaceDetector(predictor_path)
        print("Detector initialized.")
    except Exception as e:
        print(f"FAILED to init detector: {e}")
        return

    # 2. Get Image
    img_path = download_sample_image()
    frame = cv2.imread(img_path)
    if frame is None:
        print("FAILED to load image.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 3. Detect Face
    print("Detecting faces...")
    rects = detector.detect_faces(gray)
    print(f"Faces found: {len(rects)}")
    
    if len(rects) == 0:
        print("FAILED: No faces detected in sample image.")
        return

    # 4. Process Landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    for rect in rects:
        shape = detector.get_landmarks(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        print(f"Face detected. Left EAR: {leftEAR:.2f}, Right EAR: {rightEAR:.2f}, Avg EAR: {ear:.2f}")
        
        # Verify EAR is reasonable (Open eyes ~0.30)
        if ear > 0.2:
            print("SUCCESS: EAR indicates open eyes (correct for Lena).")
        else:
            print("WARNING: EAR is low (closed eyes?). check logic.")

if __name__ == "__main__":
    test_simulation()
