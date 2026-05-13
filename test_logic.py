import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from eye_aspect_ratio import eye_aspect_ratio, mouth_aspect_ratio
from alarm import play_sound_file

def test_logic():
    print("Testing Drowsiness Logic...")

    # Test 1: EAR Calculation
    # Fake eye landmarks (6 points) - Open Eye
    # p1--p2
    # |    |
    # p6--p3
    # |    |
    # p5--p4
    # Distance vertical (p2-p6, p3-p5) should be large
    # Distance horizontal (p1-p4) is constant
    
    # 6 points: (x, y)
    eye_open = np.array([
        (10, 10), (20, 5), (30, 5), 
        (40, 10), (30, 15), (20, 15)
    ])
    ear_open = eye_aspect_ratio(eye_open)
    print(f"EAR (Open): {ear_open:.2f}")
    
    # Fake eye - Closed (vertical distance near 0)
    eye_closed = np.array([
        (10, 10), (20, 10), (30, 10), 
        (40, 10), (30, 10), (20, 10)
    ])
    ear_closed = eye_aspect_ratio(eye_closed)
    print(f"EAR (Closed): {ear_closed:.2f}")
    
    # Assert
    assert ear_open > ear_closed
    print("SUCCESS: EAR Logic Correct.")

    # Test 2: Sound (Mock)
    print("Testing Sound (This might beep)...")
    play_sound_file() 
    print("SUCCESS: Sound function executed.")
    
    # Test 3: Detector Initialization
    print("Testing Detector Loading...")
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print(f"SKIPPING: {predictor_path} not found yet (downloading?).")
    else:
        try:
            import dlib
            dlib.shape_predictor(predictor_path)
            print("SUCCESS: Dlib Predictor loaded.")
        except Exception as e:
            print(f"FAILED to load predictor: {e}")

if __name__ == "__main__":
    test_logic()
