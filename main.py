import cv2
import time
import numpy as np
import dlib
from imutils import face_utils
from threading import Thread

from detector import FaceDetector
from eye_aspect_ratio import eye_aspect_ratio, mouth_aspect_ratio
from alarm import sound_alarm

# Constants
EAR_THRESHOLD = 0.25 # If EAR drops below this, potential drowsiness
EAR_CONSEC_FRAMES = 20 # Number of consecutive frames to trigger alarm

MAR_THRESHOLD = 0.6 # If MAR is above this, potential yawning (mouth open)
MAR_CONSEC_FRAMES = 20

def main():
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    try:
        detector_wrapper = FaceDetector(predictor_path)
    except FileNotFoundError:
        print(f"Error: {predictor_path} not found. Run 'python download_dlib_model.py' first.")
        return

    # Initialize counters
    COUNTER_EYE = 0
    ALARM_ON = False
    
    COUNTER_MOUTH = 0
    
    # Start video
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)
    
    if not cap.isOpened():
        print("Error: Could not open video source (webcam).")
        return

    print("Starting Drowsiness Detection... Press 'q' to quit.")
    
    # Grab landmark indices for eyes and mouth
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector_wrapper.detect_faces(gray)
        
        for rect in rects:
            shape = detector_wrapper.get_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            
            # Compute EAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # Compute MAR (for yawning)
            mar = mouth_aspect_ratio(mouth)
            
            # Visuals - Draw eyes and mouth
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)
            
            # Check Drowsiness (Eyes)
            if ear < EAR_THRESHOLD:
                COUNTER_EYE += 1
                
                if COUNTER_EYE >= EAR_CONSEC_FRAMES:
                    # if not ALARM_ON:
                    #     ALARM_ON = True
                    #     sound_alarm()
                    
                    sound_alarm() # Play alarm repeatedly or once depends on alarm.py implementation
                    
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER_EYE = 0
                ALARM_ON = False
                
            # Check Yawning (Mouth)
            if mar > MAR_THRESHOLD:
                COUNTER_MOUTH += 1
                if COUNTER_MOUTH >= MAR_CONSEC_FRAMES:
                    cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER_MOUTH = 0

            # Display EAR/MAR
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
