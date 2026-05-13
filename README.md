# Driver Drowsiness Detection System

A real-time safety system that monitors driver fatigue using Computer Vision. It detects closed eyes (drowsiness) and yawning using facial landmarks.

## Features
- **Real-time Detection**: Monitors face via webcam.
- **Eye Aspect Ratio (EAR)**: Detects extended eye closure (sleeping).
- **Mouth Aspect Ratio (MAR)**: Detects yawning.
- **Alerts**:
    - **Visual**: On-screen "DROWSINESS ALERT!" and "YAWNING ALERT!" text.
    - **Audio**: Plays an alarm sound when drowsiness is detected.

## Project Structure
```
Driver Drowsiness Detection/
│
├── detector.py           # Dlib wrapper for face/landmark detection
├── eye_aspect_ratio.py   # EAR/MAR calculation logic
├── alarm.py              # Audio alarm handler
├── main.py               # Main application loop
├── download_dlib_model.py # Helper to download the predictor
└── shape_predictor_68_face_landmarks.dat (downloaded)
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `dlib` fails, try `pip install dlib-bin` or ensure CMake is installed.*

2.  **Download Model**:
    ```bash
    python download_dlib_model.py
    ```
    *This downloads the required 100MB landmark predictor.*

## Running the Application

1.  **Start Detection**:
    ```bash
    python main.py
    ```
2.  **Stop**: Press `q` to exit.

## Usage
- Ensure good lighting for best face detection.
- The alarm triggers if eyes remain closed for ~1-2 seconds (configurable in `main.py`).
