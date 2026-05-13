import platform
import threading
import time

# Try importing playsound, fallback to winsound or print
try:
    from playsound import playsound
    HAS_PLAYSOUND = True
except ImportError:
    HAS_PLAYSOUND = False

# For Windows beep
import winsound

def play_sound_file(path='alarm.wav'):
    if HAS_PLAYSOUND:
        try:
            playsound(path)
        except Exception as e:
            print(f"Error playing sound: {e}")
            # Fallback
            winsound.Beep(2500, 1000)
    else:
        # Windows fallback
        winsound.Beep(2500, 1000)

def sound_alarm(path='alarm.wav'):
    # Run in a separate thread to not block the main video loop
    t = threading.Thread(target=play_sound_file, args=(path,))
    t.daemon = True
    t.start()
