import requests
import bz2
import os
import sys

URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FILE_BZ2 = "shape_predictor_68_face_landmarks.dat.bz2"
FILE_DAT = "shape_predictor_68_face_landmarks.dat"

def download_and_extract():
    if os.path.exists(FILE_DAT):
        print(f"{FILE_DAT} already exists.")
        return

    print(f"Downloading {FILE_BZ2} from {URL}...")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        with open(FILE_BZ2, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    print(f"Extracting {FILE_BZ2}...")
    try:
        with bz2.open(FILE_BZ2, 'rb') as source, open(FILE_DAT, 'wb') as dest:
            dest.write(source.read())
        print(f"Extracted to {FILE_DAT}")
        
        # Cleanup
        os.remove(FILE_BZ2)
    except Exception as e:
        print(f"Error extracting: {e}")

if __name__ == "__main__":
    download_and_extract()
