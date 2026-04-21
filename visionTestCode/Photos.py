import cv2
import os
import time
from datetime import datetime

# -------- Settings --------
SAVE_FOLDER = "/home/pi/camera_photos"   # Change this if needed
NUM_PHOTOS = 50
CAMERA_INDEX = 0                         # Usually 0 for first USB camera
DELAY_BETWEEN_PHOTOS = 0.5               # Seconds between shots
# --------------------------

def main():
    # Create save folder if it doesn't exist
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        return

    print(f"Camera opened successfully.")
    print(f"Saving {NUM_PHOTOS} photos to: {SAVE_FOLDER}")

    for i in range(NUM_PHOTOS):
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to capture photo {i+1}")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SAVE_FOLDER, f"photo_{i+1:02d}_{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        time.sleep(DELAY_BETWEEN_PHOTOS)

    cap.release()
    print("Done capturing photos.")

if __name__ == "__main__":
    main()