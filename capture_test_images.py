import cv2
import os
from datetime import datetime

# Create output directory
output_dir = "test_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

count = 0
print(f"Saving images to '{output_dir}' directory")
print("Press 'SPACE' to capture an image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space to capture
        filename = os.path.join(output_dir, f"test_image_{count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1
    elif key == ord('q'):  # q to quit
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total images saved: {count}")
