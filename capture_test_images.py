import cv2
import os
from datetime import datetime


def configure_max_resolution(cap):
    """
    Prefer 1080p capture by forcing MJPG first, then probe fallbacks.
    Returns (width, height, fps, pixel_format).
    """
    # 1080p at full frame rate is usually only available with MJPG on USB cams.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)

    preferred = [
        (1920, 1080),
        (1600, 900),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
    ]

    chosen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    chosen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for w, h in preferred:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Let backend apply settings.
        for _ in range(3):
            cap.read()

        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if got_w == w and got_h == h:
            chosen_w, chosen_h = got_w, got_h
            break

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip() or "UNKNOWN"
    return chosen_w, chosen_h, fps, fourcc

# Create output directory
output_dir = "test_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Could not open camera device 0")

w, h, fps, pix_fmt = configure_max_resolution(cap)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

count = 20
print(f"Saving images to '{output_dir}' directory")
print(f"Capture mode: {w}x{h} @ {fps:.1f} FPS, format={pix_fmt}")
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
