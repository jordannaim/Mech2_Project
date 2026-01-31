import numpy as np
import cv2

# Try different backends - V4L2 often works better on Pi
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Optional: improve contour stability
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 80, 160)

    # Close small gaps so cup rims become single loops
    # edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    edges_closed = edges
    # Identifying Contours (find on binary image, not raw gray)
    contours, hierarchy = cv2.findContours(
        edges_closed,
        #cv2.RETR_EXTERNAL,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours that look like Solo cup lids (roughly circular/elliptical)
    out = frame.copy()
    contOut = frame.copy()
    cv2.drawContours(contOut, contours, -1, (0, 255, 0), 2)
    cv2.imshow("all contours", contOut)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:   # ignore noise; tune as needed
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)  # 1.0 = perfect circle

        # Lids/rims usually have moderately high circularity
        if circularity < 0.0:
            continue

        # Fit ellipse if enough points, helps reject non-rim shapes
        if len(c) >= 5:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)
            if MA <= 0 or ma <= 0:
                continue
            aspect = min(MA, ma) / max(MA, ma)  # 1.0 = circle
            if aspect < 0.2:
                continue

            cv2.ellipse(out, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
            cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        else:
            cv2.drawContours(out, [c], -1, (0, 255, 0), 2)

    cv2.imshow("edges", edges_closed)
    cv2.imshow("lid_contours", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
