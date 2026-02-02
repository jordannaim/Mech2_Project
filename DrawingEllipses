import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def best_ellipse_from_edges(edge_roi):
    # Find contours inside a single separated cup region, then pick best ellipse by score
    contours, _ = cv2.findContours(edge_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best = None
    best_score = -1.0

    for c in contours:
        if len(c) < 30:
            continue

        area = cv2.contourArea(c)
        if area < 250:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue

        # Ellipse fit
        if len(c) < 5:
            continue
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
        if MA <= 0 or ma <= 0:
            continue

        aspect = min(MA, ma) / max(MA, ma)  # 1.0 is circle
        if aspect < 0.25:
            continue

        circularity = 4.0 * np.pi * area / (peri * peri)

        # Prefer: ellipse-like, reasonably round, reasonably sized
        # Score weights: aspect + circularity + log(area)
        score = (2.0 * aspect) + (1.5 * np.clip(circularity, 0, 1)) + (0.3 * np.log(area + 1))

        if score > best_score:
            best_score = score
            best = ((cx, cy), (MA, ma), ang)

    return best

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out = frame.copy()

    # -------------------------
    # KEEP YOUR EXISTING MASK
    # -------------------------
    # Replace this block with your current working cup mask (binary 0/255).
    # This default is a red-cup HSV mask as a fallback.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K9, iterations=2)

    # -------------------------
    # SPLIT TOUCHING CUPS (WATERSHED)
    # -------------------------
    # If two cups touch, their mask blob merges. Watershed splits them into instances.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Tune threshold: higher -> fewer markers; lower -> more markers
    _, sure_fg = cv2.threshold(dist_norm, 0.35, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, K5, iterations=1)

    sure_bg = cv2.dilate(mask, K9, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_img = frame.copy()
    markers = cv2.watershed(ws_img, markers)  # markers now has separated regions

    # -------------------------
    # CANNY EDGES (UNCHANGED CORE)
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 160)

    # Optional: keep edges thin but connect tiny breaks along rims
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K5, iterations=1)

    # -------------------------
    # FIT ONE ELLIPSE PER WATERSHED REGION
    # -------------------------
    # Labels: -1 are boundaries, 1 is background, 2.. are objects
    unique_labels = np.unique(markers)
    for lab in unique_labels:
        if lab <= 1:
            continue

        region = (markers == lab).astype(np.uint8) * 255

        # Restrict to a slightly eroded region so the boundary between touching cups
        # doesn't pollute the rim edge set
        region_eroded = cv2.erode(region, K5, iterations=2)

        edge_roi = cv2.bitwise_and(edges, edges, mask=region_eroded)

        ell = best_ellipse_from_edges(edge_roi)
        if ell is None:
            continue

        (cx, cy), (MA, ma), ang = ell

        # Reject obviously tiny/huge ellipses (tune to your camera distance)
        if MA * ma < 800:
            continue

        cv2.ellipse(out, ((cx, cy), (MA, ma), ang), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    # Debug views
    cv2.imshow("mask", mask)
    cv2.imshow("edges", edges)
    cv2.imshow("lid_contours", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
