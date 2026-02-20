import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ---- Tunables ----
CANNY_LO, CANNY_HI = 80, 160

MIN_COMP_PIX = 90
MAX_COMP_PIX = 4000

MIN_AXIS = 40
MAX_AXIS = 380

MIN_ASPECT = 0.3
MAX_ASPECT = 1.00

SAMPLE_N = 300

# NMS
NMS_CENTER_DIST = 18
NMS_AXIS_DIST = 12

def ellipse_points(cx, cy, a, b, ang_deg, n=120):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ca, sa = np.cos(np.deg2rad(ang_deg)), np.sin(np.deg2rad(ang_deg))
    x = a*np.cos(t)
    y = b*np.sin(t)
    xr = ca*x - sa*y + cx
    yr = sa*x + ca*y + cy
    return np.stack([xr, yr], axis=1)

def support_score(edges, cx, cy, MA, ma, angle_deg, thick=2, n=120):
    a = MA / 2.0
    b = ma / 2.0
    pts = ellipse_points(cx, cy, a, b, angle_deg, n=n)

    h, w = edges.shape
    hits = 0
    for (xf, yf) in pts:
        x = int(round(xf)); y = int(round(yf))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x0 = max(0, x - thick); x1 = min(w, x + thick + 1)
        y0 = max(0, y - thick); y1 = min(h, y + thick + 1)
        if np.any(edges[y0:y1, x0:x1] > 0):
            hits += 1
    return hits / float(n)

def nms_ellipses(ells):
    ells = sorted(ells, key=lambda d: d["score"], reverse=True)
    kept = []
    for e in ells:
        ok = True
        for k in kept:
            dc = np.hypot(e["cx"] - k["cx"], e["cy"] - k["cy"])
            da = abs(e["MA"] - k["MA"]) + abs(e["ma"] - k["ma"])
            if dc < NMS_CENTER_DIST and da < NMS_AXIS_DIST:
                ok = False
                break
        if ok:
            kept.append(e)
    return kept

def clamp_roi(x0, y0, x1, y1, w, h):
    x0 = int(max(0, min(w - 1, x0)))
    y0 = int(max(0, min(h - 1, y0)))
    x1 = int(max(0, min(w, x1)))
    y1 = int(max(0, min(h, y1)))
    if x1 <= x0 + 5 or y1 <= y0 + 5:
        return 0, 0, w, h
    return x0, y0, x1, y1

def biggest_red_roi(frame_bgr, pad=60, min_area=1500):
    """
    Returns (x0,y0,x1,y1, red_mask) for the biggest red blob.
    If nothing solid found, returns full-frame ROI.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # red wraps hue, so use two bands
    lower1 = np.array([0, 80, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 80, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red = cv2.bitwise_or(m1, m2)

    # clean up
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, ker, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, ker, iterations=2)

    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, w, h, red

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return 0, 0, w, h, red

    x, y, rw, rh = cv2.boundingRect(cnt)
    x0 = x - pad
    y0 = y - pad
    x1 = x + rw + pad
    y1 = y + rh + pad
    x0, y0, x1, y1 = clamp_roi(x0, y0, x1, y1, w, h)
    return x0, y0, x1, y1, red


def biggest_blue_roi(frame_bgr, pad=60, min_area=1500):
    """
    Returns (x0,y0,x1,y1, red_mask) for the biggest red blob.
    If nothing solid found, returns full-frame ROI.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Blue range (primary band)
    lower_blue = np.array([100, 100, 50], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # clean up
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, ker, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, ker, iterations=2)

    cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, w, h, blue_mask

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return 0, 0, w, h, blue_mask

    x, y, rw, rh = cv2.boundingRect(cnt)
    x0 = x - pad
    y0 = y - pad
    x1 = x + rw + pad
    y1 = y + rh + pad
    x0, y0, x1, y1 = clamp_roi(x0, y0, x1, y1, w, h)
    return x0, y0, x1, y1, blue_mask

# ---------- UI ----------
cv2.namedWindow("Tuning")

cv2.createTrackbar("Ring Thick", "Tuning", 5, 20, lambda x: None)
cv2.createTrackbar("Support Thresh x100", "Tuning", 60, 100, lambda x: None)
cv2.createTrackbar("Dilate Iterations", "Tuning", 3, 10, lambda x: None)
cv2.createTrackbar("Canny Low", "Tuning", 80, 255, lambda x: None)
cv2.createTrackbar("Min Axis", "Tuning", 40, 200, lambda x: None)
cv2.createTrackbar("Mask Lower", "Tuning", 0, 255, lambda x: None)

# new ROI controls
cv2.createTrackbar("Use Red ROI", "Tuning", 1, 1, lambda x: None)
cv2.createTrackbar("Red Pad", "Tuning", 70, 200, lambda x: None)
cv2.createTrackbar("Red MinArea", "Tuning", 15, 200, lambda x: None)  # *100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # ------- your existing "bright/white-ish" mask -------
    MASK_LOWER = cv2.getTrackbarPos("Mask Lower", "Tuning")
    lower = np.array([MASK_LOWER, MASK_LOWER, MASK_LOWER], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    mask0 = cv2.inRange(frame, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # ------- NEW: find biggest red blob ROI -------
    use_roi = cv2.getTrackbarPos("Use Red ROI", "Tuning") == 1
    red_pad = cv2.getTrackbarPos("Red Pad", "Tuning")
    red_min_area = cv2.getTrackbarPos("Red MinArea", "Tuning") * 100

    if use_roi:
        x0, y0, x1, y1, redmask = biggest_red_roi(frame, pad=red_pad, min_area=red_min_area)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
        redmask = np.zeros((H, W), dtype=np.uint8)

    # Apply your mask (global), then crop for edge/ellipse work
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    roi_frame = masked_frame[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]

    # ------- trackbar values -------
    RING_THICK = cv2.getTrackbarPos("Ring Thick", "Tuning")
    SUPPORT_THRESH = cv2.getTrackbarPos("Support Thresh x100", "Tuning") / 100.0
    DILATE_ITER = cv2.getTrackbarPos("Dilate Iterations", "Tuning")
    CANNY_LO = cv2.getTrackbarPos("Canny Low", "Tuning")
    MIN_AXIS = cv2.getTrackbarPos("Min Axis", "Tuning")

    if CANNY_LO == 0:
        CANNY_LO = 1
    CANNY_HI = CANNY_LO * 2

    out = frame.copy()

    # ROI debug box
    if use_roi:
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 2)

    # ------- run edges ONLY on ROI -------
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LO, CANNY_HI)
    edges_stable = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K3, iterations=1)
    edges_stable = cv2.dilate(edges_stable, K3, iterations=DILATE_ITER)

    # Connected components on ROI edges
    num, labels, stats, _ = cv2.connectedComponentsWithStats((edges_stable > 0).astype(np.uint8), connectivity=8)

    candidates = []
    for lab in range(1, num):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < MIN_COMP_PIX or area > MAX_COMP_PIX:
            continue

        ys, xs = np.where(labels == lab)
        if len(xs) < 5:
            continue

        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(pts)

        if MA < MIN_AXIS or ma < MIN_AXIS or MA > MAX_AXIS or ma > MAX_AXIS:
            continue

        aspect = min(MA, ma) / max(MA, ma)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue

        sc = support_score(edges_stable, cx, cy, MA, ma, angle, thick=RING_THICK, n=SAMPLE_N)
        if sc < SUPPORT_THRESH:
            continue

        candidates.append({"cx": cx, "cy": cy, "MA": MA, "ma": ma, "angle": angle, "score": sc})

    kept = nms_ellipses(candidates)

    # Draw ellipses back in full-frame coords
    for e in kept:
        cx_roi, cy_roi = e["cx"], e["cy"]
        cx = cx_roi + x0
        cy = cy_roi + y0

        MA, ma, angle = e["MA"], e["ma"], e["angle"]

        cv2.ellipse(out, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(out, f'{e["score"]:.2f}', (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display: show ROI edge maps but also a full-frame overlay
    cv2.imshow("ellipses", out)
    cv2.imshow("edges_canny_roi", edges)
    cv2.imshow("edges_stable_roi", edges_stable)
    cv2.imshow("Mask", mask)
    cv2.imshow("RedMask", redmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()