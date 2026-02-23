import numpy as np
import cv2

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) # (0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ---- Tunables ----
CANNY_LO, CANNY_HI = 80, 160

MIN_COMP_PIX = 90
MAX_COMP_PIX = 4000

MIN_MAJOR_AXIS = 40
MIN_MINOR_AXIS = 12
MAX_AXIS = 200

MIN_ASPECT = 0.18
MAX_ASPECT = 1.00

SAMPLE_N = 300

# robust fitting
RANDOM_FIT_TRIALS = 22
RANDOM_FIT_POINTS = 90
MIN_CONTOUR_POINTS = 24

# fit quality
MIN_INLIER_SCORE = 0.52
MIN_SUPPORT_FALLBACK = 0.28
INLIER_TOL = 0.18
STRICT_INLIER_SCORE = 0.72
MIN_INLIER_POINTS = 30
BRIDGE_ITER = 1

# NMS
NMS_CENTER_DIST = 28
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

def ellipse_inlier_score(points_xy, cx, cy, MA, ma, angle_deg, tol=0.18):
    if len(points_xy) == 0:
        return 0.0

    a = MA / 2.0
    b = ma / 2.0
    if a <= 1e-6 or b <= 1e-6:
        return 0.0

    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)

    x = points_xy[:, 0].astype(np.float32) - cx
    y = points_xy[:, 1].astype(np.float32) - cy

    xr = c * x + s * y
    yr = -s * x + c * y

    val = (xr / a) ** 2 + (yr / b) ** 2
    return float(np.mean(np.abs(val - 1.0) <= tol))

def maybe_add_candidate(candidates, points_xy, edges_stable, cx, cy, MA, ma, angle, ring_thick, support_thresh):
    major_axis = max(MA, ma)
    minor_axis = min(MA, ma)

    if major_axis < MIN_MAJOR_AXIS or minor_axis < MIN_MINOR_AXIS or major_axis > MAX_AXIS:
        return

    aspect = minor_axis / major_axis
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return

    sc = support_score(edges_stable, cx, cy, MA, ma, angle, thick=ring_thick, n=SAMPLE_N)
    inlier = ellipse_inlier_score(points_xy, cx, cy, MA, ma, angle, tol=INLIER_TOL)

    keep = (
        (sc >= support_thresh)
        or (sc >= MIN_SUPPORT_FALLBACK and inlier >= MIN_INLIER_SCORE)
        or (inlier >= STRICT_INLIER_SCORE and len(points_xy) >= MIN_INLIER_POINTS)
    )
    if not keep:
        return

    candidates.append({
        "cx": cx,
        "cy": cy,
        "MA": MA,
        "ma": ma,
        "angle": angle,
        "score": sc,
        "inlier": inlier,
        "rank": 0.55 * sc + 0.45 * inlier,
    })

def nms_ellipses(ells):
    ells = sorted(ells, key=lambda d: d["rank"], reverse=True)
    kept = []
    for e in ells:
        keep = True
        for k in kept:
            dc = np.hypot(e["cx"] - k["cx"], e["cy"] - k["cy"])
            if dc <= NMS_CENTER_DIST:
                keep = False
                break
        if keep:
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
cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuning", 520, 720)

cv2.createTrackbar("Ring Thick", "Tuning", 5, 20, lambda x: None)
cv2.createTrackbar("Support Thresh x100", "Tuning", 60, 100, lambda x: None)
cv2.createTrackbar("Dilate Iterations", "Tuning", 3, 10, lambda x: None)
cv2.createTrackbar("Canny Low", "Tuning", 80, 255, lambda x: None)
cv2.createTrackbar("Min Major Axis", "Tuning", 40, 250, lambda x: None)
cv2.createTrackbar("Min Minor Axis", "Tuning", 12, 120, lambda x: None)
cv2.createTrackbar("Max Axis", "Tuning", 180, 450, lambda x: None)
cv2.createTrackbar("Min Aspect x100", "Tuning", 18, 100, lambda x: None)
cv2.createTrackbar("Mask Lower", "Tuning", 0, 255, lambda x: None)
cv2.createTrackbar("Fallback Supp x100", "Tuning", 28, 100, lambda x: None)
cv2.createTrackbar("Min Inlier x100", "Tuning", 52, 100, lambda x: None)
cv2.createTrackbar("Strict Inlier x100", "Tuning", 72, 100, lambda x: None)
cv2.createTrackbar("Min Inlier Pts", "Tuning", 30, 300, lambda x: None)
cv2.createTrackbar("Inlier Tol x100", "Tuning", 18, 60, lambda x: None)
cv2.createTrackbar("Rand Fit Trials", "Tuning", 22, 80, lambda x: None)
cv2.createTrackbar("Rand Fit Points", "Tuning", 90, 220, lambda x: None)
cv2.createTrackbar("Min Cnt Pts", "Tuning", 24, 200, lambda x: None)
cv2.createTrackbar("Bridge Iter", "Tuning", 1, 6, lambda x: None)
cv2.createTrackbar("NMS Ctr Dist", "Tuning", 28, 120, lambda x: None)

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
    MIN_MAJOR_AXIS = cv2.getTrackbarPos("Min Major Axis", "Tuning")
    MIN_MINOR_AXIS = cv2.getTrackbarPos("Min Minor Axis", "Tuning")
    MAX_AXIS = max(MIN_MAJOR_AXIS + 1, cv2.getTrackbarPos("Max Axis", "Tuning"))
    MIN_ASPECT = cv2.getTrackbarPos("Min Aspect x100", "Tuning") / 100.0
    MIN_SUPPORT_FALLBACK = cv2.getTrackbarPos("Fallback Supp x100", "Tuning") / 100.0
    MIN_INLIER_SCORE = cv2.getTrackbarPos("Min Inlier x100", "Tuning") / 100.0
    STRICT_INLIER_SCORE = cv2.getTrackbarPos("Strict Inlier x100", "Tuning") / 100.0
    MIN_INLIER_POINTS = max(5, cv2.getTrackbarPos("Min Inlier Pts", "Tuning"))
    INLIER_TOL = max(0.01, cv2.getTrackbarPos("Inlier Tol x100", "Tuning") / 100.0)
    RANDOM_FIT_TRIALS = cv2.getTrackbarPos("Rand Fit Trials", "Tuning")
    RANDOM_FIT_POINTS = max(5, cv2.getTrackbarPos("Rand Fit Points", "Tuning"))
    MIN_CONTOUR_POINTS = max(5, cv2.getTrackbarPos("Min Cnt Pts", "Tuning"))
    BRIDGE_ITER = cv2.getTrackbarPos("Bridge Iter", "Tuning")
    NMS_CENTER_DIST = max(1, cv2.getTrackbarPos("NMS Ctr Dist", "Tuning"))

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

    # fit map: extra bridging for broken arcs; support still checked on edges_stable
    if BRIDGE_ITER > 0:
        edges_fit = cv2.morphologyEx(edges_stable, cv2.MORPH_CLOSE, K5, iterations=BRIDGE_ITER)
        edges_fit = cv2.dilate(edges_fit, K3, iterations=BRIDGE_ITER)
    else:
        edges_fit = edges_stable

    # Connected components on ROI edges
    num, labels, stats, _ = cv2.connectedComponentsWithStats((edges_fit > 0).astype(np.uint8), connectivity=8)

    candidates = []

    # Path A: contour-based fits (works better on open arc fragments)
    cnts, _ = cv2.findContours(edges_fit, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        if len(cnt) < MIN_CONTOUR_POINTS:
            continue

        pts_cnt = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts_cnt) < 5:
            continue

        (cc, rr), (MA, ma), angle = cv2.fitEllipse(cnt)
        maybe_add_candidate(
            candidates,
            pts_cnt,
            edges_stable,
            cc,
            rr,
            MA,
            ma,
            angle,
            RING_THICK,
            SUPPORT_THRESH,
        )

    # Path B: component fits + random subset fits (handles merged overlapping rims)
    for lab in range(1, num):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < MIN_COMP_PIX or area > MAX_COMP_PIX:
            continue

        ys, xs = np.where(labels == lab)
        if len(xs) < 5:
            continue

        pts_xy = np.stack([xs, ys], axis=1).astype(np.float32)
        pts_cv = pts_xy.astype(np.int32).reshape(-1, 1, 2)

        (cx, cy), (MA, ma), angle = cv2.fitEllipse(pts_cv)
        maybe_add_candidate(
            candidates,
            pts_xy,
            edges_stable,
            cx,
            cy,
            MA,
            ma,
            angle,
            RING_THICK,
            SUPPORT_THRESH,
        )

        if len(pts_xy) >= RANDOM_FIT_POINTS:
            for _ in range(RANDOM_FIT_TRIALS):
                idx = np.random.choice(len(pts_xy), RANDOM_FIT_POINTS, replace=False)
                sub = pts_xy[idx]
                sub_cv = sub.astype(np.int32).reshape(-1, 1, 2)
                try:
                    (cx2, cy2), (MA2, ma2), ang2 = cv2.fitEllipse(sub_cv)
                except cv2.error:
                    continue

                maybe_add_candidate(
                    candidates,
                    pts_xy,
                    edges_stable,
                    cx2,
                    cy2,
                    MA2,
                    ma2,
                    ang2,
                    RING_THICK,
                    SUPPORT_THRESH,
                )

    kept = nms_ellipses(candidates)

    # Draw ellipses back in full-frame coords
    for e in kept:
        cx_roi, cy_roi = e["cx"], e["cy"]
        cx = cx_roi + x0
        cy = cy_roi + y0

        MA, ma, angle = e["MA"], e["ma"], e["angle"]

        cv2.ellipse(out, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(out, f's:{e["score"]:.2f} i:{e["inlier"]:.2f}', (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display: show ROI edge maps but also a full-frame overlay
    cv2.imshow("ellipses", out)
    cv2.imshow("edges_canny_roi", edges)
    cv2.imshow("edges_stable_roi", edges_stable)
    cv2.imshow("edges_fit_roi", edges_fit)
    cv2.imshow("Mask", mask)
    cv2.imshow("RedMask", redmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()