import numpy as np
import cv2
import json
import os
import glob


def configure_max_resolution(cap):
    """
    Prefer highest stable capture mode, prioritizing 1080p MJPG.
    Returns (width, height, fps, pixel_format).
    """
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

# ---- Use test images instead of camera feed ----
USE_TEST_IMAGES = False

if USE_TEST_IMAGES:
    test_image_dir = "test_images/newer_test_images"
    test_images = sorted(glob.glob(os.path.join(test_image_dir, "*.jpg")))
    if not test_images:
        print(f"No images found in {test_image_dir}")
        exit(1)
    image_index = 0
    print(f"Loaded {len(test_images)} test images")
    cap = None
else:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera device 0")

    w, h, fps, pix_fmt = configure_max_resolution(cap)
    print(f"Camera mode: {w}x{h} @ {fps:.1f} FPS, format={pix_fmt}")
    cap.set(cv2.CAP_PROP_CONTRAST, 50)
    test_images = None
    image_index = 0

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "ellipse_tuning.json")

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

# white infill tuning

# target persistence / aiming
TARGET_NEIGHBOR_RADIUS = 180.0
TARGET_PERSIST_DIST = 140.0
TARGET_MATCH_MAX_DIST = 90.0
TARGET_SWITCH_MARGIN = 0.06
TARGET_MISS_TOLERANCE = 3

def ellipse_points(cx, cy, a, b, ang_deg, n=120):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ca, sa = np.cos(np.deg2rad(ang_deg)), np.sin(np.deg2rad(ang_deg))
    x = a*np.cos(t)
    y = b*np.sin(t)
    xr = ca*x - sa*y + cx
    yr = sa*x + ca*y + cy
    return np.stack([xr, yr], axis=1)

def support_score(edges, cx, cy, MA, ma, angle_deg, thick=2, n=120):
    """
    Calculate what fraction of the ellipse perimeter has supporting edges.
    Thicker ring catches gaps better. Returns 0-1 score.
    """
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

def partial_arc_score(edges, cx, cy, MA, ma, angle_deg, tol=0.18):
    """
    For partial ellipses: score based on arc continuity.
    Find which parts of the ellipse have edge support, then check if those
    regions form continuous smooth arcs rather than random scattered points.
    High score = good arc alignment + continuity even if incomplete.
    """
    a = MA / 2.0
    b = ma / 2.0
    if a <= 1e-6 or b <= 1e-6:
        return 0.0
    
    h, w = edges.shape
    pts = ellipse_points(cx, cy, a, b, angle_deg, n=360)  # High resolution for arc detection
    
    th = np.deg2rad(angle_deg)
    c_ang, s_ang = np.cos(th), np.sin(th)
    
    # Check which points have support
    support_mask = np.zeros(len(pts), dtype=bool)
    for i, (xf, yf) in enumerate(pts):
        x = int(round(xf)); y = int(round(yf))
        if 0 <= x < w and 0 <= y < h:
            if edges[y, x] > 0:
                support_mask[i] = True
    
    if np.sum(support_mask) < 0.3 * len(pts):  # Need at least 30% support
        return 0.0
    
    # Check arc continuity: supported points should form smooth connected arcs
    # not random scattered pixels
    support_idx = np.where(support_mask)[0]
    if len(support_idx) == 0:
        return 0.0
    
    # Calculate gaps between consecutive supported points
    gaps = np.diff(support_idx)
    gaps = np.append(gaps, len(pts) - support_idx[-1] + support_idx[0])  # Wrap-around gap
    
    # Score: prefer smaller gaps (continuity) and high coverage
    gap_score = 1.0 - np.mean(np.minimum(gaps, 45) / 45.0)  # Normalize gap penalty
    coverage_score = np.sum(support_mask) / len(pts)
    
    # Combined: weight coverage heavily but reward continuity
    return 0.7 * coverage_score + 0.3 * gap_score

def maybe_add_candidate(candidates, points_xy, edges_stable, cx, cy, MA, ma, angle, support_thresh):
    major_axis = max(MA, ma)
    minor_axis = min(MA, ma)

    if major_axis < MIN_MAJOR_AXIS or minor_axis < MIN_MINOR_AXIS or minor_axis > MAX_MINOR_AXIS or major_axis > MAX_AXIS:
        return

    aspect = minor_axis / major_axis
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return
    
    # Filter for nearly horizontal major axis (within 25 degrees of 0 or 180)
    otherAngle = (angle + 90) % 180  # angle of minor axis
    angle_to_horizontal = min(abs(otherAngle), abs(180 - otherAngle))
    if angle_to_horizontal > 15:
        return

    sc = support_score(edges_stable, cx, cy, MA, ma, angle, thick=2, n=SAMPLE_N)
    inlier = ellipse_inlier_score(points_xy, cx, cy, MA, ma, angle, tol=INLIER_TOL)
    
    # For potentially partial ellipses (low support), use arc continuity scoring
    arc_score = partial_arc_score(edges_stable, cx, cy, MA, ma, angle)
    
    keep = (
        (sc >= support_thresh)
        or (sc >= MIN_SUPPORT_FALLBACK and inlier >= MIN_INLIER_SCORE)
        or (inlier >= STRICT_INLIER_SCORE and len(points_xy) >= MIN_INLIER_POINTS)
        # Accept decent support + inlier combination
        or (sc >= 0.60 and inlier >= 0.40)
        # NEW: Accept partial ellipses with good arc continuity (e.g., 90% complete with gap)
        or (arc_score >= 0.55)
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
        "arc_score": arc_score,
        "rank": 0.50 * sc + 0.35 * inlier + 0.15 * arc_score,
    })

def detect_spatial_outliers(ells, neighbor_radius=80.0, min_neighbors=1):
    """
    Identify outlier ellipses based on spatial isolation.
    For each ellipse, count how many neighbors are within neighbor_radius.
    Ellipses with fewer neighbors than min_neighbors are marked as outliers.
    Returns list of outlier indices.
    """
    if len(ells) <= 1:
        return []
    
    outlier_indices = []
    for i, e1 in enumerate(ells):
        neighbor_count = 0
        for j, e2 in enumerate(ells):
            if i == j:
                continue
            dist = np.hypot(e1["cx"] - e2["cx"], e1["cy"] - e2["cy"])
            if dist <= neighbor_radius:
                neighbor_count += 1
        
        # If fewer neighbors than threshold, mark as outlier
        if neighbor_count < min_neighbors:
            outlier_indices.append(i)
    
    return outlier_indices


def compute_cup_confidence(ells):
    """
    Compute confidence that each ellipse is a cup based on spatial clustering and similarity.
    Cups are expected to form tight clusters in space with multiple similar ellipses.
    Location proximity is the PRIMARY criterion.
    
    Returns list of ellipses with added 'cup_confidence' and 'is_outlier' fields.
    """
    if not ells:
        return ells
    
    # Detect spatial outliers (isolated ellipses far from others)
    outlier_indices = detect_spatial_outliers(ells, neighbor_radius=80.0, min_neighbors=1)
    
    for i, e1 in enumerate(ells):
        is_outlier = i in outlier_indices
        
        similarity_score = 0.0
        location_score = 0.0
        similar_count = 0
        close_neighbors = 0
        
        for j, e2 in enumerate(ells):
            if i == j:
                continue
            
            # Distance between centers (LOCATION IS PRIMARY)
            dist = np.hypot(e1["cx"] - e2["cx"], e1["cy"] - e2["cy"])
            
            # Normalize by typical cup size to get relative distance
            max_size = max(e1["MA"], e2["MA"])
            relative_dist = dist / max_size if max_size > 0 else 1.0
            
            # Close neighbors contribute to location_score
            # Prefer very nearby ellipses (within 50% of size)
            if relative_dist < 0.5:
                close_neighbors += 1
                location_score += (1.0 - relative_dist)  # Closer = higher score
            
            # Size similarity (ratio of major axes)
            size_ratio = min(e1["MA"], e2["MA"]) / max(e1["MA"], e2["MA"])
            
            # Shape similarity (aspect ratio comparison)
            aspect1 = min(e1["MA"], e1["ma"]) / max(e1["MA"], e1["ma"])
            aspect2 = min(e2["MA"], e2["ma"]) / max(e2["MA"], e2["ma"])
            aspect_diff = abs(aspect1 - aspect2)
            
            # Angle similarity
            angle_diff = min(abs(e1["angle"] - e2["angle"]), 
                           180 - abs(e1["angle"] - e2["angle"]))
            
            # Good match: close together, similar size, shape, angle
            if (relative_dist < 0.5 and size_ratio > 0.6 and 
                aspect_diff < 0.3 and angle_diff < 30):
                # Weight by how good the match is
                # Location proximity is weighted heavily
                match_score = ((1.0 - relative_dist) * 0.5 +
                             size_ratio * 0.3 + 
                             (1.0 - aspect_diff) * 0.15 +
                             (1.0 - angle_diff / 30.0) * 0.05)
                similarity_score += match_score
                similar_count += 1
        
        # Build confidence: location proximity is dominant
        # Normalize scores
        if close_neighbors > 0:
            location_score = location_score / close_neighbors
        else:
            location_score = 0.0
        
        if similar_count > 0:
            avg_similarity = similarity_score / similar_count
            # Multiple similar ellipses boost confidence
            count_boost = min(similar_count / 2.0, 1.0)  # Cap at 2 similar ellipses
            similarity_boost = count_boost * 0.3
        else:
            avg_similarity = 0.0
            similarity_boost = 0.0
        
        # PRIMARY: Location proximity. SECONDARY: Similarity and rank
        # Outliers get heavily penalized
        outlier_penalty = 0.3 if is_outlier else 1.0
        
        confidence = (location_score * 0.6 +
                     (0.25 * avg_similarity + 0.15 * e1.get("rank", 0.0)) +
                     similarity_boost)
        confidence = confidence * outlier_penalty
        confidence = min(max(confidence, 0.0), 1.0)
        
        e1["cup_confidence"] = confidence
        e1["is_outlier"] = is_outlier
        e1["close_neighbors"] = close_neighbors
    
    return ells


def compute_cluster_scores(ells, radius_px=180.0):
    """
    Score each ellipse by how close it is to other ellipses (0-1).
    Higher score means the cup sits in a denser local cluster.
    Y-location is weighted more heavily (1.5x) for better vertical targeting.
    Expects global coords in `gx`,`gy`.
    """
    if not ells:
        return []

    raw = []
    for i, e1 in enumerate(ells):
        s = 0.0
        for j, e2 in enumerate(ells):
            if i == j:
                continue
            dx = e1["gx"] - e2["gx"]
            dy = e1["gy"] - e2["gy"]
            # Weight y-distance 1.5x more than x for better vertical accuracy
            d = np.sqrt(dx**2 + (1.5 * dy)**2)
            if d <= radius_px:
                s += (1.0 - (d / radius_px))
        raw.append(s)

    max_raw = max(raw) if raw else 0.0
    if max_raw <= 1e-6:
        return [0.0 for _ in raw]
    return [r / max_raw for r in raw]


def choose_best_target(ells, target_state):
    """
    Choose best current target, preferring:
    1) high cup confidence,
    2) dense nearby cup cluster,
    3) continuity with last chosen target.
    Returns (selected_index, selected_score).
    """
    if not ells:
        return None, 0.0

    cluster_scores = compute_cluster_scores(ells, radius_px=TARGET_NEIGHBOR_RADIUS)

    scored = []
    for i, e in enumerate(ells):
        cup_conf = e.get("cup_confidence", 0.0)
        rank = e.get("rank", 0.0)
        cluster = cluster_scores[i]

        persist_bonus = 0.0
        if target_state["active"]:
            dx = e["gx"] - target_state["cx"]
            dy = e["gy"] - target_state["cy"]
            # Weight y-distance 1.5x more than x for better vertical accuracy
            d_prev = np.sqrt(dx**2 + (1.5 * dy)**2)
            # Only boost continuity for candidates plausibly matching the same cup.
            if d_prev <= TARGET_MATCH_MAX_DIST:
                persist_bonus = np.exp(-d_prev / TARGET_PERSIST_DIST) * target_state["strength"]

        score = 0.52 * cup_conf + 0.30 * cluster + 0.08 * rank + 0.20 * persist_bonus
        scored.append(score)

    best_idx = int(np.argmax(scored))
    best_score = scored[best_idx]

    # Hysteresis: if previous target is still visible, don't switch for tiny gains.
    if target_state["active"] and target_state["missed"] <= TARGET_MISS_TOLERANCE:
        dists = [np.hypot(e["gx"] - target_state["cx"], e["gy"] - target_state["cy"]) for e in ells]
        prev_idx = int(np.argmin(dists))
        if dists[prev_idx] <= TARGET_MATCH_MAX_DIST:
            if scored[prev_idx] + TARGET_SWITCH_MARGIN >= best_score:
                best_idx = prev_idx

    return best_idx, best_score


def update_target_state(target_state, selected_ellipse, selected_score=0.0):
    """
    Update persistent target state with selected ellipse or handle miss.
    """
    if selected_ellipse is None:
        if target_state["active"]:
            target_state["missed"] += 1
            target_state["strength"] *= 0.85
            target_state["target_score"] *= 0.90
            if target_state["missed"] > TARGET_MISS_TOLERANCE:
                target_state["active"] = False
                target_state["strength"] = 0.0
                target_state["target_score"] = 0.0
        return

    gx = float(selected_ellipse["gx"])
    gy = float(selected_ellipse["gy"])
    MA = float(selected_ellipse["MA"])
    ma = float(selected_ellipse["ma"])
    angle = float(selected_ellipse["angle"])

    if target_state["active"]:
        d = np.hypot(gx - target_state["cx"], gy - target_state["cy"])
        # If we were in hold/lost state, snap to reacquired target (no walking).
        if d <= TARGET_MATCH_MAX_DIST and target_state["missed"] == 0:
            a = 0.35  # EMA smoothing for stable aim point
            target_state["cx"] = (1.0 - a) * target_state["cx"] + a * gx
            target_state["cy"] = (1.0 - a) * target_state["cy"] + a * gy
            target_state["MA"] = (1.0 - a) * target_state["MA"] + a * MA
            target_state["ma"] = (1.0 - a) * target_state["ma"] + a * ma
            target_state["angle"] = (1.0 - a) * target_state["angle"] + a * angle
            target_state["strength"] = min(1.0, target_state["strength"] + 0.18)
        else:
            target_state["cx"] = gx
            target_state["cy"] = gy
            target_state["MA"] = MA
            target_state["ma"] = ma
            target_state["angle"] = angle
            target_state["strength"] = 0.65
    else:
        target_state["active"] = True
        target_state["cx"] = gx
        target_state["cy"] = gy
        target_state["MA"] = MA
        target_state["ma"] = ma
        target_state["angle"] = angle
        target_state["strength"] = 0.65

    target_state["target_score"] = float(selected_score)
    target_state["missed"] = 0

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
    Returns (x0,y0,x1,y1, red_mask, found_red) for the biggest red blob.
    If nothing solid found, returns full-frame ROI and found_red=False.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # red wraps hue, so use two bands - tightened to exclude skin tones
    # Solo cups are much more saturated and darker than skin
    lower1 = np.array([0, 150, 80], dtype=np.uint8)
    upper1 = np.array([8, 255, 255], dtype=np.uint8)
    lower2 = np.array([172, 150, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red = cv2.bitwise_or(m1, m2)

    # clean up
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, ker, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, ker, iterations=2)
    
    # Merge nearby blobs by dilating before finding contours
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    red_merged = cv2.dilate(red, merge_kernel, iterations=2)

    cnts, _ = cv2.findContours(red_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, w, h, red, False

    # Filter by minimum area and collect valid contours
    valid_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not valid_cnts:
        return 0, 0, w, h, red, False

    # Combine all valid blobs into a single bounding box
    all_points = np.vstack(valid_cnts)
    x, y, rw, rh = cv2.boundingRect(all_points)
    
    # Shift ROI center upward (cups' rims are above the red blob)
    vertical_shift = int(rh * 0.3)  # Shift up by 30% of blob height
    
    x0 = x - pad
    y0 = y - pad - vertical_shift
    x1 = x + rw + pad
    y1 = y + rh + pad - vertical_shift
    x0, y0, x1, y1 = clamp_roi(x0, y0, x1, y1, w, h)
    return x0, y0, x1, y1, red, True


def load_trackbar_settings(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}

def save_trackbar_settings(path, values):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2, sort_keys=True)
    except OSError:
        pass

TRACKBAR_MAX = {
    "Support Thresh x100": 100,
    "Dilate Iterations": 10,
    "Canny Low": 255,
    "Min Major Axis": 250,
    "Min Minor Axis": 120,
    "Max Minor Axis": 250,
    "Max Axis": 450,
    "Min Aspect x100": 100,
    "Max Aspect x100": 100,
    "Mask Lower": 255,
    "Fallback Supp x100": 100,
    "Min Inlier x100": 100,
    "Strict Inlier x100": 100,
    "Min Inlier Pts": 300,
    "Inlier Tol x100": 60,
    "Min Cnt Pts": 200,
    "Bridge Iter": 6,
    "NMS Ctr Dist": 120,
    "Use Red ROI": 1,
    "Red Pad": 200,
    "Red MinArea": 200,
    "Cup Confidence x100": 100,
}

def apply_trackbar_settings(settings):
    for name, maxv in TRACKBAR_MAX.items():
        if name not in settings:
            continue
        try:
            val = int(settings[name])
        except (TypeError, ValueError):
            continue
        val = max(0, min(maxv, val))
        cv2.setTrackbarPos(name, "Tuning", val)

def collect_trackbar_settings():
    values = {}
    for name in TRACKBAR_MAX.keys():
        values[name] = cv2.getTrackbarPos(name, "Tuning")
    return values

# ---------- UI ----------
cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuning", 520, 720)

cv2.createTrackbar("Support Thresh x100", "Tuning", 60, 100, lambda x: None)
cv2.createTrackbar("Dilate Iterations", "Tuning", 3, 10, lambda x: None)
cv2.createTrackbar("Canny Low", "Tuning", 80, 255, lambda x: None)
cv2.createTrackbar("Min Major Axis", "Tuning", 40, 250, lambda x: None)
cv2.createTrackbar("Min Minor Axis", "Tuning", 12, 120, lambda x: None)
cv2.createTrackbar("Max Minor Axis", "Tuning", 120, 250, lambda x: None)
cv2.createTrackbar("Max Axis", "Tuning", 180, 450, lambda x: None)
cv2.createTrackbar("Min Aspect x100", "Tuning", 18, 100, lambda x: None)
cv2.createTrackbar("Max Aspect x100", "Tuning", 100, 100, lambda x: None)
cv2.createTrackbar("Mask Lower", "Tuning", 0, 255, lambda x: None)
cv2.createTrackbar("Fallback Supp x100", "Tuning", 28, 100, lambda x: None)
cv2.createTrackbar("Min Inlier x100", "Tuning", 52, 100, lambda x: None)
cv2.createTrackbar("Strict Inlier x100", "Tuning", 72, 100, lambda x: None)
cv2.createTrackbar("Min Inlier Pts", "Tuning", 30, 300, lambda x: None)
cv2.createTrackbar("Inlier Tol x100", "Tuning", 18, 60, lambda x: None)
cv2.createTrackbar("Min Cnt Pts", "Tuning", 24, 200, lambda x: None)
cv2.createTrackbar("Bridge Iter", "Tuning", 1, 6, lambda x: None)
cv2.createTrackbar("NMS Ctr Dist", "Tuning", 28, 120, lambda x: None)

# new ROI controls
cv2.createTrackbar("Use Red ROI", "Tuning", 1, 1, lambda x: None)
cv2.createTrackbar("Red Pad", "Tuning", 70, 200, lambda x: None)
cv2.createTrackbar("Red MinArea", "Tuning", 15, 200, lambda x: None)  # *100

# Cup confidence threshold
cv2.createTrackbar("Cup Confidence x100", "Tuning", 30, 100, lambda x: None)

apply_trackbar_settings(load_trackbar_settings(SETTINGS_PATH))

target_state = {
    "active": False,
    "cx": 0.0,
    "cy": 0.0,
    "MA": 0.0,
    "ma": 0.0,
    "angle": 0.0,
    "strength": 0.0,
    "missed": 0,
    "target_score": 0.0,
}

while True:
    if USE_TEST_IMAGES:
        frame = cv2.imread(test_images[image_index])
        if frame is None:
            print(f"Failed to load image: {test_images[image_index]}")
            break
        ret = True
    else:
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
        x0, y0, x1, y1, redmask, red_found = biggest_red_roi(frame, pad=red_pad, min_area=red_min_area)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
        redmask = np.zeros((H, W), dtype=np.uint8)
        red_found = True

    # Only run heavy analysis when red is found (when ROI mode is enabled).
    analyze_enabled = (not use_roi) or red_found

    # Apply your mask (global), then crop for edge/ellipse work
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    roi_frame = masked_frame[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]

    # ------- trackbar values -------
    SUPPORT_THRESH = cv2.getTrackbarPos("Support Thresh x100", "Tuning") / 100.0
    DILATE_ITER = cv2.getTrackbarPos("Dilate Iterations", "Tuning")
    CANNY_LO = cv2.getTrackbarPos("Canny Low", "Tuning")
    MIN_MAJOR_AXIS = cv2.getTrackbarPos("Min Major Axis", "Tuning")
    MIN_MINOR_AXIS = cv2.getTrackbarPos("Min Minor Axis", "Tuning")
    MAX_MINOR_AXIS = max(MIN_MINOR_AXIS + 1, cv2.getTrackbarPos("Max Minor Axis", "Tuning"))
    MAX_AXIS = max(MIN_MAJOR_AXIS + 1, cv2.getTrackbarPos("Max Axis", "Tuning"))
    MIN_ASPECT = cv2.getTrackbarPos("Min Aspect x100", "Tuning") / 100.0
    MAX_ASPECT = max(MIN_ASPECT, cv2.getTrackbarPos("Max Aspect x100", "Tuning") / 100.0)
    MIN_SUPPORT_FALLBACK = cv2.getTrackbarPos("Fallback Supp x100", "Tuning") / 100.0
    MIN_INLIER_SCORE = cv2.getTrackbarPos("Min Inlier x100", "Tuning") / 100.0
    STRICT_INLIER_SCORE = cv2.getTrackbarPos("Strict Inlier x100", "Tuning") / 100.0
    MIN_INLIER_POINTS = max(5, cv2.getTrackbarPos("Min Inlier Pts", "Tuning"))
    INLIER_TOL = max(0.01, cv2.getTrackbarPos("Inlier Tol x100", "Tuning") / 100.0)
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

    roi_h = max(1, y1 - y0)
    roi_w = max(1, x1 - x0)
    edges = np.zeros((roi_h, roi_w), dtype=np.uint8)
    edges_stable = np.zeros((roi_h, roi_w), dtype=np.uint8)
    edges_fit = np.zeros((roi_h, roi_w), dtype=np.uint8)
    kept = []
    best_target_idx = None

    if analyze_enabled:
        # ------- run edges ONLY on ROI -------
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)

        edges = cv2.Canny(blur, CANNY_LO, CANNY_HI)
        edges_stable = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K3, iterations=1)
        edges_stable = cv2.dilate(edges_stable, K3, iterations=DILATE_ITER)

        # fit map: extra bridging for broken arcs; support still checked on edges_stable
        # Use both dilate and close to bridge gaps more effectively
        if BRIDGE_ITER > 0:
            # First: dilate to bridge small gaps
            edges_fit = cv2.dilate(edges_stable, K5, iterations=BRIDGE_ITER)
            # Then: close to connect nearby components
            edges_fit = cv2.morphologyEx(edges_fit, cv2.MORPH_CLOSE, K5, iterations=BRIDGE_ITER // 2 + 1)
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
                SUPPORT_THRESH,
            )

        # Path B: component fits (helps with merged overlapping rims)
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
                SUPPORT_THRESH,
            )

        kept = nms_ellipses(candidates)

        # Compute cup confidence for all kept ellipses
        kept = compute_cup_confidence(kept)

        # Get cup confidence threshold
        CUP_CONFIDENCE_THRESH = cv2.getTrackbarPos("Cup Confidence x100", "Tuning") / 100.0

        # Filter by cup confidence threshold
        kept = [e for e in kept if e.get("cup_confidence", 0) >= CUP_CONFIDENCE_THRESH]

        for e in kept:
            e["gx"] = e["cx"] + x0
            e["gy"] = e["cy"] + y0

        best_target_idx, best_target_score = choose_best_target(kept, target_state)
        selected = kept[best_target_idx] if best_target_idx is not None else None
        update_target_state(target_state, selected, best_target_score)
    else:
        # No red ROI detected: skip expensive full-image analysis and just hold target briefly.
        update_target_state(target_state, None, 0.0)

    # Create visualization for ROI edges with ellipses overlay
    edges_vis = cv2.cvtColor(edges_stable, cv2.COLOR_GRAY2BGR)

    # Draw ellipses back in full-frame coords
    for idx, e in enumerate(kept):
        cx_roi, cy_roi = e["cx"], e["cy"]
        cx = cx_roi + x0
        cy = cy_roi + y0

        MA, ma, angle = e["MA"], e["ma"], e["angle"]
        
        # Use different color for selected persistent target
        is_target = (best_target_idx is not None and idx == best_target_idx)
        ellipse_color = (255, 0, 255) if is_target else (0, 255, 0)  # Purple target, green others

        # Draw on full frame
        cv2.ellipse(out, ((cx, cy), (MA, ma), angle), ellipse_color, 2)
        cv2.circle(out, (int(cx), int(cy)), 4 if is_target else 3, ellipse_color, -1)
        
        # Draw on ROI edges visualization (in ROI coordinates)
        cv2.ellipse(edges_vis, ((cx_roi, cy_roi), (MA, ma), angle), ellipse_color, 2)
        cv2.circle(edges_vis, (int(cx_roi), int(cy_roi)), 4 if is_target else 3, ellipse_color, -1)

    # Persistent aim marker (survives short dropouts)
    if target_state["active"]:
        tx = int(round(target_state["cx"]))
        ty = int(round(target_state["cy"]))
        color = (0, 255, 255) if target_state["missed"] == 0 else (0, 165, 255)
        cv2.drawMarker(out, (tx, ty), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        status = "LOCK" if target_state["missed"] == 0 else f"HOLD {target_state['missed']}/{TARGET_MISS_TOLERANCE}"
        cv2.putText(out, f"TARGET SCORE: {target_state['target_score']:.2f}  {status}", (tx + 10, ty + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Display: show ROI edge maps but also a full-frame overlay
    cv2.imshow("ellipses", out)
    cv2.imshow("edges_with_fits", edges_vis)
    cv2.imshow("edges_canny_roi", edges)
    cv2.imshow("edges_stable_roi", edges_stable)
    cv2.imshow("edges_fit_roi", edges_fit)
    cv2.imshow("Mask", mask)
    cv2.imshow("RedMask", redmask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        save_trackbar_settings(SETTINGS_PATH, collect_trackbar_settings())
        break
    elif key == ord(' ') and USE_TEST_IMAGES:
        image_index = (image_index + 1) % len(test_images)
        print(f"Image {image_index}/{len(test_images)-1}: {test_images[image_index]}")

if cap is not None:
    cap.release()
save_trackbar_settings(SETTINGS_PATH, collect_trackbar_settings())
cv2.destroyAllWindows()