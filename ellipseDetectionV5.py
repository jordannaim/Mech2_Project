import numpy as np
import cv2
import json
import os
import glob

# ---- Use test images instead of camera feed ----
USE_TEST_IMAGES = True

if USE_TEST_IMAGES:
    test_image_dir = "test_images"
    test_images = sorted(glob.glob(os.path.join(test_image_dir, "*.jpg")))
    if not test_images:
        print(f"No images found in {test_image_dir}")
        exit(1)
    image_index = 0
    print(f"Loaded {len(test_images)} test images")
    cap = None
else:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    #cap.set(cv2.CAP_PROP_FPS,15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

# Pyramid detection
PYRAMID_ROW_TOLERANCE = 25  # pixels: how close ellipse centers must be to group into a row
PYRAMID_MIN_ROWS = 2  # minimum rows to form a valid pyramid
PYRAMID_SIZE_TOLERANCE = 0.15  # relative tolerance for ellipse sizes in same row

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

def maybe_add_candidate(candidates, points_xy, edges_stable, cx, cy, MA, ma, angle, ring_thick, support_thresh):
    major_axis = max(MA, ma)
    minor_axis = min(MA, ma)

    if major_axis < MIN_MAJOR_AXIS or minor_axis < MIN_MINOR_AXIS or major_axis > MAX_AXIS:
        return

    aspect = minor_axis / major_axis
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return
    
    # Filter for nearly horizontal major axis (within 25 degrees of 0 or 180)
    otherAngle = (angle + 90) % 180  # angle of minor axis
    angle_to_horizontal = min(abs(otherAngle), abs(180 - otherAngle))
    if angle_to_horizontal > 15:
        return

    # Use thicker ring for support scoring to better detect partial ellipses
    sc = support_score(edges_stable, cx, cy, MA, ma, angle, thick=ring_thick + 2, n=SAMPLE_N)
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

def find_anchor_cup(ellipses):
    """
    Find the anchor cup (single cup at front = highest Y value = closest to viewer).
    In image coords, Y increases downward, so front cup appears at bottom of frame.
    Returns the ellipse that's most likely to be the front cup, or None.
    """
    if not ellipses:
        return None
    
    # The anchor is the cup closest to viewer = HIGHEST Y value (bottom of frame)
    return max(ellipses, key=lambda e: e["cy"])

def generate_pyramid_positions(anchor, num_rows=4):
    """
    Generate expected cup positions for a full pyramid based on anchor ellipse.
    The anchor ellipse shape (aspect ratio, rotation) is preserved in extrapolation.
    Pyramid goes AWAY from viewer (Y decreases) with perspective scaling.
    Anchor at bottom (high Y), pyramid extends upward (lower Y).
    """
    if anchor is None:
        return []
    
    anchor_x = anchor["cx"]
    anchor_y = anchor["cy"]
    anchor_MA = anchor["MA"]
    anchor_ma = anchor["ma"]
    anchor_angle = anchor["angle"]
    
    # Use actual major and minor axis dimensions for spacing
    # Preserves aspect ratio of cup ellipses
    spacing_x = anchor_MA * 0.95  # Horizontal spacing based on major axis
    spacing_y = anchor_ma * 0.85  # Vertical spacing based on minor axis
    
    positions = []
    
    # Row 0: Anchor cup (1 cup at front/bottom)
    positions.append({
        "cx": anchor_x,
        "cy": anchor_y,
        "MA": anchor_MA,
        "ma": anchor_ma,
        "angle": anchor_angle,
        "row": 0,
        "col": 0,
    })
    
    # Generate subsequent rows with perspective scaling
    # Cups farther away appear smaller (perspective foreshortening)
    for row_idx in range(1, num_rows):
        row_count = row_idx + 1  # Row 1 has 2 cups, row 2 has 3, etc.
        # Go UPWARD (Y decreases) as we go away from viewer
        row_y = anchor_y - row_idx * spacing_y
        
        # Perspective scaling: cups farther away are smaller
        # Typical perspective ratio: ~0.8-0.9 per row
        scale_factor = 0.85 ** row_idx
        
        scaled_MA = anchor_MA * scale_factor
        scaled_ma = anchor_ma * scale_factor
        
        # Center the row horizontally around anchor_x
        row_width = (row_count - 1) * spacing_x * scale_factor
        start_x = anchor_x - row_width / 2.0
        
        for cup_idx in range(row_count):
            cup_x = start_x + cup_idx * spacing_x * scale_factor
            
            positions.append({
                "cx": cup_x,
                "cy": row_y,
                "MA": scaled_MA,
                "ma": scaled_ma,
                "angle": anchor_angle,  # Same rotation as anchor
                "row": row_idx,
                "col": cup_idx,
            })
    
    return positions

def match_cups_to_positions(ellipses, predicted_positions, tolerance=30):
    """
    Match detected ellipses to predicted pyramid positions.
    
    Args:
        ellipses: list of detected ellipse dicts
        predicted_positions: list of predicted position dicts from generate_pyramid_positions
        tolerance: max distance to match
    
    Returns:
        matched: list of matched ellipses with their predicted positions
        unmatched_positions: list of unmatched predicted positions
        unmatched_ellipses: list of detected ellipses that don't fit pyramid
    """
    matched = []
    unmatched_positions = list(predicted_positions)
    unmatched_ellipses = list(ellipses)
    
    for pred_pos in predicted_positions:
        best_match = None
        best_dist = float('inf')
        best_ellipse = None
        
        # Find closest ellipse to this predicted position
        for ellipse in unmatched_ellipses:
            dist = np.hypot(ellipse["cx"] - pred_pos["cx"], ellipse["cy"] - pred_pos["cy"])
            if dist < tolerance and dist < best_dist:
                best_match = pred_pos
                best_dist = dist
                best_ellipse = ellipse
        
        if best_ellipse:
            matched.append({
                "predicted": best_match,
                "detected": best_ellipse,
                "distance": best_dist,
            })
            unmatched_ellipses.remove(best_ellipse)
            unmatched_positions.remove(pred_pos)
    
    return matched, unmatched_positions, unmatched_ellipses

def check_white_center(frame, cx, cy, radius=10):
    """
    Check if the center region is predominantly white (cup interior).
    Returns True if white, False if red/other color.
    """
    h, w = frame.shape[:2]
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius))
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius))
    
    if x1 <= x0 or y1 <= y0:
        return False
    
    region = frame[y0:y1, x0:x1]
    
    # Convert to HSV
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Check for white: low saturation, high value
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_region, lower_white, upper_white)
    
    # Check for red (should be absent)
    lower_red1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 100, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    
    red_mask1 = cv2.inRange(hsv_region, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_region, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    white_ratio = np.count_nonzero(white_mask) / float(white_mask.size)
    red_ratio = np.count_nonzero(red_mask) / float(red_mask.size)
    
    # Accept if mostly white and not much red
    return white_ratio > 0.4 and red_ratio < 0.2

def search_for_cup_at_position(edges_stable, frame, pred_pos, search_radius=40):
    """
    Search for an ellipse at a specific predicted position.
    
    Args:
        edges_stable: edge image
        frame: original frame
        pred_pos: dict with {cx, cy, MA, ma, angle} of predicted position
        search_radius: how far to search around predicted position
    
    Returns:
        ellipse dict if found and validated, None otherwise.
    """
    h, w = edges_stable.shape
    pred_x = pred_pos["cx"]
    pred_y = pred_pos["cy"]
    pred_size = np.sqrt(pred_pos["MA"] * pred_pos["ma"])
    
    # Extract region around predicted position
    x0 = max(0, int(pred_x - search_radius))
    x1 = min(w, int(pred_x + search_radius))
    y0 = max(0, int(pred_y - search_radius))
    y1 = min(h, int(pred_y + search_radius))
    
    if x1 <= x0 + 10 or y1 <= y0 + 10:
        return None
    
    region_edges = edges_stable[y0:y1, x0:x1]
    
    # Find contours in this region
    cnts, _ = cv2.findContours(region_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    best_ellipse = None
    best_score = 0
    
    for cnt in cnts:
        if len(cnt) < 5:
            continue
        
        try:
            ((ec_x, ec_y), (ec_MA, ec_ma), ec_angle) = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        
        # Convert to global coordinates
        global_cx = ec_x + x0
        global_cy = ec_y + y0
        
        # Check if this ellipse is close to predicted position
        dist = np.hypot(global_cx - pred_x, global_cy - pred_y)
        if dist > search_radius * 0.7:
            continue
        
        # Check size similarity
        ellipse_size = np.sqrt(ec_MA * ec_ma)
        size_ratio = min(ellipse_size, pred_size) / max(ellipse_size, pred_size)
        if size_ratio < 0.7:
            continue
        
        # Check aspect ratio (should be elliptical)
        aspect = min(ec_MA, ec_ma) / max(ec_MA, ec_ma)
        if aspect < 0.15 or aspect > 1.0:
            continue
        
        # Validate white center
        if not check_white_center(frame, global_cx, global_cy, radius=int(pred_size * 0.3)):
            continue
        
        # Score based on proximity and size match
        score = size_ratio * (1.0 - dist / search_radius)
        
        if score > best_score:
            best_score = score
            best_ellipse = {
                "cx": global_cx,
                "cy": global_cy,
                "MA": ec_MA,
                "ma": ec_ma,
                "angle": ec_angle,
                "predicted": True,
                "confidence": score,
            }
    
    return best_ellipse if best_score > 0.5 else None

def cluster_ellipses_by_row(ellipses, row_tolerance=25):
    """
    Group ellipses into horizontal rows based on Y position.
    Returns list of rows, where each row is a list of ellipses sorted by X position.
    """
    if not ellipses:
        return []
    
    # Sort by Y coordinate (vertical position)
    sorted_ellipses = sorted(ellipses, key=lambda e: e["cy"])
    
    rows = []
    current_row = [sorted_ellipses[0]]
    
    for e in sorted_ellipses[1:]:
        # Check if this ellipse belongs to the current row
        if abs(e["cy"] - current_row[0]["cy"]) <= row_tolerance:
            current_row.append(e)
        else:
            # Start a new row
            rows.append(sorted(current_row, key=lambda x: x["cx"]))
            current_row = [e]
    
    # Don't forget the last row
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x["cx"]))
    
    return rows

def validate_pyramid_structure(rows):
    """
    Validate if the rows form a reasonable pyramid:
    - Each row should have <= previous row (decreasing count)
    - Cup sizes should be similar within acceptable tolerance
    - Should have at least PYRAMID_MIN_ROWS rows
    Returns (is_valid, structure_score)
    """
    if len(rows) < PYRAMID_MIN_ROWS:
        return False, 0.0
    
    # Check size consistency within each row
    size_consistency = []
    for row in rows:
        if len(row) == 1:
            size_consistency.append(1.0)
        else:
            sizes = [np.sqrt(e["MA"] * e["ma"]) for e in row]
            mean_size = np.mean(sizes)
            size_std = np.std(sizes) / mean_size
            consistency = 1.0 - np.clip(size_std, 0, PYRAMID_SIZE_TOLERANCE) / PYRAMID_SIZE_TOLERANCE
            size_consistency.append(consistency)
    
    # Check if row sizes INCREASE as we go away (inverted pyramid - beer pong style)
    # Viewer sees 1 cup at bottom (closest), then 2, 3, 4 as you go away
    row_counts = [len(row) for row in rows]
    is_increasing = True
    for i in range(len(row_counts) - 1):
        if row_counts[i + 1] < row_counts[i]:
            is_increasing = False
            break
    
    if not is_increasing:
        return False, 0.0, []
    
    # Compute structure score
    consistency_score = np.mean(size_consistency)
    
    # Expected pattern for standard beer pong: 1, 2, 3, 4 from closest to farthest
    expected_counts = list(range(1, len(rows) + 1))
    
    count_diff = sum(abs(row_counts[i] - expected_counts[i]) for i in range(len(rows)))
    count_score = 1.0 - (count_diff / sum(expected_counts))
    count_score = max(0.0, count_score)
    
    structure_score = 0.6 * consistency_score + 0.4 * count_score
    
    return is_increasing, structure_score, expected_counts

def get_pyramid_structure_str(rows):
    """Return string description of pyramid structure."""
    if not rows:
        return "No pyramid"
    counts = [str(len(row)) for row in rows]
    return " -> ".join(counts) + " (closest to farthest)"

def predict_missing_cups(rows, expected_counts, min_integrity=0.65):
    """
    Predict where missing cups should be based on pyramid structure.
    Uses each cup's own major/minor axis dimensions for spacing.
    
    Args:
        rows: List of detected cup rows (each row is list of ellipses)
        expected_counts: Expected cup counts per row [1, 2, 3, 4, ...]
        min_integrity: Min % of expected cups needed to make predictions (0.65 = 65%)
    
    Returns:
        List of predicted cup positions: [{"x": cx, "y": cy, "confidence": score}, ...]
    """
    if not rows or not expected_counts:
        return []
    
    # Calculate pyramid integrity
    actual_counts = [len(row) for row in rows]
    total_expected = sum(expected_counts)
    total_actual = sum(actual_counts)
    integrity = total_actual / float(total_expected)
    
    if integrity < min_integrity:
        # Pyramid too degraded, don't make predictions
        return []
    
    predictions = []
    
    for row_idx, row in enumerate(rows):
        if row_idx >= len(expected_counts):
            break
        
        expected_in_row = expected_counts[row_idx]
        actual_in_row = len(row)
        missing = expected_in_row - actual_in_row
        
        if missing <= 0:
            continue  # No missing cups in this row
        
        # Get row Y position (average of detected cups in row)
        row_y = np.mean([e["cy"] for e in row])
        
        # Get row X positions (horizontal spacing)
        row_xs = sorted([e["cx"] for e in row])
        
        # Estimate missing cup positions by interpolation
        if len(row_xs) == 0:
            continue
        
        # Calculate spacing from detected cup dimensions directly
        # Use horizontal (major-ish) axis with tight spacing multiplier
        if len(row) > 0:
            # Use each cup's dimensions - average the spacing multipliers
            spacings = [np.sqrt(e["MA"] * e["ma"]) * 0.95 for e in row]
            spacing = np.mean(spacings) if spacings else 50
        else:
            spacing = 50
        
        # Find gaps in the detected positions
        # Generate expected positions and find which ones are missing
        first_x = row_xs[0]
        for i in range(expected_in_row):
            expected_x = first_x + i * spacing
            
            # Check if a cup exists near this position
            found = False
            for detected in row:
                if abs(detected["cx"] - expected_x) < spacing * 0.4:
                    found = True
                    break
            
            if not found:
                # This position is missing a cup
                # Confidence decreases with more missing cups
                confidence = 1.0 - (missing / float(expected_in_row)) * 0.3
                confidence = np.clip(confidence, 0.5, 0.95)
                
                predictions.append({
                    "x": expected_x,
                    "y": row_y,
                    "confidence": confidence,
                    "row": row_idx,
                })
    
    return predictions

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
        return 0, 0, w, h, red

    # Filter by minimum area and collect valid contours
    valid_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not valid_cnts:
        return 0, 0, w, h, red

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
    return x0, y0, x1, y1, red


def biggest_blue_roi(frame_bgr, pad=60, min_area=1500):
    """
    Returns (x0,y0,x1,y1, blue_mask) for the biggest blue blob.
    If nothing solid found, returns full-frame ROI.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Blue range (primary band) - tightened for better isolation
    lower_blue = np.array([100, 150, 80], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # clean up
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, ker, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, ker, iterations=2)
    
    # Merge nearby blobs by dilating before finding contours
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    blue_merged = cv2.dilate(blue_mask, merge_kernel, iterations=2)

    cnts, _ = cv2.findContours(blue_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, w, h, blue_mask

    # Filter by minimum area and collect valid contours
    valid_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not valid_cnts:
        return 0, 0, w, h, blue_mask

    # Combine all valid blobs into a single bounding box
    all_points = np.vstack(valid_cnts)
    x, y, rw, rh = cv2.boundingRect(all_points)
    
    # Shift ROI center upward (cups' rims are above the blue blob)
    vertical_shift = int(rh * 0.3)  # Shift up by 30% of blob height
    
    x0 = x - pad
    y0 = y - pad - vertical_shift
    x1 = x + rw + pad
    y1 = y + rh + pad - vertical_shift
    x0, y0, x1, y1 = clamp_roi(x0, y0, x1, y1, w, h)
    return x0, y0, x1, y1, blue_mask

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
    "Ring Thick": 20,
    "Support Thresh x100": 100,
    "Dilate Iterations": 10,
    "Canny Low": 255,
    "Min Major Axis": 250,
    "Min Minor Axis": 120,
    "Max Axis": 450,
    "Min Aspect x100": 100,
    "Mask Lower": 255,
    "Fallback Supp x100": 100,
    "Min Inlier x100": 100,
    "Strict Inlier x100": 100,
    "Min Inlier Pts": 300,
    "Inlier Tol x100": 60,
    "Rand Fit Trials": 80,
    "Rand Fit Points": 220,
    "Min Cnt Pts": 200,
    "Bridge Iter": 6,
    "NMS Ctr Dist": 120,
    "Use Red ROI": 1,
    "Red Pad": 200,
    "Red MinArea": 200,
    "Pyr Row Tol": 50,
    "Pyr Size Tol x100": 15,    "Pyr Min Integrity x100": 65,}

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

# Pyramid detection
cv2.createTrackbar("Pyr Row Tol", "Tuning", 25, 50, lambda x: None)
cv2.createTrackbar("Pyr Size Tol x100", "Tuning", 15, 50, lambda x: None)
cv2.createTrackbar("Pyr Min Integrity x100", "Tuning", 65, 100, lambda x: None)

apply_trackbar_settings(load_trackbar_settings(SETTINGS_PATH))

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
    
    # Pyramid parameters from trackbars
    PYRAMID_ROW_TOLERANCE = cv2.getTrackbarPos("Pyr Row Tol", "Tuning")
    PYRAMID_SIZE_TOLERANCE = cv2.getTrackbarPos("Pyr Size Tol x100", "Tuning") / 100.0
    PYRAMID_MIN_INTEGRITY = cv2.getTrackbarPos("Pyr Min Integrity x100", "Tuning") / 100.0

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
        
        # NEW: Extra random trials for partial arc detection
        # Try smaller random subsets to find good partial arcs
        if len(pts_xy) >= RANDOM_FIT_POINTS * 0.6:
            for _ in range(RANDOM_FIT_TRIALS // 2):
                # Use smaller subset (60% of normal) to find best partial fit
                subset_size = max(5, int(RANDOM_FIT_POINTS * 0.6))
                idx = np.random.choice(len(pts_xy), subset_size, replace=False)
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
    
    # NEW APPROACH: Anchor-based pyramid detection
    # 1. Find the anchor cup (front single cup)
    anchor = find_anchor_cup(kept)
    
    # 2. Generate expected pyramid positions based on anchor
    predicted_positions = []
    matched_cups = []
    unmatched_positions = []
    found_cups = []
    
    if anchor:
        predicted_positions = generate_pyramid_positions(anchor, num_rows=4)
        
        # 3. Match detected cups to predicted positions
        matched_cups, unmatched_positions, unmatched_ellipses = match_cups_to_positions(
            kept, predicted_positions, tolerance=40
        )
        
        # 4. Search for cups at unmatched positions (validate with white center check)
        for pred_pos in unmatched_positions:
            # Search in edges for cup at this position
            found = search_for_cup_at_position(
                edges_stable, 
                roi_frame, 
                pred_pos, 
                search_radius=int(np.sqrt(pred_pos["MA"] * pred_pos["ma"]) * 0.8)
            )
            if found:
                found_cups.append(found)

    # Create visualization for ROI edges with ellipses overlay
    edges_vis = cv2.cvtColor(edges_stable, cv2.COLOR_GRAY2BGR)

    # Draw detected ellipses (green)
    for e in kept:
        cx_roi, cy_roi = e["cx"], e["cy"]
        cx = cx_roi + x0
        cy = cy_roi + y0

        MA, ma, angle = e["MA"], e["ma"], e["angle"]

        # Draw on full frame
        cv2.ellipse(out, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        arc_str = f' a:{e.get("arc_score", 0):.2f}' if "arc_score" in e else ''
        cv2.putText(out, f's:{e["score"]:.2f} i:{e["inlier"]:.2f}{arc_str}', (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Draw on ROI edges visualization (in ROI coordinates)
        cv2.ellipse(edges_vis, ((cx_roi, cy_roi), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(edges_vis, (int(cx_roi), int(cy_roi)), 3, (0, 0, 255), -1)
    
    # Draw anchor cup with special marking (yellow ellipse)
    if anchor:
        anchor_x = int(anchor["cx"]) + x0
        anchor_y = int(anchor["cy"]) + y0
        cv2.ellipse(out, ((anchor_x, anchor_y), (int(anchor["MA"]), int(anchor["ma"])), anchor["angle"]), (0, 255, 255), 3)
        cv2.putText(out, "ANCHOR", (anchor_x - 25, anchor_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw predicted pyramid positions (cyan ellipses showing expected cups)
    for pred_pos in predicted_positions:
        pred_x = int(pred_pos["cx"]) + x0
        pred_y = int(pred_pos["cy"]) + y0
        pred_MA = int(pred_pos["MA"])
        pred_ma = int(pred_pos["ma"])
        pred_angle = pred_pos["angle"]
        
        # Check if this position is matched
        is_matched = any(match["predicted"]["cx"] == pred_pos["cx"] and 
                        match["predicted"]["cy"] == pred_pos["cy"] 
                        for match in matched_cups)
        
        if is_matched:
            # Already has a detected cup - draw small green marker
            cv2.circle(out, (pred_x, pred_y), 4, (0, 255, 0), -1)
        else:
            # Missing cup position - draw cyan ellipse outline
            cv2.ellipse(out, ((pred_x, pred_y), (pred_MA, pred_ma), pred_angle), (255, 255, 0), 1)
            cv2.circle(out, (pred_x, pred_y), 3, (255, 255, 0), -1)
    
    # Draw found cups (magenta - cups found by search at predicted positions)
    for found in found_cups:
        fx = int(found["cx"]) + x0
        fy = int(found["cy"]) + y0
        fMA = found["MA"]
        fma = found["ma"]
        fangle = found["angle"]
        conf = found.get("confidence", 0)
        
        # Draw with magenta color
        cv2.ellipse(out, ((fx, fy), (fMA, fma), fangle), (255, 0, 255), 2)
        cv2.circle(out, (fx, fy), 3, (255, 0, 255), -1)
        cv2.putText(out, f'FOUND:{conf:.2f}', (fx + 6, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    
    # Display pyramid info
    if anchor and predicted_positions:
        total_expected = len(predicted_positions)
        total_detected = len(matched_cups) + len(found_cups)
        info_str = f"Pyramid: {total_detected}/{total_expected} cups | Found: {len(found_cups)} via search"
        cv2.putText(out, info_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(out, "No anchor cup detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
