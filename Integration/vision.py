"""
vision.py — Cup detection pipeline for the beer pong turret.

Approach (from ellipseDetectionV6.py):
  1. Red HSV mask to find ROI around cups
  2. Canny edge detection in ROI
  3. Ellipse fitting on contours
  4. Score and pick best candidate
  5. Distance estimate via pinhole model using known cup diameter

DetectionResult fields:
  x_norm    : -1.0 (left) to +1.0 (right), 0 = horizontally centered
  y_norm    : -1.0 (bottom) to +1.0 (top), 0 = vertically centered
  distance_m: estimated distance in meters using pinhole camera model
  confidence: 0.0–1.0
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard red solo cup rim diameter in millimeters
KNOWN_CUP_DIAMETER_MM = 90.0

# Baseline tuning resolution (ellipse_tuning.json values were tuned around 720p)
_REF_W = 1920
_REF_H = 1080

# Morphology kernel
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ---------------------------------------------------------------------------
# Load tuning parameters from ellipse_tuning.json (teammate-tuned values).
# Falls back to safe defaults if the file is missing or malformed.
# ---------------------------------------------------------------------------

_TUNING_PATH = os.path.join(os.path.dirname(__file__), "ellipse_tuning.json")

def _load_tuning() -> dict:
    try:
        with open(_TUNING_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        logger.warning("ellipse_tuning.json not found or invalid — using built-in defaults")
    return {}

_T = _load_tuning()

def _t(key: str, default: float, scale: float = 1.0) -> float:
    """Read a tuning value, applying optional scale (e.g. x100 params → divide by 100)."""
    return int(_T[key]) * scale if key in _T else default

# Detection geometry (from tuning JSON)
_CANNY_LO        = int(_t("Canny Low",        37))
_CANNY_HI        = _CANNY_LO * 2
_DILATE_ITER     = int(_t("Dilate Iterations", 2))
_MIN_MAJOR       = int(_t("Min Major Axis",   48))
_MIN_MINOR       = int(_t("Min Minor Axis",   11))
_MAX_MINOR       = int(_t("Max Minor Axis",   24))
_MAX_AXIS        = int(_t("Max Axis",         87))
_MIN_ASPECT      = _t("Min Aspect x100",  20, scale=0.01)
_MAX_ASPECT      = _t("Max Aspect x100",  46, scale=0.01)
_SUPPORT_THRESH  = _t("Support Thresh x100", 38, scale=0.01)
_RED_PAD         = int(_t("Red Pad",          16))
_RED_MIN_AREA    = int(_t("Red MinArea",      52)) * 100

# Target locking / persistence
_LOCK_RADIUS     = 0.35   # max x_norm/y_norm distance to consider "same cup"
_PERSIST_FRAMES  = 5      # frames to hold last valid result when detection fails

# Camera exposure controls (Linux/V4L2/OpenCV)
# - Many UVC cameras expect CAP_PROP_AUTO_EXPOSURE=1 for manual, 3 for auto.
# - CAP_PROP_EXPOSURE units are camera-specific (often negative on Linux UVC).
_CAMERA_AUTO_EXPOSURE = False
_CAMERA_EXPOSURE = float(os.getenv("CUP_CAMERA_EXPOSURE", "-16"))
_CAMERA_GAIN = float(os.getenv("CUP_CAMERA_GAIN", "0"))
_CAMERA_BRIGHTNESS = float(os.getenv("CUP_CAMERA_BRIGHTNESS", "0"))


def _set_cap_prop(cap: cv2.VideoCapture, prop: int, value: float, label: str) -> None:
    """Best-effort camera property set + readback logging."""
    ok = cap.set(prop, value)
    readback = cap.get(prop)
    logger.info("Camera ctrl %-16s set=%s requested=%.3f readback=%.3f", label, ok, value, readback)


def _configure_camera_image_controls(cap: cv2.VideoCapture) -> None:
    """
    Configure camera image controls in a robust order.
    Some Linux UVC webcams use auto-exposure values {1,3}, others {0.25,0.75}.
    """
    if _CAMERA_AUTO_EXPOSURE:
        _set_cap_prop(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 3.0, "auto_exposure")
        _set_cap_prop(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, "auto_exposure")
    else:
        # Force manual mode with both conventions, then apply explicit exposure.
        _set_cap_prop(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 1.0, "auto_exposure")
        _set_cap_prop(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure")
        time.sleep(0.05)
        _set_cap_prop(cap, cv2.CAP_PROP_EXPOSURE, _CAMERA_EXPOSURE, "exposure")
        time.sleep(0.05)
        _set_cap_prop(cap, cv2.CAP_PROP_EXPOSURE, _CAMERA_EXPOSURE, "exposure")

    # Reduce other auto controls that can make the frame blow out.
    _set_cap_prop(cap, cv2.CAP_PROP_AUTO_WB, 0.0, "auto_wb")
    _set_cap_prop(cap, cv2.CAP_PROP_GAIN, _CAMERA_GAIN, "gain")
    _set_cap_prop(cap, cv2.CAP_PROP_BRIGHTNESS, _CAMERA_BRIGHTNESS, "brightness")


def _scaled_detection_params(frame_w: int, frame_h: int) -> Tuple[int, int, int, int, int, int, int]:
        """
        Scale geometry thresholds from the 1280x720 baseline to current frame size.
        Returns:
            (min_major, min_minor, max_minor, max_axis, red_pad, red_min_area, line_thickness)
        """
        # Linear scale for lengths (major/minor/pad), area scale for contour area.
        sx = frame_w / float(_REF_W)
        sy = frame_h / float(_REF_H)
        s_len = (sx + sy) * 0.5
        s_area = sx * sy

        min_major = max(8, int(round(_MIN_MAJOR * s_len)))
        min_minor = max(4, int(round(_MIN_MINOR * s_len)))
        max_minor = max(min_minor + 2, int(round(_MAX_MINOR * s_len)))
        max_axis = max(min_major + 4, int(round(_MAX_AXIS * s_len)))
        red_pad = max(4, int(round(_RED_PAD * s_len)))
        red_min_area = max(200, int(round(_RED_MIN_AREA * s_area)))
        line_thickness = max(1, int(round(2 * s_len)))

        return min_major, min_minor, max_minor, max_axis, red_pad, red_min_area, line_thickness


@dataclass
class DetectionResult:
    valid: bool = False
    x_norm: float = 0.0       # horizontal offset: -1=left, +1=right
    y_norm: float = 0.0       # vertical offset:   -1=bottom, +1=top
    distance_m: float = 0.0   # estimated distance to cup
    confidence: float = 0.0
    frame_w: int = 0
    frame_h: int = 0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Low-level detection helpers
# ---------------------------------------------------------------------------

def _biggest_red_roi(
    frame_bgr: np.ndarray,
    pad: int = _RED_PAD,
    min_area: int = _RED_MIN_AREA,
) -> Tuple[int, int, int, int, bool]:
    """Return (x0, y0, x1, y1, found) bounding box of largest red region."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,   150, 80], dtype=np.uint8)
    upper1 = np.array([8,   255, 255], dtype=np.uint8)
    lower2 = np.array([172, 150, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    red = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN,  ker, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, ker, iterations=2)
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=2)

    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not valid:
        return 0, 0, w, h, False

    pts = np.vstack(valid)
    x, y, rw, rh = cv2.boundingRect(pts)
    vshift = int(rh * 0.3)

    x0 = max(0, x - pad)
    y0 = max(0, y - pad - vshift)
    x1 = min(w, x + rw + pad)
    y1 = min(h, y + rh + pad - vshift)
    return x0, y0, x1, y1, True


def _support_score(
    edges: np.ndarray,
    cx: float, cy: float,
    major: float, minor: float,
    angle_deg: float,
    n: int = 120,
) -> float:
    """Fraction of n evenly-spaced ellipse perimeter points that land on an edge."""
    a = major / 2.0
    b = minor / 2.0
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca = np.cos(np.deg2rad(angle_deg))
    sa = np.sin(np.deg2rad(angle_deg))
    xpts = ca * (a * np.cos(t)) - sa * (b * np.sin(t)) + cx
    ypts = sa * (a * np.cos(t)) + ca * (b * np.sin(t)) + cy

    h, w = edges.shape
    hits = 0
    for xf, yf in zip(xpts, ypts):
        xi, yi = int(round(xf)), int(round(yf))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        y0 = max(0, yi - 2); y1 = min(h, yi + 3)
        x0 = max(0, xi - 2); x1 = min(w, xi + 3)
        if np.any(edges[y0:y1, x0:x1] > 0):
            hits += 1
    return hits / n


def _detect_in_frame(
    frame: np.ndarray,
    focal_length_px: float,
    lock_pos: Optional[Tuple[float, float]] = None,
) -> Tuple[DetectionResult, np.ndarray]:
    """
    Run cup detection on one frame. Returns (result, annotated_frame).

    lock_pos: if provided as (x_norm, y_norm), prefer the candidate nearest
    to that position over the highest-scoring one. Falls back to best score
    if no candidates are within _LOCK_RADIUS.
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    (
        min_major,
        min_minor,
        max_minor,
        max_axis,
        red_pad,
        red_min_area,
        line_thickness,
    ) = _scaled_detection_params(w, h)

    # --- Red ROI ---
    x0, y0, x1, y1, found_red = _biggest_red_roi(frame, pad=red_pad, min_area=red_min_area)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), line_thickness)

    if not found_red:
        cv2.putText(out, "NO RED ROI", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return DetectionResult(valid=False, frame_w=w, frame_h=h), out

    # --- Edges in ROI ---
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    edges = cv2.Canny(blur, _CANNY_LO, _CANNY_HI)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _K3, iterations=1)
    edges = cv2.dilate(edges, _K3, iterations=_DILATE_ITER)

    # --- Ellipse candidates ---
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for cnt in contours:
        if len(cnt) < 12:
            continue
        try:
            (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        major = max(ma, mi)
        minor = min(ma, mi)

        if not (min_major <= major <= max_axis and min_minor <= minor <= max_minor):
            continue
        aspect = minor / major if major > 1e-6 else 0.0
        if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
            continue

        # Cup rims viewed from above always have a near-horizontal major axis.
        # Reject ellipses whose minor axis is more than 15° off horizontal.
        minor_axis_angle = (angle + 90) % 180
        if min(minor_axis_angle, 180 - minor_axis_angle) > 15:
            continue

        support = _support_score(edges, cx, cy, ma, mi, angle)
        score = 0.7 * support + 0.3 * (1.0 - abs(0.6 - aspect))
        if support < _SUPPORT_THRESH:
            continue

        # Pre-compute global normalized coords for lock-based selection
        gx_c = float(cx + x0)
        gy_c = float(cy + y0)
        xn_c = float(np.clip((gx_c - w / 2.0) / (w / 2.0), -1.0, 1.0))
        yn_c = float(np.clip((h / 2.0 - gy_c) / (h / 2.0), -1.0, 1.0))

        candidates.append((score, cx, cy, ma, mi, angle, support, xn_c, yn_c))

    if not candidates:
        cv2.putText(out, "NO TARGET", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return DetectionResult(valid=False, frame_w=w, frame_h=h), out

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Select candidate: prefer locked cup, fall back to best score
    if lock_pos is not None:
        lx, ly = lock_pos
        nearby = [c for c in candidates
                  if (c[7] - lx) ** 2 + (c[8] - ly) ** 2 < _LOCK_RADIUS ** 2]
        selected = nearby[0] if nearby else candidates[0]
    else:
        selected = candidates[0]

    # Draw all candidates (green), selected (magenta), others that are locked-out (gray)
    for i, c in enumerate(candidates[:8]):
        if c is selected:
            color = (255, 0, 255)   # magenta — the chosen one
        elif lock_pos is not None:
            color = (100, 100, 100) # gray — locked out
        else:
            color = (0, 255, 0)     # green — unfiltered
        gx_draw = int(round(c[1] + x0))
        gy_draw = int(round(c[2] + y0))
        cv2.ellipse(out, ((gx_draw, gy_draw), (c[3], c[4]), c[5]), color, line_thickness)
        cv2.circle(out, (gx_draw, gy_draw), 3, color, -1)

    best_score, cx, cy, ma, mi, angle, _, x_norm, y_norm = selected
    gx = float(cx + x0)
    gy = float(cy + y0)
    major = max(ma, mi)
    confidence = float(np.clip(best_score, 0.0, 1.0))

    # Pinhole distance estimate: D = (real_diameter * focal_px) / apparent_px
    distance_m = 0.0
    if major > 1.0:
        distance_m = (KNOWN_CUP_DIAMETER_MM / 1000.0) * focal_length_px / major

    cv2.drawMarker(out, (int(gx), int(gy)), (0, 255, 255), cv2.MARKER_CROSS, 20, line_thickness)
    cv2.putText(
        out,
        f"x={x_norm:+.3f} y={y_norm:+.3f} d={distance_m:.2f}m conf={confidence:.2f}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
    )

    return (
        DetectionResult(
            valid=True,
            x_norm=x_norm,
            y_norm=y_norm,
            distance_m=distance_m,
            confidence=confidence,
            frame_w=w,
            frame_h=h,
            timestamp=time.time(),
        ),
        out,
    )


# ---------------------------------------------------------------------------
# Thread-safe camera class
# ---------------------------------------------------------------------------

class CupDetector:
    """
    Continuously captures frames from a USB camera and runs cup detection.
    Thread-safe: call get_result() from any thread to get the latest detection.
    """

    def __init__(self, camera_index: int = 0, focal_length_px: float = 1050.0) -> None:
        """
        focal_length_px: camera focal length in pixels.
                    ~1050 is a reasonable starting point for a typical USB webcam at 1080p
                    (~700 at 720p scaled by 1080/720).
          Calibrate empirically: at a known distance, measure the cup's major
          axis in pixels and solve focal = (distance_m * major_px) / diameter_m.
        """
        self._camera_index = camera_index
        self._focal_length_px = focal_length_px

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._last_result = DetectionResult()
        self._last_frame: Optional[np.ndarray] = None

        # Target locking — set by lock_target(), cleared by unlock_target()
        self._lock_pos: Optional[Tuple[float, float]] = None

        # Persistence — keep the last valid result for up to _PERSIST_FRAMES
        # missed frames, with confidence decaying each frame
        self._last_valid_result = DetectionResult()
        self._missed_frames: int = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="cup-detector"
        )
        self._thread.start()
        logger.info("CupDetector started (camera %d)", self._camera_index)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("CupDetector stopped")

    def get_result(self) -> DetectionResult:
        with self._lock:
            return self._last_result

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def set_focal_length(self, focal_px: float) -> None:
        self._focal_length_px = focal_px

    def lock_target(self, x_norm: float, y_norm: float) -> None:
        """
        Lock detection to the cup nearest this normalized position.
        The lock position auto-updates each frame to follow the cup as the
        turret moves. Call before alignment; call unlock_target() after firing.
        """
        with self._lock:
            self._lock_pos = (x_norm, y_norm)
            self._missed_frames = 0
        logger.info("Target locked at (x=%.3f y=%.3f)", x_norm, y_norm)

    def unlock_target(self) -> None:
        """Clear the target lock so the next shot can pick a new cup."""
        with self._lock:
            self._lock_pos = None
            self._missed_frames = 0
        logger.info("Target unlocked")

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            # Request a 1080p stream from the camera.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # Let the stream initialize, then apply image controls.
            time.sleep(0.10)
            _configure_camera_image_controls(cap)

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("Camera stream resolution: %dx%d", actual_w, actual_h)

            # Quick sanity metric for overexposure (0=black, 255=white).
            ok0, f0 = cap.read()
            if ok0 and f0 is not None:
                gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
                logger.info("Initial frame mean brightness: %.1f", float(np.mean(gray0)))
        else:
            logger.error("Failed to open camera %d", self._camera_index)
            cap.release()
            return

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Frame read failed")
                time.sleep(0.05)
                continue

            with self._lock:
                lock_pos = self._lock_pos

            result, annotated = _detect_in_frame(frame, self._focal_length_px, lock_pos)

            with self._lock:
                if result.valid:
                    self._last_valid_result = result
                    self._missed_frames = 0
                    # Auto-advance the lock position to follow the cup
                    if self._lock_pos is not None:
                        self._lock_pos = (result.x_norm, result.y_norm)
                else:
                    self._missed_frames += 1
                    if self._missed_frames <= _PERSIST_FRAMES:
                        # Return last known position with decaying confidence
                        r = self._last_valid_result
                        result = DetectionResult(
                            valid=True,
                            x_norm=r.x_norm,
                            y_norm=r.y_norm,
                            distance_m=r.distance_m,
                            confidence=r.confidence * (0.8 ** self._missed_frames),
                            frame_w=r.frame_w,
                            frame_h=r.frame_h,
                        )
                self._last_result = result
                self._last_frame = annotated

        cap.release()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CupDetector":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
