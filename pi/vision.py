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

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard red solo cup rim diameter in millimeters
KNOWN_CUP_DIAMETER_MM = 90.0

# Morphology kernel
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


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
    pad: int = 60,
    min_area: int = 1500,
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
) -> Tuple[DetectionResult, np.ndarray]:
    """Run cup detection on one frame. Returns (result, annotated_frame)."""
    out = frame.copy()
    h, w = frame.shape[:2]

    # --- Red ROI ---
    x0, y0, x1, y1, found_red = _biggest_red_roi(frame)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 2)

    if not found_red:
        cv2.putText(out, "NO RED ROI", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return DetectionResult(valid=False, frame_w=w, frame_h=h), out

    # --- Edges in ROI ---
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _K3, iterations=1)
    edges = cv2.dilate(edges, _K3, iterations=2)

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

        if not (40 <= major <= 240 and 12 <= minor <= 180):
            continue
        aspect = minor / major if major > 1e-6 else 0.0
        if not (0.18 <= aspect <= 1.0):
            continue

        support = _support_score(edges, cx, cy, ma, mi, angle)
        score = 0.7 * support + 0.3 * (1.0 - abs(0.6 - aspect))
        if score < 0.30:
            continue

        candidates.append((score, cx, cy, ma, mi, angle, support))

    if not candidates:
        cv2.putText(out, "NO TARGET", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return DetectionResult(valid=False, frame_w=w, frame_h=h), out

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Draw all candidates (green), best (magenta)
    for i, (score, cx, cy, ma, mi, angle, _) in enumerate(candidates[:8]):
        color = (255, 0, 255) if i == 0 else (0, 255, 0)
        gx = int(round(cx + x0))
        gy = int(round(cy + y0))
        cv2.ellipse(out, ((gx, gy), (ma, mi), angle), color, 2)
        cv2.circle(out, (gx, gy), 3, color, -1)

    best_score, cx, cy, ma, mi, angle, _ = candidates[0]
    gx = float(cx + x0)
    gy = float(cy + y0)
    major = max(ma, mi)

    x_norm = float(np.clip((gx - w / 2.0) / (w / 2.0), -1.0, 1.0))
    y_norm = float(np.clip((h / 2.0 - gy) / (h / 2.0), -1.0, 1.0))
    confidence = float(np.clip(best_score, 0.0, 1.0))

    # Pinhole distance estimate: D = (real_diameter * focal_px) / apparent_px
    distance_m = 0.0
    if major > 1.0:
        distance_m = (KNOWN_CUP_DIAMETER_MM / 1000.0) * focal_length_px / major

    cv2.drawMarker(out, (int(gx), int(gy)), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
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

    def __init__(self, camera_index: int = 0, focal_length_px: float = 700.0) -> None:
        """
        focal_length_px: camera focal length in pixels.
          ~700 is a reasonable starting point for a typical USB webcam at 720p.
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

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

            result, annotated = _detect_in_frame(frame, self._focal_length_px)

            with self._lock:
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
