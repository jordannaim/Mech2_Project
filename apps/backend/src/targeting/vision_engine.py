from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from typing import Any
import glob

import cv2
import numpy as np

from .config import DEFAULT_TUNING, TUNING_PATH
from .models import VisionResult


K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass
class TargetEllipse:
    cx: float
    cy: float
    ma: float
    mi: float
    angle: float
    support: float
    score: float


def clamp_roi(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = int(max(0, min(w - 1, x0)))
    y0 = int(max(0, min(h - 1, y0)))
    x1 = int(max(0, min(w, x1)))
    y1 = int(max(0, min(h, y1)))
    if x1 <= x0 + 5 or y1 <= y0 + 5:
        return 0, 0, w, h
    return x0, y0, x1, y1


def biggest_red_roi(frame_bgr: np.ndarray, pad: int = 60, min_area: int = 1500) -> tuple[int, int, int, int, bool]:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 150, 80], dtype=np.uint8)
    upper1 = np.array([8, 255, 255], dtype=np.uint8)
    lower2 = np.array([172, 150, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red = cv2.bitwise_or(m1, m2)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, ker, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, ker, iterations=2)

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    red_merged = cv2.dilate(red, merge_kernel, iterations=2)

    cnts, _ = cv2.findContours(red_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, w, h, False

    valid_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not valid_cnts:
        return 0, 0, w, h, False

    all_points = np.vstack(valid_cnts)
    x, y, rw, rh = cv2.boundingRect(all_points)
    vertical_shift = int(rh * 0.3)

    x0 = x - pad
    y0 = y - pad - vertical_shift
    x1 = x + rw + pad
    y1 = y + rh + pad - vertical_shift
    x0, y0, x1, y1 = clamp_roi(x0, y0, x1, y1, w, h)
    return x0, y0, x1, y1, True


def ellipse_points(cx: float, cy: float, a: float, b: float, ang_deg: float, n: int = 90) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(np.deg2rad(ang_deg)), np.sin(np.deg2rad(ang_deg))
    x = a * np.cos(t)
    y = b * np.sin(t)
    xr = ca * x - sa * y + cx
    yr = sa * x + ca * y + cy
    return np.stack([xr, yr], axis=1)


def support_score(edges: np.ndarray, cx: float, cy: float, major: float, minor: float, angle_deg: float) -> float:
    a = major / 2.0
    b = minor / 2.0
    pts = ellipse_points(cx, cy, a, b, angle_deg, n=120)
    h, w = edges.shape
    hits = 0

    for xf, yf in pts:
        x = int(round(xf))
        y = int(round(yf))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        y0, y1 = max(0, y - 2), min(h, y + 3)
        x0, x1 = max(0, x - 2), min(w, x + 3)
        if np.any(edges[y0:y1, x0:x1] > 0):
            hits += 1

    return hits / 120.0


class VisionEngine:
    def __init__(self, camera_index: int = 0, tuning_path: Path = TUNING_PATH) -> None:
        self.camera_index = camera_index
        self.tuning_path = tuning_path
        self.test_image_dir = self.tuning_path.parent / "test_images" / "newer_test_images"
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._source_mode = "camera"
        self._test_images: list[Path] = []
        self._test_index = 0
        self._test_auto_advance = False

        self._last_jpeg = b""
        self._last_result = VisionResult(valid=False, lock_state="NO_TARGET")
        self._last_error = ""

        self._reload_test_images()

    def load_tuning(self) -> dict[str, int]:
        settings = dict(DEFAULT_TUNING)
        try:
            with self.tuning_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k in settings:
                    if k in raw:
                        settings[k] = int(raw[k])
        except Exception:
            pass
        return settings

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[VisionResult, str]:
        with self._lock:
            return self._last_result, self._last_error

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._last_jpeg

    def source_status(self) -> dict[str, Any]:
        with self._lock:
            current_name = ""
            if self._test_images and 0 <= self._test_index < len(self._test_images):
                current_name = self._test_images[self._test_index].name
            return {
                "mode": self._source_mode,
                "test_auto_advance": self._test_auto_advance,
                "test_image_count": len(self._test_images),
                "test_image_index": self._test_index,
                "test_image_name": current_name,
                "test_image_dir": str(self.test_image_dir),
            }

    def set_source_mode(self, mode: str) -> dict[str, Any]:
        mode = mode.strip().lower()
        if mode not in {"camera", "test-images"}:
            raise ValueError("mode must be 'camera' or 'test-images'")
        with self._lock:
            self._source_mode = mode
        return self.source_status()

    def set_test_auto_advance(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self._test_auto_advance = bool(enabled)
        return self.source_status()

    def next_test_image(self) -> dict[str, Any]:
        with self._lock:
            if not self._test_images:
                self._reload_test_images()
            if self._test_images:
                self._test_index = (self._test_index + 1) % len(self._test_images)
        return self.source_status()

    def _reload_test_images(self) -> None:
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(Path(p) for p in glob.glob(str(self.test_image_dir / pattern)))
        paths = sorted(paths)
        self._test_images = paths
        if self._test_index >= len(self._test_images):
            self._test_index = 0

    def _read_test_frame(self) -> tuple[np.ndarray | None, str]:
        with self._lock:
            if not self._test_images:
                self._reload_test_images()
            if not self._test_images:
                return None, "no_test_images"

            path = self._test_images[self._test_index]
            auto_advance = self._test_auto_advance

        frame = cv2.imread(str(path))
        if frame is None:
            return None, "test_image_read_failed"

        if auto_advance:
            with self._lock:
                if self._test_images:
                    self._test_index = (self._test_index + 1) % len(self._test_images)

        return frame, ""

    def _publish(self, frame: np.ndarray, result: VisionResult, error: str = "") -> None:
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with self._lock:
            if ok:
                self._last_jpeg = enc.tobytes()
            self._last_result = result
            self._last_error = error

    def _run_loop(self) -> None:
        cap: cv2.VideoCapture | None = None

        while self._running:
            with self._lock:
                mode = self._source_mode

            if mode == "camera":
                if cap is None:
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    else:
                        cap.release()
                        cap = None

                if cap is None or not cap.isOpened():
                    self._publish(
                        np.zeros((480, 640, 3), dtype=np.uint8),
                        VisionResult(valid=False, lock_state="NO_CAMERA", timestamp=time(), frame_width=640, frame_height=480),
                        "camera_open_failed",
                    )
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    self._publish(
                        np.zeros((480, 640, 3), dtype=np.uint8),
                        VisionResult(valid=False, lock_state="FRAME_FAIL", timestamp=time(), frame_width=640, frame_height=480),
                        "frame_read_failed",
                    )
                    continue
            else:
                if cap is not None:
                    cap.release()
                    cap = None

                frame, err = self._read_test_frame()
                if frame is None:
                    self._publish(
                        np.zeros((480, 640, 3), dtype=np.uint8),
                        VisionResult(valid=False, lock_state="NO_TEST_IMAGE", timestamp=time(), frame_width=640, frame_height=480),
                        err,
                    )
                    continue

            result, overlay = self._process_frame(frame)
            self._publish(overlay, result)

            if mode == "test-images":
                sleep(0.08)

        if cap is not None:
            cap.release()

    def _process_frame(self, frame: np.ndarray) -> tuple[VisionResult, np.ndarray]:
        tune = self.load_tuning()
        out = frame.copy()
        h, w = frame.shape[:2]

        use_roi = int(tune.get("Use Red ROI", 1)) == 1
        red_pad = int(tune.get("Red Pad", 60))
        red_min_area = int(tune.get("Red MinArea", 15)) * 100

        if use_roi:
            x0, y0, x1, y1, found_red = biggest_red_roi(frame, pad=red_pad, min_area=red_min_area)
        else:
            x0, y0, x1, y1, found_red = 0, 0, w, h, True

        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 2)
        if not found_red:
            cv2.putText(out, "No red ROI", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return (
                VisionResult(valid=False, lock_state="NO_RED_ROI", timestamp=time(), frame_width=w, frame_height=h),
                out,
            )

        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)

        canny_lo = max(1, int(tune.get("Canny Low", 60)))
        canny_hi = canny_lo * 2
        edges = cv2.Canny(blur, canny_lo, canny_hi)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K3, iterations=1)
        edges = cv2.dilate(edges, K3, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        min_major = int(tune.get("Min Major Axis", 40))
        min_minor = int(tune.get("Min Minor Axis", 12))
        max_minor = max(min_minor + 1, int(tune.get("Max Minor Axis", 180)))
        max_axis = max(min_major + 1, int(tune.get("Max Axis", 240)))
        min_aspect = int(tune.get("Min Aspect x100", 18)) / 100.0
        max_aspect = max(min_aspect, int(tune.get("Max Aspect x100", 100)) / 100.0)
        conf_thresh = int(tune.get("Cup Confidence x100", 30)) / 100.0

        candidates: list[TargetEllipse] = []
        for cnt in contours:
            if len(cnt) < 12:
                continue
            try:
                (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
            except cv2.error:
                continue

            major = max(ma, mi)
            minor = min(ma, mi)
            if major < min_major or minor < min_minor or major > max_axis or minor > max_minor:
                continue

            aspect = minor / major if major > 1e-6 else 0.0
            if not (min_aspect <= aspect <= max_aspect):
                continue

            support = support_score(edges, cx, cy, ma, mi, angle)
            score = 0.7 * support + 0.3 * (1.0 - abs(0.6 - aspect))
            if score < conf_thresh:
                continue

            candidates.append(TargetEllipse(cx=cx, cy=cy, ma=ma, mi=mi, angle=angle, support=support, score=score))

        if not candidates:
            cv2.putText(out, "NO TARGET", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return (
                VisionResult(valid=False, lock_state="NO_TARGET", timestamp=time(), frame_width=w, frame_height=h),
                out,
            )

        candidates.sort(key=lambda e: e.score, reverse=True)
        best = candidates[0]

        for idx, e in enumerate(candidates[:8]):
            color = (255, 0, 255) if idx == 0 else (0, 255, 0)
            gx = int(round(e.cx + x0))
            gy = int(round(e.cy + y0))
            cv2.ellipse(out, ((gx, gy), (e.ma, e.mi), e.angle), color, 2)
            cv2.circle(out, (gx, gy), 3, color, -1)

        gx = float(best.cx + x0)
        gy = float(best.cy + y0)

        x_norm = (gx - (w / 2.0)) / (w / 2.0)
        y_norm = ((h / 2.0) - gy) / (h / 2.0)
        confidence = float(np.clip(best.score, 0.0, 1.0))

        cv2.drawMarker(out, (int(gx), int(gy)), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            out,
            f"TARGET x={x_norm:+.3f} y={y_norm:+.3f} conf={confidence:.2f}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return (
            VisionResult(
                valid=True,
                x_norm=float(np.clip(x_norm, -1.0, 1.0)),
                y_norm=float(np.clip(y_norm, -1.0, 1.0)),
                confidence=confidence,
                lock_state="LOCK",
                timestamp=time(),
                frame_width=w,
                frame_height=h,
            ),
            out,
        )
