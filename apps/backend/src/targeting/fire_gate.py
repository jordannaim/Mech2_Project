from __future__ import annotations

from time import time

from .config import FIRE_COOLDOWN_SEC, TARGET_STALE_SEC
from .models import RuntimeState


def compute_fire_enabled(state: RuntimeState) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if state.fault:
        reasons.append("fault")
    if not state.yaw_homed:
        reasons.append("yaw_not_homed")
    if not state.pitch_zero_assumed:
        reasons.append("pitch_not_confirmed")
    if not state.armed:
        reasons.append("not_armed")
    if not state.flywheel_ready:
        reasons.append("flywheel_not_ready")
    if not state.aim_ready:
        reasons.append("aim_not_ready")

    if not state.vision.valid:
        reasons.append("no_target")
    else:
        age = time() - state.vision.timestamp
        if age > TARGET_STALE_SEC:
            reasons.append("target_stale")
        if state.vision.confidence < 0.25:
            reasons.append("low_confidence")

    if time() - state.last_fire_ts < FIRE_COOLDOWN_SEC:
        reasons.append("cooldown")

    if state.fire_in_progress:
        reasons.append("fire_in_progress")

    return len(reasons) == 0, reasons
