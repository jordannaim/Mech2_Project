from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any


class SystemMode(str, Enum):
    BOOT = "BOOT"
    HOMING_YAW = "HOMING_YAW"
    READY_IDLE = "READY_IDLE"
    ARMED = "ARMED"
    FIRING = "FIRING"
    FAULT = "FAULT"


@dataclass
class VisionResult:
    valid: bool = False
    x_norm: float = 0.0
    y_norm: float = 0.0
    confidence: float = 0.0
    lock_state: str = "NO_TARGET"
    timestamp: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


@dataclass
class RuntimeState:
    mode: SystemMode = SystemMode.BOOT
    yaw_homed: bool = False
    pitch_zero_assumed: bool = False
    flywheel_ready: bool = False
    aim_ready: bool = False
    armed: bool = False
    fault: str = ""
    fire_in_progress: bool = False
    fire_count: int = 0
    last_fire_ts: float = 0.0
    vision: VisionResult = field(default_factory=VisionResult)
    updated_at: float = field(default_factory=time)

    def as_dict(self) -> dict[str, Any]:
        v = self.vision
        return {
            "mode": self.mode.value,
            "yaw_homed": self.yaw_homed,
            "pitch_zero_assumed": self.pitch_zero_assumed,
            "flywheel_ready": self.flywheel_ready,
            "aim_ready": self.aim_ready,
            "armed": self.armed,
            "fault": self.fault,
            "fire_in_progress": self.fire_in_progress,
            "fire_count": self.fire_count,
            "last_fire_ts": self.last_fire_ts,
            "vision": {
                "valid": v.valid,
                "x_norm": v.x_norm,
                "y_norm": v.y_norm,
                "confidence": v.confidence,
                "lock_state": v.lock_state,
                "timestamp": v.timestamp,
                "frame_width": v.frame_width,
                "frame_height": v.frame_height,
            },
            "updated_at": self.updated_at,
        }
