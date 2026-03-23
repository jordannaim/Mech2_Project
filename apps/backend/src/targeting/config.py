from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "ellipseDetectionV6.py").exists():
            return parent
    # fallback for expected layout: <root>/apps/backend/src/targeting/config.py
    return here.parents[4]


PROJECT_ROOT = _find_project_root()
TUNING_PATH = PROJECT_ROOT / "ellipse_tuning.json"
UI_ROOT = PROJECT_ROOT / "apps" / "ui" / "src"

# Vision processing defaults (overridden by tuning file values when available)
DEFAULT_TUNING = {
    "Canny Low": 60,
    "Min Major Axis": 40,
    "Min Minor Axis": 12,
    "Max Minor Axis": 180,
    "Max Axis": 240,
    "Min Aspect x100": 18,
    "Max Aspect x100": 100,
    "Use Red ROI": 1,
    "Red Pad": 50,
    "Red MinArea": 30,
    "Cup Confidence x100": 30,
}

HEARTBEAT_SEC = 0.5
FIRE_COOLDOWN_SEC = 1.5
TARGET_STALE_SEC = 0.6
