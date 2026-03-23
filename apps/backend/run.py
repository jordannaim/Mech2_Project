from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


# Allow running this file directly: python /path/to/apps/backend/run.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    uvicorn.run("apps.backend.src.targeting.main:app", host="0.0.0.0", port=8000, reload=False)
