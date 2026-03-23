from __future__ import annotations

import asyncio
from time import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import HEARTBEAT_SEC, UI_ROOT
from .fire_gate import compute_fire_enabled
from .models import RuntimeState
from .state_machine import (
    arm,
    begin_homing,
    clear_fault,
    complete_homing,
    disarm,
    mark_boot,
    mark_fire_complete,
    mark_firing,
    set_fault,
    set_pitch_confirmed,
)
from .vision_engine import VisionEngine
from .ws_hub import WebSocketHub

app = FastAPI(title="Turret Targeting", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if UI_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(UI_ROOT)), name="static")


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, "payload": payload, "ts": time()}


state = RuntimeState()
vision = VisionEngine()
hub = WebSocketHub()
heartbeat_task: asyncio.Task | None = None


@app.on_event("startup")
async def on_startup() -> None:
    global heartbeat_task
    mark_boot(state)
    set_pitch_confirmed(state, True)  # per hardware assumption
    begin_homing(state)
    vision.start()
    heartbeat_task = asyncio.create_task(_heartbeat_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if heartbeat_task:
        heartbeat_task.cancel()
    vision.stop()


async def _heartbeat_loop() -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SEC)
        _sync_vision()
        await hub.broadcast(_event("state", _state_payload()))
        await hub.broadcast(_event("heartbeat", {"ok": True}))


def _sync_vision() -> None:
    result, err = vision.snapshot()
    state.vision = result
    if err and not state.fault:
        set_fault(state, err)
    state.updated_at = time()


def _state_payload() -> dict[str, Any]:
    fire_enabled, reasons = compute_fire_enabled(state)
    payload = state.as_dict()
    payload["fire_enabled"] = fire_enabled
    payload["fire_block_reasons"] = reasons
    payload["vision_source"] = vision.source_status()
    return payload


@app.get("/")
async def root() -> Response:
    index = UI_ROOT / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"ok": True, "message": "UI not found"})


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    _sync_vision()
    return _state_payload()


@app.post("/api/control/home")
async def api_home() -> dict[str, Any]:
    if state.fault:
        return {"ok": False, "error": "fault_active"}
    begin_homing(state)
    await hub.broadcast(_event("system", {"message": "homing_started"}))
    # First-pass: treat homing as immediate acknowledgement.
    complete_homing(state)
    await hub.broadcast(_event("system", {"message": "homing_complete"}))
    return {"ok": True}


@app.post("/api/control/pitch/confirm-low")
async def api_confirm_pitch() -> dict[str, Any]:
    set_pitch_confirmed(state, True)
    await hub.broadcast(_event("system", {"message": "pitch_confirmed_low"}))
    return {"ok": True}


@app.post("/api/control/arm")
async def api_arm() -> dict[str, Any]:
    arm(state)
    ok = state.armed
    await hub.broadcast(_event("system", {"message": "armed" if ok else "arm_rejected"}))
    return {"ok": ok, "state": _state_payload()}


@app.post("/api/control/disarm")
async def api_disarm() -> dict[str, Any]:
    disarm(state)
    await hub.broadcast(_event("system", {"message": "disarmed"}))
    return {"ok": True, "state": _state_payload()}


@app.post("/api/control/flywheel-ready")
async def api_flywheel_ready(ready: bool) -> dict[str, Any]:
    state.flywheel_ready = bool(ready)
    state.updated_at = time()
    await hub.broadcast(_event("readiness", {"flywheel_ready": state.flywheel_ready}))
    return {"ok": True}


@app.post("/api/control/aim-ready")
async def api_aim_ready(ready: bool) -> dict[str, Any]:
    state.aim_ready = bool(ready)
    state.updated_at = time()
    await hub.broadcast(_event("readiness", {"aim_ready": state.aim_ready}))
    return {"ok": True}


@app.post("/api/control/fire")
async def api_fire() -> dict[str, Any]:
    _sync_vision()
    fire_enabled, reasons = compute_fire_enabled(state)
    if not fire_enabled:
        await hub.broadcast(_event("fire_state", {"status": "blocked", "reasons": reasons}))
        return {"ok": False, "reasons": reasons}

    mark_firing(state)
    await hub.broadcast(_event("fire_state", {"status": "pending_approval"}))

    # First pass: user press is approval, emulate short fire pulse.
    await asyncio.sleep(0.2)
    mark_fire_complete(state)
    await hub.broadcast(_event("fire_state", {"status": "fired", "count": state.fire_count}))
    return {"ok": True, "count": state.fire_count}


@app.post("/api/control/estop")
async def api_estop() -> dict[str, Any]:
    set_fault(state, "estop")
    await hub.broadcast(_event("fault", {"reason": "estop"}))
    return {"ok": True}


@app.post("/api/control/reset-fault")
async def api_reset_fault() -> dict[str, Any]:
    clear_fault(state)
    await hub.broadcast(_event("system", {"message": "fault_cleared"}))
    return {"ok": True}


@app.post("/api/vision/source")
async def api_set_vision_source(mode: str) -> dict[str, Any]:
    try:
        status = vision.set_source_mode(mode)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    await hub.broadcast(_event("vision_source", status))
    return {"ok": True, "vision_source": status}


@app.post("/api/vision/test-images/next")
async def api_vision_next_test_image() -> dict[str, Any]:
    status = vision.next_test_image()
    await hub.broadcast(_event("vision_source", status))
    return {"ok": True, "vision_source": status}


@app.post("/api/vision/test-images/auto")
async def api_vision_test_auto(enabled: bool) -> dict[str, Any]:
    status = vision.set_test_auto_advance(enabled)
    await hub.broadcast(_event("vision_source", status))
    return {"ok": True, "vision_source": status}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await hub.connect(websocket)
    await websocket.send_json(_event("state", _state_payload()))
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(websocket)


@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    async def frame_generator():
        while True:
            jpg = vision.latest_jpeg()
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            await asyncio.sleep(0.04)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
