import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

app = FastAPI()

from fastapi import Body

@app.post("/stop")
async def stop_all():
    # Later this will call your motor controller
    print("STOP pressed")
    return {"ok": True}

@app.post("/command")
async def command(payload: dict = Body(...)):
    # Example payload: {"type":"stepper_move","id":1,"pos":1234}
    print("COMMAND:", payload)
    return {"ok": True}

clients = set()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        i = 0
        while True:
            i += 1
            await ws.send_json({"type": "heartbeat", "count": i})
            await asyncio.sleep(1)
    finally:
        clients.discard(ws)

@app.post("/hit")
async def hit(payload: dict = Body(...)):
    # payload example: {"target_id": 3}
    msg = {"type": "hit", "target_id": payload.get("target_id"), "t": payload.get("t")}
    dead = []
    for c in clients:
        try:
            await c.send_json(msg)
        except Exception:
            dead.append(c)
    for c in dead:
        clients.discard(c)
    return {"ok": True}

# Mount static LAST so it does not interfere with /ws
app.mount("/", StaticFiles(directory="static", html=True), name="static")
