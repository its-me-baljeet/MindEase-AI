# emotion_ws.py
import websocket
import json
import time

WS_URL = "ws://localhost:3000/ws"

ws = None


def connect():
    global ws
    while True:
        try:
            print("[WS] Connecting...")
            ws = websocket.WebSocket()
            ws.connect(WS_URL)
            print("[WS] Connected!")
            return
        except Exception as e:
            print("[WS] Connect error:", e)
            time.sleep(2)


def send_ws_emotion(emotion: str, confidence: float, ts: float):
    global ws
    try:
        ws.send(json.dumps({
            "type": "emotion",
            "emotion": emotion,
            "confidence": confidence,
            "ts": ts
        }))
    except Exception:
        # reconnect and retry
        connect()
        ws.send(json.dumps({
            "type": "emotion",
            "emotion": emotion,
            "confidence": confidence,
            "ts": ts
        }))
