# emotion_client.py
import cv2
from deepface import DeepFace
import time
import requests
import json

# ====== CONFIG ======
API_URL = "http://localhost:3000/api/emotion/ingest"   # or your deployed URL
EMOTION_API_KEY = "7wM2UJYkKjrRa_JXtHKtEG5jbFzim7I45pvXH1xoVTo"  # <-- paste the key here
SEND_EVERY_SECONDS = 2.0
# ====================

# Map DeepFace emotions -> schema Emotion enum
EMOTION_MAP = {
    "happy": "HAPPY",
    "neutral": "NEUTRAL",
    "sad": "SAD",
    "angry": "ANGRY",
    # Collapse others to STRESSED
    "fear": "STRESSED",
    "disgust": "STRESSED",
    "surprise": "STRESSED",
    "stress": "STRESSED",
    "stressed": "STRESSED",
}

def send_emotion(emotion: str, confidence: float, ts_sec: float):
    payload = {
        "emotion": emotion,          # raw string; server will map/normalize
        "confidence": float(confidence),
        "timestamp": ts_sec,         # seconds epoch; server handles both str/num
    }
    headers = {
        "Content-Type": "application/json",
        "X-Emotion-Key": EMOTION_API_KEY,
    }
    try:
        r = requests.post(API_URL, data=json.dumps(payload), headers=headers, timeout=5)
        print("[POST]", r.status_code, r.text)
    except Exception as e:
        print("[POST] error:", e)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit.")
    last_sent = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            result = DeepFace.analyze(small, actions=['emotion'], enforce_detection=False)
            dom = result[0]['dominant_emotion']
            conf = result[0]['emotion'][dom]

            # Overlay
            cv2.putText(frame, f"{dom} ({conf:.1f}%)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Throttle outbound posts
            now = time.time()
            if now - last_sent >= SEND_EVERY_SECONDS:
                last_sent = now
                send_emotion(dom, conf, now)

        except Exception as e:
            # If a frame fails, keep going
            print("Analyze error:", e)

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
