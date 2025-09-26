import cv2
from deepface import DeepFace
import time

emotions_log = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Log emotion with timestamp
        emotions_log.append((time.time(), emotion))

        cv2.putText(frame, f"Emotion: {emotion}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    except:
        pass

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save emotions to file
with open("emotions_log.txt", "w") as f:
    for ts, emo in emotions_log:
        f.write(f"{ts},{emo}\n")
