import cv2
from deepface import DeepFace
import time
import matplotlib.pyplot as plt
import datetime
from collections import Counter

# -----------------------------
# Initialize list to store emotions
# Each entry: (timestamp, emotion, confidence)
emotions_log = []

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the webcam and see results.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Analyze emotions
        result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        # Log timestamp, emotion, confidence
        ts = round(time.time(), 2)
        emotions_log.append((ts, emotion, confidence))

        # Show emotion on the webcam feed
        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print("Error analyzing frame:", e)

    # Display webcam feed
    cv2.imshow("Facial Emotion Recognition", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Save emotions to file
with open("emotions_log.txt", "w") as f:
    for ts, emo, conf in emotions_log:
        f.write(f"{ts},{emo},{conf}\n")

print("Session log saved as emotions_log.txt")

# -----------------------------
# Print real-time stats
if emotions_log:
    counts = Counter([e[1] for e in emotions_log])
    print("Emotion counts this session:", counts)

# -----------------------------
# Visualize emotion trends
if emotions_log:
    timestamps, emotions, _ = zip(*emotions_log)
    times = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

    # Map emotions to numbers for plotting
    emotion_map = {'angry': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'sad': 5, 'surprise': 6, 'neutral': 7}
    emotion_nums = [emotion_map.get(e, 0) for e in emotions]

    plt.figure(figsize=(10,5))
    plt.plot(times, emotion_nums, marker='o', linestyle='-')
    plt.yticks(list(emotion_map.values()), list(emotion_map.keys()))
    plt.xlabel("Time")
    plt.ylabel("Emotion")
    plt.title("Emotion Trend Over Time")
    plt.grid(True)
    plt.show()
