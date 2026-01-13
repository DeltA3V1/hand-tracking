import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import hand_landmarker

# 1️⃣ Import BaseOptions to specify model
from mediapipe.tasks.python.vision.core import BaseOptions

# 2️⃣ Create options
options = hand_landmarker.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),  # model file
    running_mode=hand_landmarker.HandLandmarkerOptions.RunningMode.VIDEO
)

# 3️⃣ Create the hand landmarker
landmarker = hand_landmarker.HandLandmarker.create_from_options(options)

# 4️⃣ Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(rgb, 0)  # frame, timestamp

    if result.handedness:
        for hand in result.handedness:
            print(hand)  # prints detected hand labels

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
