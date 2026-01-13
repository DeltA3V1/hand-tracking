import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print("Camera index", i, "is available")
        cap.release()
