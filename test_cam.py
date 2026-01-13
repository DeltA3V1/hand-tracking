import cv2

cap = cv2.VideoCapture(2)
ret, frame = cap.read()
if ret:
    cv2.imshow("Camera 2", frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
