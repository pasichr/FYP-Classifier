from ultralytics import YOLO
import cv2
#import cvzone

cap=cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)

model = YOLO("yolo-weights/yolov8n.pt")
results=model()


while True:
    success, img =cap.read()
    cv2.imshow("IMAGE",img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
