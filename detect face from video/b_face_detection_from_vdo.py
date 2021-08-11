import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture('for_face_detection.mp4')
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 3)

        for(x, y, w, h) in face:
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)


        cv2.imshow('image', frame)

        if cv2.waitKey(20) & 0xFF == ord('e'):
            break

video.release()
cv2.destroyAllWindows()