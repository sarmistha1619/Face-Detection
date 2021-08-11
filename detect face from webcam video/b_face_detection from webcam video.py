import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
video = cv2.VideoCapture('for_face_detection.mp4')
while (video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3,5)
        for (x, y, w, h) in face:
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            gray_scale = gray[y:y+h, x:x+w]
            color_scale = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(color_scale)
            for (m, n, o, p) in eyes:
                cv2.rectangle(color_scale, (m,n), (m+o, n+p), (0,255,0),2)
        cv2.imshow('Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video.release()
cv2.destroyAllWindows()