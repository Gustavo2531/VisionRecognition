#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Wed May 30 23:16:18 2018

#@author: gustavomendez
#"""

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (x2, y2, w2, h2) in eyes:
            cv2.rectangle(roi_color, (x2,y2), (x2+w2,y2+h2), (0, 255,  0), 2)
        smile = smile_cascade.detectMultiScale(roi_gray,1.7,22)
        for (x3, y3, w3, h3) in smile:
            cv2.rectangle(roi_color, (x3,y3), (x3+w3,y3+h3), (0, 0,  255), 2)
    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
    
        