import cv2 as cv 
import time 
import mediapipe as mp 
import HandTrackingModule as htm 
import numpy as np 

wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmsList = detector.findPosition(img, 1)
    if len(lmsList)!=0:
        thumb = 0
        index_fin = 0
        mid_fin = 0
        ring_fin = 0
        pinky_fin = 0
        x, y = lmsList[0][1], lmsList[0][2]
        x1, y1 = lmsList[4][1], lmsList[4][2]
        x2, y2 = lmsList[8][1], lmsList[8][2]
        x3, y3 = lmsList[12][1], lmsList[12][2]
        x4, y4 = lmsList[16][1], lmsList[16][2]
        x5, y5 = lmsList[20][1], lmsList[20][2]
        
        xt, yt = lmsList[3][1], lmsList[3][2]
        x6, y6 = lmsList[6][1], lmsList[6][2]
        x10, y10 = lmsList[10][1], lmsList[10][2]
        x14, y14 = lmsList[14][1], lmsList[14][2]
        x18, y18 = lmsList[18][1], lmsList[18][2]
        
        if y2 <= y6:
            index_fin = 1
        if y3 <= y10:
            mid_fin = 1
        if y4 <= y14:
            ring_fin = 1
        if y5 <= y18:
            pinky_fin = 1
        if x1 >= xt:
            thumb = 1
            
        
        fing_count = index_fin + mid_fin + ring_fin + pinky_fin + thumb
        print(fing_count)
        
        cv.putText(img, f'Finger Up : {fing_count}', (10, 30), cv.FONT_HERSHEY_COMPLEX ,0.7, (255,0,0), 2)
        
        
        
        
    cv.imshow("Image", img)
    cv.waitKey(1)