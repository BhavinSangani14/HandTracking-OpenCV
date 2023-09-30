import cv2 as cv 
import time 
import numpy as np 
import mediapipe as mp 
import HandTrackingModule as htm
# from _ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


wCam, hCam = 600, 480 

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prev_time = 0


# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(
#     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# volumeRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
# minVol = volumeRange[0]
# maxVol = volumeRange[1]


detector = htm.HandDetector(detectionCon = 0.75)
while True:
    success, img = cap.read()
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    img = detector.findHands(img)
    lms_list = detector.findPosition(img, 0)
    if len(lms_list)!=0:
        x1, y1 = lms_list[4][1], lms_list[4][2]
        x2, y2 = lms_list[8][1], lms_list[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        d = int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
        # print(d)
        vol = np.interp(d, [50, 260], [-65, 0])
        print(int(330 - (vol + 65)*2))
        # volume.SetMasterVolumeLevel(vol, None)
        cv.circle(img, (x1, y1), 10, (0, 0, 255), -1)
        cv.circle(img, (x2, y2), 10, (0, 0, 255), -1)
        cv.circle(img, (cx, cy), 10, (0, 0, 255), -1)
        cv.rectangle(img, (10, int(330 - (vol + 65)*2)), (50, 330), (0,255,0), -1)
        cv.putText(img, f'{int(((65 + vol)/65) * 100)}%', (10, 350), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255,0), 2)
        if d < 30:
            cv.circle(img, (cx, cy), 10, (0, 255, 255), -1)
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
    cv.putText(img, f'FPS : {int(fps)}', (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    cv.rectangle(img, (10, 200), (50, 330), (0, 255, 0), 2)
    cv.imshow("Image", img)
    cv.waitKey(1)