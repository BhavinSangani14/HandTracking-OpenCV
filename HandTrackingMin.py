import cv2 as cv 
import mediapipe as mp 
import time 

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
video = cv.VideoCapture(0)

prev_time = 0
cur_time = 0

while True:
    success, frame = video.read()
    rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgbFrame)
    # print(result.multi_hand_landmarks)
    n_hands = 0
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)
            n_hands+=1
            h, w, c = frame.shape
            for id, lm in enumerate(handlms.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                print(id, x, y)
                if id == 0:
                    cv.circle(frame, (x, y), 15, (0,0,255), thickness=-1)
    
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv.putText(frame, str(fps), (20,30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv.putText(frame, "Hands : " + str(n_hands), (frame.shape[1]-200,30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv.imshow("Frame", frame)

    cv.waitKey(1)