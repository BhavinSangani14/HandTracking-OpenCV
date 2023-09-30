import cv2 as cv 
import mediapipe as mp 
import time 


class HandDetector:
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode 
        self.maxHands = maxHands 
        self.detectionCon = detectionCon 
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands = self.maxHands, 
                                        min_detection_confidence = self.detectionCon, min_tracking_confidence = self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, frame, draw = True):

        rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(rgbFrame)
        # print(result.multi_hand_landmarks)
        # n_hands = 0
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, img, handno = 0, draw = True):
        lmsList = []
        if self.result.multi_hand_landmarks:
            
            h, w, c = img.shape
            for id, lm in enumerate(self.handlms.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                print(id, x, y)
                if id == 0:
                    cv.circle(img, (x, y), 15, (0,0,255), thickness=-1)
            



def main():
    video = cv.VideoCapture(0)
    detector = HandDetector()
    prev_time = 0
    cur_time = 0

    while True:
        success, frame = video.read()
        img = detector.findhands(frame)
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time
        cv.putText(frame, str(fps), (20,30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv.imshow("Frame", frame)
        cv.waitKey(1)


main()