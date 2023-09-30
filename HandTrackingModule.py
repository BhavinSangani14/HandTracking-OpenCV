import cv2 as cv 
import mediapipe as mp 
import time 

class HandDetector:
    
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode 
        self.maxHands = maxHands
        self.detectionCon = detectionCon 
        self.trackingCon = trackingCon 

        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands = self.maxHands, 
                                        min_detection_confidence = self.detectionCon, min_tracking_confidence = self.trackingCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(rgb_img)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo):
        lms_list = []
        h, w, c = img.shape 
        if self.result.multi_hand_landmarks:
            for id, handLms in enumerate(self.result.multi_hand_landmarks[handNo - 1].landmark):
                x = int(handLms.x * w)
                y = int(handLms.y * h)
                lms_list.append([id, x, y])

        return lms_list

                



# def main():
#     video = cv.VideoCapture(0)
#     detector = HandDetector()
#     prev_time = 0
#     cur_time = 0
#     while True:
#         success, img = video.read()
#         img = detector.findHands(img, draw=True)
#         detector.findPosition(img, 1)
#         prev_time = cur_time
#         cur_time = time.time()
#         fps = int(1 / (cur_time - prev_time))
#         cv.putText(img, str(fps), (20, 40), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
#         cv.imshow("Video", img)
#         cv.waitKey(1)



# main()