import cv2
import mediapipe as mp
import asl_model

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

_, frame = cap.read()

h, w, c = frame.shape

while True:
    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            print(handLMs)
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for index, lm in enumerate(handLMs.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)
            pred_img = frame[y_min - 40:y_max + 40, x_min - 40:x_max + 40]
            if pred_img.shape[0] != 0 and pred_img.shape[1] != 0:
                pred_class = asl_model.get_asl_class(frame[y_min - 40:y_max + 40, x_min - 40:x_max + 40])
                cv2.putText(frame, "ASL: " + pred_class, (x_min - 20, y_min - 20 - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)
