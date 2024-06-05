import cv2
import mediapipe as mp
import numpy as np

import warnings

warnings.filterwarnings("ignore")
# UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.

import pickle

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

CAPTURE = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:

    data_aux = []
    x_ = []
    y_ = []

    _, frame = CAPTURE.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)
        

        prediction = model.predict([np.asarray(data_aux)])
        # labels_dict = {0: 'A', 1: 'B', 2: 'C'}

        predicted_character = str(prediction[0])

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)
        cv2.putText(img=frame,
                                text=predicted_character,
                                org=(x1, y1 - 10),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale=1.3,
                                color=(0, 0, 0),
                                thickness=3,
                                lineType=cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

# CAPTURE.release()
# cv2.destroyAllWindows()
