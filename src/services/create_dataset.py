import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import json
import logging 

import pickle


def load_CONFIG():
    file = open(os.path.join('src', 'config.json'))
    return json.load(file)


CONFIG = load_CONFIG()

def main():

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True,
                            min_detection_confidence=0.3)

    data = []
    labels = []

    for dir_ in os.listdir(CONFIG["paths"]["data"]):
        for img_path in os.listdir(os.path.join(CONFIG["paths"]["data"], dir_)):
            img = cv2.imread(os.path.join(
                CONFIG["paths"]["data"], dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            data_aux = []
            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)

                # Draw the hand marks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
    #         plt.figure()
    #         plt.imshow(img_rgb)

    # plt.show()

    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()

main()