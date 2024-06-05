import cv2
import json
import os
from functools import partial

import logging


def load_config():
    file = open(os.path.join('src', 'config.json'))
    return json.load(file)


CONFIG = load_config()
CAPTURE = cv2.VideoCapture(0)


def initial_screen():
    key = -1
    while key == -1:
        _, frame = CAPTURE.read()

        cv2.putText(
            img=frame,
            text='Press any letter to collect image data',
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.8,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            img=frame,
            text='or "ESC" to quit.',
            org=(50, 90),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.8,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)

    logging.info('User pressed the "%s" key.', chr(
        key) if key != ord('\x1b') else "Esc")

    return (True, chr(key)) if key != ord('\x1b') else (False, None)


def collect_images(key: str):
    key = key.upper()

    class_path = os.path.join(CONFIG["paths"]["data"], key)
    os.makedirs(class_path) if not os.path.exists(class_path) else None

    logging.info('Collecting data for letter %s.', key.upper())

    putTextOnFrame = partial(cv2.putText,
                             org=(50, 50),
                             fontFace=cv2.FONT_HERSHEY_COMPLEX,
                             fontScale=1,
                             color=(0, 0, 0),
                             thickness=2,
                             lineType=cv2.LINE_AA)

    for i in range(3, 0, -1):
        ms = 50
        for j in range(1000//ms):
            _, frame = CAPTURE.read()
            putTextOnFrame(img=frame, text=f'{i}...')
            cv2.imshow('frame', frame)
            cv2.waitKey(ms)

    counter = 0
    while counter < CONFIG["dataset"]["subset_size"]:
        _, frame = CAPTURE.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(50)
        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)

        counter += 1


def main():
    logging.basicConfig(level=logging.INFO)

    while True:
        shall_continue, key = initial_screen()

        if shall_continue:
            collect_images(key)
        else:
            break


main()
