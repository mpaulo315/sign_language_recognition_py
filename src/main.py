import cv2
import os
import json


if __name__ == '__main__':
    with open(os.path.join('src', 'config.json')) as file:
        config = json.load(file)

        CAPTURE = cv2.VideoCapture(0)
        NUMBER_OF_CLASSES = 3
        DATASET_SIZE = 100

        def start():
            while True:
                _, frame = CAPTURE.read()
                
                cv2.putText(img=frame,
                            text='Ready? Press "Q" or "E" to exit.',
                            org=(100, 50),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1.3,
                            color=(255, 255, 0),
                            thickness=3,
                            lineType=cv2.LINE_AA)

                cv2.imshow('frame', frame)

                return_key = cv2.waitKey(25)

                if return_key == ord('q'):
                    return True
                elif return_key == ord('e'):
                    return False

    shall_continue = start()
    if shall_continue:
        for j in range(NUMBER_OF_CLASSES):
            class_path = os.path.join(config["paths"]["data"], str(j))
            os.makedirs(class_path) if not os.path.exists(class_path) else None

            print(f'Collecting data for class {j}')

            counter = 0
            while counter < DATASET_SIZE:
                ret, frame = CAPTURE.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(
                    class_path, f'{counter}.jpg'), frame)
                
                counter += 1

    CAPTURE.release()
    cv2.destroyAllWindows()

