import cv2
import numpy as np
import time

FILE_NAME = "../New Videos/2-2.mp4"
GAMMA = 1.8
THRESHOLD_VALUE = 70


def adjust_gamma(image, gamma=0.75):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == "__main__":
    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = frame[140:965, 0:580, :]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            adjusted_gray = adjust_gamma(gray_frame, GAMMA)
            _, adjusted_gray = cv2.threshold(adjusted_gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            adjusted_gray_rgb = cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(adjusted_gray_rgb, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Gamma {}'.format(GAMMA), adjusted_gray_rgb)

        time.sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


