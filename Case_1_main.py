import cv2
import time
import numpy as np
import Case_1_Processing
from skimage import measure
import matplotlib.pyplot as plt

FILE_NAME = "New Videos/1-1.mp4"


def adjust_gamma(image, gamma=1.0):
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
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Case1.avi", fourcc, fps=24, frameSize=(580, 825))

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = frame[140:965, 0:580, :]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = Case_1_Processing.self_multiply(gray_frame)
            h, w, _ = frame.shape
            original_frame = frame.copy()
            cutoff_height, top_contour, r = Case_1_Processing.bottom_segmentation_recursion_helper(gray_frame)

            if cutoff_height != -1:
                height_offset = h - cutoff_height
                top_contour[:, 1] += height_offset
                top_contour = top_contour[top_contour[:,0].argsort()]
                cv2.polylines(frame, [top_contour], False, (0, 255, 255), 1)
                cv2.putText(frame, str(r), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if r < 2.2:
                    cv2.putText(frame, "Unreliable", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            time.sleep(1 / 20)
            videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
