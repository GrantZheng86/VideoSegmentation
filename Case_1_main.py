import cv2
import time
import numpy as np
import Case_1_Processing
from skimage import measure
import matplotlib.pyplot as plt

FILE_NAME = "New Videos/2-1.mp4"


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

        if counter == 26:
            print()

        if ret:
            frame = frame[140:965, 0:580, :]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = Case_1_Processing.self_multiply(gray_frame)
            h, w, _ = frame.shape
            original_frame = frame.copy()
            cutoff_height, top_hull, top_contour = Case_1_Processing.bottom_segmentation_recursion_helper(gray_frame)


            print("Frame {}".format(counter))

            if cutoff_height != -1:
                top_hull, flipped = Case_1_Processing.rearrange_hull_contour(top_hull)
                extended_top_contour = Case_1_Processing.extend_top_contour(top_contour, top_hull, flipped, gray_frame)
                height_offset = h - cutoff_height
                top_hull[:, 1] += height_offset
                top_hull = top_hull[top_hull[:,0].argsort()]

                if extended_top_contour is not None:
                    extended_top_contour[:, 1] = extended_top_contour[:, 1] + height_offset
                    cv2.polylines(frame, [extended_top_contour], False, (0, 255, 255), 1)
                    # cv2.imshow("Extended Truncation", frame)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                # cv2.polylines(frame, [extended_top_contour], False, (0, 255, 255), 1)
                # bottom_segmentation_code = Case_1_Processing.detect_valid_bottom_segmentation(slope_with_weight)
                # cv2.putText(frame, str(bottom_segmentation_code), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            time.sleep(1 / 20)
            videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
