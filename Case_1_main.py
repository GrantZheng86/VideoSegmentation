import cv2
import time
import numpy as np
import Case_1_Processing
import Case1_Middle_Layer
import Case_1_2nd_layer
from skimage import measure
import matplotlib.pyplot as plt
PATIENT_NUM = 1
FILE_NAME = "New Videos/{}-1.mp4".format(PATIENT_NUM)


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

        if counter == 63:
            print()

        if ret:
            frame = frame[140:965, 0:580, :]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            original_gray_frame = gray_frame.copy()
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
                    blacked_out_frame = Case_1_Processing.black_out_bottom_part(extended_top_contour,
                                                                                original_gray_frame)

                    top_middle_contour, bottom_middle_contour = Case1_Middle_Layer.extract_contour(blacked_out_frame)
                    bottom_middle_contour = Case1_Middle_Layer.post_process_bottom_contour(bottom_middle_contour)
                    top_middle_contour = Case1_Middle_Layer.post_process_top_contour(top_middle_contour)
                    bottom_middle_contour = Case1_Middle_Layer.fill_contour(bottom_middle_contour, frame.shape[1])
                    top_middle_contour = Case1_Middle_Layer.fill_contour(top_middle_contour, frame.shape[1])
                    Case1_Middle_Layer.correct_extended_contour(top_middle_contour, bottom_middle_contour, frame.shape[1])

                    if top_middle_contour is not None:
                        blackout_contour_2 = top_middle_contour.copy()
                        blackout_contour_2[:, 1] = blackout_contour_2[:, 1] - 20
                        blacked_out_frame_2 = Case_1_Processing.black_out_bottom_part(blackout_contour_2, original_gray_frame)

                        second_top_contour, second_bottom_contour = Case_1_2nd_layer.extract_contours(
                            blacked_out_frame_2, crop_location=np.min(blackout_contour_2[:, 1]), img_configuration=PATIENT_NUM)


                        if second_top_contour is not None:
                            blackout_contour_3 = second_top_contour.copy()
                            blackout_contour_3[:, 1] = blackout_contour_3[:, 1] - 10
                            blacked_out_frame_3 = Case_1_Processing.black_out_bottom_part(blackout_contour_3, original_gray_frame)

                            first_top_contour, first_bottom_contour = Case_1_2nd_layer.extract_contours(
                                blacked_out_frame_3, crop_location=np.min(blackout_contour_2[:, 1]),
                                img_configuration=PATIENT_NUM)

                            cv2.polylines(frame, [top_middle_contour], False, (255, 0, 0), 2)
                            cv2.polylines(frame, [bottom_middle_contour], False, (0, 0, 255), 2)
                            if second_top_contour is not None:
                                cv2.polylines(frame, [second_top_contour], False, (255, 0, 0), 2)
                                cv2.polylines(frame, [second_bottom_contour], False, (0, 0, 255), 2)

                            if first_top_contour is not None:
                                cv2.polylines(frame, [first_top_contour], False, (255, 0, 0), 2)
                                cv2.polylines(frame, [first_bottom_contour], False, (0, 0, 255), 2)


            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            time.sleep(1 / 20)
            videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
