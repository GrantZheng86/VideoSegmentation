import cv2
import time
import numpy as np
import Case_1_Processing
import Case1_Middle_Layer
import Case_1_2nd_layer
from skimage import measure
import matplotlib.pyplot as plt

PATIENT_NUM = 2
FILE_NAME = "New Videos/{}-1.mp4".format(PATIENT_NUM)


def annotate_sandwich_lines(frame, l1, l2, l3):
    l_list = [l1, l2, l3]

    color_list = [(255, 0, 255), (0, 255, 100), (255, 255, 50)]
    counter = 0
    for l in l_list:
        center = l[1]
        length = l[0]

        end_pt = (center[0], center[1] + length)
        cv2.line(frame, center, end_pt, color_list[counter], 2)
        cv2.putText(frame, str(abs(length)), center, cv2.FONT_HERSHEY_SIMPLEX, 1, color_list[counter], 2)
        counter += 1


def get_sandwich_lines(first_layer, second_layer, third_layer, bottom_layer, frame_width):
    layer_1_distance, center_1 = calculate_distance(first_layer[1], second_layer[0], frame_width)
    layer_2_distance, center_2 = calculate_distance(second_layer[1], third_layer[0], frame_width)
    layer_3_distance, center_3 = calculate_distance(third_layer[1], bottom_layer, frame_width)

    return (layer_1_distance, center_1), (layer_2_distance, center_2), (layer_3_distance, center_3)


def calculate_distance(top_line, bottom_line, frame_width):
    preferred_x = int(frame_width / 2)

    bottom_center_index = None
    while bottom_center_index is None:
        bottom_center_index = np.where(bottom_line[:, 0] == preferred_x)[0]

        if len(bottom_center_index) == 0:
            preferred_x += 1
            bottom_center_index = None
        else:
            bottom_center_index = bottom_center_index[0]

    bottom_center_x = int(bottom_line[bottom_center_index, 0])
    bottom_center_y = int(bottom_line[bottom_center_index, 1])
    base_center = (int(bottom_center_x), int(bottom_center_y))

    top_line_y = top_line[:, 1]
    return int(np.average(top_line_y - bottom_center_y)), base_center


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == "__main__":
    # Extended top contour is the bottom most one
    # top middle contour is the third one from the top
    # second top_contour is the second one from the top
    # first top_contour is the first one from the top

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
                top_hull = top_hull[top_hull[:, 0].argsort()]

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
                    Case1_Middle_Layer.correct_extended_contour(top_middle_contour, bottom_middle_contour,
                                                                frame.shape[1])

                    if top_middle_contour is not None:
                        blackout_contour_2 = top_middle_contour.copy()
                        blackout_contour_2[:, 1] = blackout_contour_2[:, 1] - 20
                        blacked_out_frame_2 = Case_1_Processing.black_out_bottom_part(blackout_contour_2,
                                                                                      original_gray_frame)

                        second_top_contour, second_bottom_contour = Case_1_2nd_layer.extract_contours(
                            blacked_out_frame_2, crop_location=np.min(blackout_contour_2[:, 1]),
                            img_configuration=PATIENT_NUM)

                        if second_top_contour is not None:
                            blackout_contour_3 = second_top_contour.copy()
                            blackout_contour_3[:, 1] = blackout_contour_3[:, 1] - 10
                            blacked_out_frame_3 = Case_1_Processing.black_out_bottom_part(blackout_contour_3,
                                                                                          original_gray_frame)

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

                            l1, l2, l3 = get_sandwich_lines([first_top_contour, first_bottom_contour],
                                                            [second_top_contour, second_bottom_contour],
                                                            [top_middle_contour, bottom_middle_contour],
                                                            extended_top_contour, frame.shape[1])
                            annotate_sandwich_lines(frame, l1, l2, l3)

            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            time.sleep(1 / 20)
            videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
