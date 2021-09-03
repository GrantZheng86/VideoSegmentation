import cv2
import numpy as np
import time
import Case_2_Processing
import Case_2_top_processing
import Case_2_middle_processing
from Case_1_main import calculate_distance
from PCAsegmentation import main_wrapper

FILE_NAME = "New Videos/2-2.mp4"

def annotate_sandwich_lines(frame, l1, l2):
    l_list = [l1, l2]
    color_list = [(255, 0, 255), (0, 255, 100), (255, 255, 50)]
    counter = 0
    for l in l_list:
        center = l[1]
        length = l[0]

        end_pt = (center[0], center[1] + length)
        cv2.line(frame, center, end_pt, color_list[counter], 2)
        cv2.putText(frame, str(abs(length)), center, cv2.FONT_HERSHEY_SIMPLEX, 1, color_list[counter], 2)
        counter += 1

def get_sandwich_lines(frame, top_line, middle_layer, bottom_line):
    first_distance, center_1 = calculate_distance(top_line, middle_layer[0], frame.shape[1])
    second_distance, center_2 = calculate_distance(middle_layer[1], bottom_line, frame.shape[1])
    return (first_distance, center_1), (second_distance, center_2)

def fill_hull(hull):
    l = len(hull)
    hull_start_x = hull[0, 0]
    hull_end_x = hull[-1, 0]

    if hull_start_x > hull_end_x:
        hull = np.flipud(hull)

    toReturn_x = np.array([])
    toReturn_y = np.array([])
    for i in range(l - 1):
        curr_x = hull[i, 0]
        next_x = hull[i+1, 0]
        if (curr_x != next_x):
            curr_y = hull[i, 1]
            next_y = hull[i+1, 1]
            filling_x = np.arange(curr_x, next_x)
            fit_line = np.poly1d(np.polyfit([curr_x, next_x], [curr_y, next_y], 1))
            filling_y = fit_line(filling_x)

            toReturn_x = np.append(toReturn_x, filling_x)
            toReturn_y = np.append(toReturn_y, filling_y)

    toReturn = np.transpose(np.stack((toReturn_x, toReturn_y)))
    return toReturn.astype(np.int32)



if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Case2.avi", fourcc, fps=30, frameSize=(580, 825))
    state_list = main_wrapper(FILE_NAME, thresh=3)


    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret:
            frame = frame[140:965, 0:580, :]
            original_frame = frame.copy()
            frame, height, valid_frame = Case_2_Processing.get_bottom_two_parts(frame, counter)

            print("Frame Number : {}".format(counter))
            if counter == 37:
                print()
            if valid_frame:
                # Partitioning the bottom frame
                bt_frame_start = frame.shape[0] - height
                bt_frame = original_frame[bt_frame_start:, :, :]
                fumer, pelvis = Case_2_Processing.partition_bottom_frame(bt_frame)

                if np.abs(pelvis[0, 0] - pelvis[-1, 0]) < 175:
                    print("Fumer Length {}, Pelvis Length {}".format(np.abs(pelvis[0, 0] - pelvis[-1, 0]),
                                                                     np.abs(fumer[0, 0] - fumer[-1, 0])))
                    cv2.putText(frame, "Unreliable segmentation", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                pelvis_contour = Case_2_Processing.fill_pelvis_contour(pelvis, bt_frame)
                fumer_contour = Case_2_Processing.fill_fumer_contour(fumer, bt_frame)
                pelvis_contour_adjusted = Case_2_Processing.height_adjustment(pelvis_contour, bt_frame_start)
                fumer_contour_adjusted = Case_2_Processing.height_adjustment(fumer_contour, bt_frame_start)
                cv2.polylines(frame, [pelvis_contour_adjusted], False, (0, 255, 255), 2)

                # partitioning the top frame
                # TODO: Check if the bottom contour needs to be moved up
                top_contour, segment_height = Case_2_top_processing.get_top_element_bottom_contour(original_frame)
                if top_contour is not None:
                    cv2.polylines(frame, [top_contour], False, (0, 255, 255), 2)

                    # partitioning the middle part
                    segment_height += 50
                    middle_frame = original_frame[segment_height:bt_frame_start, :, :]
                    middle_top_contour, middle_bottom_contour = Case_2_middle_processing.partition_middle_part(middle_frame)

                    middle_top_contour[:, 1] += segment_height
                    middle_bottom_contour[:, 1] += segment_height

                    filled_top = fill_hull(top_contour)
                    filled_middle_top = fill_hull(middle_top_contour)
                    filled_middle_bottom = fill_hull(middle_bottom_contour)
                    filled_bottom = fill_hull(pelvis_contour_adjusted)
                    cv2.polylines(frame, [filled_middle_top], False, (155, 0, 255), 1)
                    cv2.polylines(frame, [filled_middle_bottom], False, (155, 0, 255), 1)
                    l1, l2 = get_sandwich_lines(frame, filled_top, (filled_middle_top, filled_middle_bottom),
                                                filled_bottom)
                    annotate_sandwich_lines(frame, l1, l2)

                else:
                    cv2.putText(frame, "Top Segmentation Failed", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                cv2.putText(frame, str(state_list[counter - 1]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)





            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            videoWriter.write(frame)

        time.sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
