import cv2
import Case_4_Processing

from time import sleep
import numpy as np
from PCAsegmentation import main_wrapper
from Case_1_binary_component import Class_1_binary_cc

FILE_NAME = "New Videos/1-3.mp4"
CASE_Num = 4


def connecting_contour_points(contour):

    filled_contour = np.empty([0, 2])
    for i in range(len(contour) - 1):
        curr_pt = contour[i, :]
        next_pt = contour[i + 1, :]
        x = [curr_pt[0], next_pt[0]]

        if np.abs(x[0] - x[1]) <= 1:
            filled_contour = np.append(filled_contour, [curr_pt], axis=0)
        else:

            y = [curr_pt[1], next_pt[1]]

            linear_fit = np.polyfit(x, y, 1)
            linear_fit = np.poly1d(linear_fit)

            if x[0] < x[1]:
                fit_range = np.arange(x[0], x[1])
            else:
                fit_range = np.arange(x[1], x[0])
            linear_fit_result = linear_fit(fit_range)
            linear_fit_result = np.transpose(np.vstack((fit_range, linear_fit_result)))
            filled_contour = np.append(filled_contour, linear_fit_result, axis=0)
    return filled_contour.astype(np.int32)

def find_height_difference(top_contour, template_x_center, template_y_center):
    top_contour = connecting_contour_points(top_contour)
    top_contour_x_candidate_index = np.where(top_contour[:, 0] == template_x_center)[0]
    top_contour_candidate = np.squeeze(top_contour[top_contour_x_candidate_index, :])

    if len(top_contour_x_candidate_index) > 1:
        top_contour_y = np.min(top_contour_candidate[:, 1])
        top_contour_y_candidate_index = np.where(top_contour[:, 1] == top_contour_y)[0]

        ioi = set(top_contour_y_candidate_index) & set(top_contour_x_candidate_index)
        ioi = ioi.pop()
        poi = top_contour[ioi, :]
    else:
        poi = top_contour[top_contour_candidate[0], :]

    height_difference = template_y_center - poi[1]
    return height_difference





if __name__ == "__main__":
    state_list = main_wrapper(FILE_NAME)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Case3.avi", fourcc, fps=30, frameSize=(616, 1080))
    cap = cv2.VideoCapture(FILE_NAME)
    counter = 0
    template = None

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1

        if ret and counter == 1:
            frame = frame[140:965, :, :]
            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(state_list[counter-1]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            template = Case_4_Processing.findLandMarkFeature(frame)

        elif ret:
            cv2.putText(frame, str(state_list[counter - 1]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            top_annotation = frame[0:140, :, :]
            bottom_annotation = frame[965:, :, :]
            frame = frame[140:965, :, :]

            try:
                spine_contour = Case_4_Processing.get_spine_bottom_contour(frame, True)
            except:
                spine_contour = Case_4_Processing.get_spine_bottom_contour(frame, True, False)

            bottom_inpainted = Case_4_Processing.bottom_inpainting(spine_contour, frame)
            spine_top_contour = Case_4_Processing.get_spine_top_contour(bottom_inpainted, spine_contour)


            top_binary = Case_4_Processing.top_half_sesgmentation(frame)
            top_binary_bottom_contour = Case_4_Processing.findBottomContour(top_binary, True)
            frame = cv2.polylines(frame, [top_binary_bottom_contour], False, (0, 0, 255), 2)
            # area = int(Case_4_Processing.find_area_enclosed(top_binary_bottom_contour, spine_contour, frame))
            # print(area)
            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            h = int(frame.shape[0] / 3)
            bottom_half = frame[h:, :, :]
            top_left, bottom_right, center, max_val = Case_4_Processing.match_template(bottom_half, template)
            if max_val < 0.6:
                cv2.putText(frame, "Unreliable Tracking", (200, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 3)

            # cv2.putText(frame, "{:.2e}".format(area), (450, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)
            top_left = (top_left[0], top_left[1] + h)
            bottom_right = (bottom_right[0], bottom_right[1] + h)
            center = (center[0], center[1] + h)
            distance = Case_4_Processing.findDistance(center, top_binary_bottom_contour)
            intersect = (int(center[0]), int(center[1] - distance))
            height_difference = find_height_difference(spine_top_contour, center[0], center[1])
            cv2.line(frame, (center[0], center[1]-height_difference), intersect, (0, 0, 255), 3)

            length = np.linalg.norm([center[0]-intersect[0],  center[1]-intersect[1]])
            cv2.putText(frame, str(length), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = Case_4_Processing.annotate_frame(frame, (top_left, bottom_right))
            # spine_top_contour[:, 1] = spine_top_contour[:, 1] - 140
            cv2.polylines(frame, [spine_top_contour], False, (255, 0, 0), 2)


            frame = np.vstack((top_annotation, frame, bottom_annotation))




            # frame = cv2.drawContours(frame, [spine_top_contour], -1, (2550, 0, 0), 2)

            cv2.imshow("Template Matching", frame)
            videoWriter.write(frame)
            sleep(1 / 30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
