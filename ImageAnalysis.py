import cv2
import glob
import numpy as np
import Case_3_processing
import Case_2_Processing
import Case_3_main
from Case_4_Processing import findDistance
import Case_2_top_processing
import Case_2_middle_processing
from Case_1_main import calculate_distance

import pandas as pd

TEMPLATE_1_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
                  "ES & LM images & data Oct2021\\Template-1.png"
TEMPLATE_2_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
                  "ES & LM images & data Oct2021\\Template-2.png"
MARKER_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
              "ES & LM images & data Oct2021\\marker_template.png"
IMAGE_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
             "ES & LM images & data Oct2021\\Images"

CSV_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
           "ES & LM images & data Oct2021\\information.csv"
CASE_4 = 'E'  # Case 4 uses case 3 code
CASE_2 = 'G'


def process_marker_image(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(img, 1, 2)
    contour = np.squeeze(contour[0])
    left = np.min(contour[:, 0])
    right = np.max(contour[:, 0])
    up = np.min(contour[:, 1])
    down = np.max(contour[:, 1])

    marker_image = img[up:down, left:right]
    return cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)


def case_2_processing(frame):
    original_frame = frame.copy()
    frame, height, valid_frame = Case_2_Processing.get_bottom_two_parts(frame)
    return valid_frame


def convert_to_bw(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def scale_calculation(frame):
    marker_width = 40
    useful_frame = frame[146:366, :, :]
    # cv2.imshow('a', useful_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    useful_frame_bw = convert_to_bw(useful_frame)

    template_1 = cv2.imread(TEMPLATE_1_PATH)
    template_1_bw = convert_to_bw(template_1)
    template_2 = cv2.imread(TEMPLATE_2_PATH)
    template_2_bw = convert_to_bw(template_2)

    h_1, w_1, _ = template_1_bw.shape
    method = cv2.TM_CCORR_NORMED
    res_1 = cv2.matchTemplate(useful_frame_bw, template_1_bw, method)
    res_2 = cv2.matchTemplate(useful_frame_bw, template_2_bw, method)
    _, _, _, top_left_1 = cv2.minMaxLoc(res_1)
    _, _, _, top_left_2 = cv2.minMaxLoc(res_2)
    bottom_right_1 = (top_left_1[0] + w_1, top_left_1[1] + h_1)
    bottom_right_2 = (top_left_2[0] + w_1, top_left_2[1] + h_1)
    bottom_right_1_with_marker = (top_left_1[0] + w_1 + marker_width, top_left_1[1] + h_1)
    bottom_right_2_with_marker = (top_left_2[0] + w_1 + marker_width, top_left_2[1] + h_1)

    marker_template = process_marker_image(MARKER_PATH)
    marker_section_1 = useful_frame_bw[top_left_1[1]:bottom_right_1_with_marker[1],
                       top_left_1[0]:bottom_right_1_with_marker[0], :]
    marker_section_2 = useful_frame_bw[top_left_2[1]:bottom_right_2_with_marker[1],
                       top_left_2[0]:bottom_right_2_with_marker[0], :]
    res_marker_1 = cv2.matchTemplate(marker_section_1, marker_template, method)
    res_marker_2 = cv2.matchTemplate(marker_section_2, marker_template, method)
    _, _, _, top_left_marker_1 = cv2.minMaxLoc(res_marker_1)
    _, _, _, top_left_marker_2 = cv2.minMaxLoc(res_marker_2)

    top_left_marker_1_abs = (top_left_1[0], top_left_1[1] + top_left_marker_1[1])
    top_left_marker_2_abs = (top_left_2[0], top_left_2[1] + top_left_marker_2[1])

    return np.abs(top_left_marker_2_abs[1] - top_left_marker_2_abs[1])

    # visu = cv2.circle(useful_frame_bw, top_left_marker_1_abs, 2, (0, 255, 0), -1)
    # visu = cv2.circle(visu, top_left_marker_2_abs, 2, (0, 255, 0), -1)
    # print()


if __name__ == "__main__":

    saving_dict = {}

    for file_name in glob.glob('{}/*.png'.format(IMAGE_PATH)):
        image_name = file_name.split('\\')[-1]
        frame = cv2.imread(file_name)
        frame_with_marker = frame[147:926, 285:903, :]
        frame_no_marker = frame[147:926, 285:863, :]
        scale = scale_calculation(frame)

        if "17ECT2.png" in image_name:
            print()

        if CASE_2 in image_name:
            pass
            #
            #
            # original_frame = frame.copy()
            # valid_frame = case_2_processing(frame_no_marker)
            # print("currently processing {}, result {}".format(image_name, valid_frame))
        else:

            successful_detection = True
            print("Processing {}".format(image_name))
            try:
                template = Case_3_processing.extract_template(frame_no_marker)
                template_shape = template.shape

                if 0 in template_shape:
                    raise Exception('Not a valid template')
            except:
                successful_detection = False

            if successful_detection:
                # cv2.imshow('t', template)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                bottom_contour = Case_3_processing.get_bottom_contour(frame_no_marker, reduction=False)
                non_tracking_frame_height = int(frame.shape[0] * Case_3_main.TEMPLATE_TRACKING_FRAME_RATIO)
                tracking_frame = frame[non_tracking_frame_height:, :, :]
                non_tracking_frame = frame[0:non_tracking_frame_height, :, :]
                top_left, bottom_right, center, max_val = Case_3_processing.match_template(tracking_frame, template)
                tracking_frame = Case_3_processing.annotate_frame(tracking_frame, (top_left, bottom_right))

                bottom_contour = Case_3_processing.correct_contour_path(bottom_contour, non_tracking_frame_height)
                if max_val < 0.81:
                    cv2.putText(frame, "Unreliable Tracking", (200, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 3)

                top_contour = Case_3_processing.find_top_bottom_contour(non_tracking_frame, reduction=False)
                center = (center[0], center[1] + non_tracking_frame_height)
                distance = findDistance(center, top_contour)
                intersect = (int(center[0]), int(center[1] - distance))
                cv2.line(frame, center, intersect, (0, 0, 255), 3)
                cv2.putText(frame, str(distance), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                encirclement = np.concatenate((np.flip(top_contour, 0), bottom_contour))
                frame = np.vstack((non_tracking_frame, tracking_frame))
                saving_dict[image_name] = [distance]

                # cv2.imshow("Template Matching", frame)
            else:
                print("{} detection failed".format(image_name))
                saving_dict[image_name] = [-1]

    to_save_df = pd.DataFrame.from_dict(saving_dict, orient='index')
    to_save_df.to_csv(CSV_PATH)
