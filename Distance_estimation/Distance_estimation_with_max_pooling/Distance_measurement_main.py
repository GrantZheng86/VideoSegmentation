import shutil
import skimage.measure
import cv2
import glob
import os
import numpy as np
import Distance_estimation.Distance_measurement as measurement
import matplotlib.pyplot as plt
import Distance_estimation.Detection_exception as DET
import pandas as pd

# ROOT_DIR = r'C:\Users\Grant\OneDrive - Colorado School of Mines\VideoSegmentation\ES & LM images & data Oct2021\ES & LM images & data Oct2021\Images\cropped_imgs'
ROOT_DIR = r'C:\Users\Grant\PycharmProjects\VideoSegmentation\Jul_9_analysis\Regrouped_images'
BANNER_HEIGHT = 140
RULER_WIDTH = 40
BOTTOM_HEIGHT = 200
POOLING_SIZE = 12
SLOPE_WINDOW = 5
UPPER_RATIO = 1 / 3
MINIMUM_WIDTH = 200
MINIMUM_INDEX = 2
SPINE_CONTOUR_OFFSET = -40

marked_frame_dir = os.path.join(ROOT_DIR, 'marked_frame')
if os.path.exists(marked_frame_dir):
    shutil.rmtree(marked_frame_dir)
os.mkdir(marked_frame_dir)


def crop_image_for_detection(img_gray):
    to_return = img_gray[BANNER_HEIGHT:-BOTTOM_HEIGHT, :-RULER_WIDTH]
    return to_return


def scale_contour(contour):
    """
    Scale the pooled contour back to the image size
    :param contour: The contour from the pooled image
    :return: a rescaled contour back into the original image scale
    """
    # contour = Distance_estimation.Distance_measurement.fill_contour(contour)
    contour = contour * POOLING_SIZE
    # contour = Distance_estimation.Distance_measurement.fill_contour(contour)
    return contour


def find_average_slope(contour, imshow=False):
    """
    Calculated a list of averaged contour. The average window is defined as a constant in the head of the file.
    :param contour: The contour in the original image scale
    :param imshow: Whether to show a plot of the calculated slope
    :return: a list of averaged slope
    """

    if (len(contour) < SLOPE_WINDOW):
        raise DET.DetectionException('Slope_calculation: Insufficient contour length')
    l, _ = contour.shape
    avg_slope_list = []

    for i in range(l - SLOPE_WINDOW - 1):
        curr_pt = contour[i]
        slope_list = []
        for j in range(SLOPE_WINDOW):
            next_pt = contour[i + j + 1]
            x_change = float(next_pt[0] - curr_pt[0])
            y_change = float(next_pt[1] - curr_pt[1])
            # Put a small delta to avoid divide by 0 issue
            if x_change == 0:
                x_change = 1
            curr_slope = y_change / x_change
            slope_list.append(curr_slope)
        avg_slope_list.append(np.average(slope_list))

    l = len(avg_slope_list)
    for i in range(SLOPE_WINDOW):
        curr_pt = contour[i + l]
        slope_list = []
        for j in range(SLOPE_WINDOW - i):
            try:
                next_pt = contour[i + l + j]
            except:
                print()
            x_change = float(next_pt[0] - curr_pt[0])
            y_change = float(next_pt[1] - curr_pt[1])
            # Put a small delta to avoid divide by 0 issue
            if x_change == 0:
                x_change = 1
            curr_slope = y_change / x_change
            slope_list.append(curr_slope)
        avg_slope_list.append(np.average(slope_list))

    avg_slope_list.append(avg_slope_list[-1])

    if imshow:
        plt.plot(avg_slope_list)
        plt.show()

    return avg_slope_list


def visualize_contour(img, bottom_contour_spine, top_contour_spine, top_contour=None, point_of_interest_index=None,
                      height_offset=0,
                      pixel_height=0,
                      actual_height=0, imshow=True):
    """
    Showing the contour of the sping. with critical points for slope calculation, and an optional point of interest
    :param img: The image in the original scale, can be either BW and GRAY
    :param contour: The contour in the original scale
    :param point_of_interest_index: Optional argument, the index of point in the contour
    :return: None, just show image
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bottom_contour_spine = bottom_contour_spine + [0, height_offset]
    top_contour_spine = top_contour_spine + [0, height_offset]
    if top_contour is not None:
        top_contour = top_contour + [0, height_offset]
    img_with_line = cv2.polylines(img, [bottom_contour_spine], False, (0, 255, 0), 3)
    img_with_line = cv2.polylines(img_with_line, [top_contour_spine], False, (0, 255, 0), 2)
    img_with_line = cv2.polylines(img_with_line, [top_contour], False, (255, 255, 0), 3)

    for each_pt in top_contour_spine:
        img_with_line = cv2.circle(img_with_line, (each_pt[0], each_pt[1]), 3, (0, 0, 255), 2)
    if point_of_interest_index is not None:
        point_of_interest = top_contour_spine[point_of_interest_index]
        img_with_line = cv2.circle(img_with_line, (point_of_interest[0], point_of_interest[1]), 5, (0, 255, 255), 5)
        img_with_line = cv2.putText(img_with_line, '{:.2f}'.format(actual_height), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, (0, 255, 0), 1, cv2.LINE_AA)
        img_with_line = cv2.line(img_with_line, (point_of_interest[0], point_of_interest[1]),
                                 (point_of_interest[0], int(point_of_interest[1] - pixel_height)), (0, 0, 255), 2)
        if imshow:
            cv2.imshow('Bottom Contour', img_with_line)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return img_with_line


def find_point_of_interest(slope_list):
    """
    Finds the index of point of interest from the slope list. This function can also throw exception when the contour
    slope profile does not match the shape we want.
    :param slope_list: A list of slope
    :return: The index at which the point that represents the spine contour
    """
    indices = np.arange(0, len(slope_list), 1, dtype=int)
    slope_with_indices = np.vstack((slope_list, indices))
    slope_with_indices = slope_with_indices.T
    slope_with_indices = slope_with_indices[slope_with_indices[:, 0].argsort()]

    found_bounds = False
    largest_slope = slope_with_indices[-1]
    second_largest_slope = slope_with_indices[-2]
    loop_count = 0
    while not found_bounds:
        second_largest_slope = slope_with_indices[-2 - loop_count]
        if loop_count > 4:
            raise DET.DetectionException('Point of interest searching: looping out of bounds')
        if np.abs(largest_slope[1] - second_largest_slope[1]) > 4:
            found_bounds = True
        else:
            loop_count += 1

    point_of_interest = int((largest_slope[1] + second_largest_slope[1]) / 2)
    return point_of_interest


def find_point_of_interest_1(contour, img_gray, imshow=False):
    """
    This is a new way to search for point of interest. It resembles the way that human looks for it. It's looking to
    find the brightest region first, then see the slope, and then compare the relative region in the image
    :param contour: The contour, unfilled, of the bottom of the spine
    :param img_gray: Gray image without any annotations
    :param imshow: whether to show contour slop and bounding box to search for the brightest region
    :return:
    """
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    _, x_max = img_gray.shape

    contour_slope = find_average_slope(contour, imshow=imshow)
    indices = np.arange(0, len(contour_slope), 1, dtype=int)
    intensity_value = []

    for pt in contour:
        avg = pixel_average_around_point(img_gray, pt, imshow=False)
        intensity_value.append(avg)

    intensity_value = np.array(intensity_value)
    contour_indices_intensity = np.vstack((indices, contour_slope, intensity_value))
    contour_indices_intensity = contour_indices_intensity.T

    sorted_by_intensity = contour_indices_intensity[contour_indices_intensity[:, -1].argsort()]

    point_to_return = 0
    meet_spec = False
    while not meet_spec:
        point_to_return -= 1

        if np.abs(point_to_return) - 1 >= len(sorted_by_intensity):
            raise DET.DetectionException('Template_match mailed to find')
        curr_slope = sorted_by_intensity[point_to_return][1]
        curr_index = int(sorted_by_intensity[point_to_return][0])
        prev_index = int(sorted_by_intensity[point_to_return][0] - SLOPE_WINDOW)

        prev_slope = None
        if prev_index > 0:
            prev_slope = contour_indices_intensity[prev_index][1]
        # Make sure the point of interest have change of slow nearby
        if prev_slope is not None and prev_slope <= 0 <= curr_slope:
            point_location = contour[curr_index]
            x = point_location[0]
            # Usually the point of interest will not be near the image boarder
            if x - MINIMUM_WIDTH > 0 and x + MINIMUM_WIDTH < x_max:
                if MINIMUM_INDEX < curr_index < len(sorted_by_intensity) - MINIMUM_INDEX:
                    meet_spec = True

    return int(sorted_by_intensity[point_to_return][0])


def find_point_of_interest_2(contour, img_gray, imshow=False):
    """
    Similar to "find point of interest 1" method, this one uses the highest region as the 1st judging factor
    :param contour: The contour, unfilled, of the bottom of the spine
    :param img_gray: Gray image without any annotations
    :param imshow: whether to show contour slop and bounding box to search for the brightest region
    :return:
    """
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    _, x_max = img_gray.shape

    contour_slope = find_average_slope(contour, imshow=imshow)
    indices = np.arange(0, len(contour_slope), 1, dtype=int)

    contour_indices_y = np.vstack((indices, contour_slope, contour[:, -1]))
    contour_indices_y = contour_indices_y.T

    sorted_by_y = contour_indices_y[contour[:, -1].argsort()]

    point_to_return = -1
    meet_spec = False
    while not meet_spec:
        point_to_return += 1

        if np.abs(point_to_return) - 1 >= len(sorted_by_y):
            raise DET.DetectionException('Template_match failed to find')
        curr_slope = sorted_by_y[point_to_return][1]
        curr_index = int(sorted_by_y[point_to_return][0])
        prev_index = int(sorted_by_y[point_to_return][0] - SLOPE_WINDOW)

        prev_slope = None
        if prev_index > 0:
            prev_slope = contour_indices_y[prev_index][1]
        # Make sure the point of interest have change of slow nearby
        # if prev_slope is not None and prev_slope <= 0 <= curr_slope:
        #     point_location = contour[curr_index]
        #     x = point_location[0]
        #     # Usually the point of interest will not be near the image boarder
        #     if x - MINIMUM_WIDTH > 0 and x + MINIMUM_WIDTH < x_max:
        #         if MINIMUM_INDEX < curr_index < len(sorted_by_y) - MINIMUM_INDEX:
        #             meet_spec = True


        point_location = contour[curr_index]
        x = point_location[0]
        if x - MINIMUM_WIDTH > 0 and x + MINIMUM_WIDTH < x_max:
            if MINIMUM_INDEX < curr_index < len(sorted_by_y) - MINIMUM_INDEX:
                meet_spec = True

    return int(sorted_by_y[point_to_return][0])


def pixel_average_around_point(img, point, imshow=False):
    y_lim, x_lim = img.shape
    BOX_LEN = 40
    # BOX_Y_OFFSET = -BOX_LEN / 2
    BOX_Y_OFFSET = 0
    x = point[0]
    y = point[1]
    x_min = int(np.max((x - BOX_LEN / 2, 0)))
    y_min = int(np.max((y - BOX_LEN / 2, 0)) + BOX_Y_OFFSET)
    x_max = int(np.min((x_min + BOX_LEN, x_lim)))
    y_max = int(np.min((y_min + BOX_LEN, y_lim)))

    if imshow:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_color = cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        img_color = cv2.circle(img_color, (x, y), 3, (0, 0, 255), 2)
        cv2.imshow('box boundary', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    area = img[y_min:y_max, x_min:x_max]
    return np.average(area)


def find_top_contour(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    upper_region = img[0:int(img.shape[0] * UPPER_RATIO), :]
    upper_region_pooled = skimage.measure.block_reduce(upper_region, (POOLING_SIZE, POOLING_SIZE), np.min)
    top_contour, successful_detection = measurement.find_lumbodorsal_bottom_1(upper_region)

    if successful_detection:
        return scale_contour(top_contour)

    raise DET.DetectionException('Top contour detection failed')


def convert_to_bw(frame):
    """
    Converts the current BGR frame to a BW image with BGR Channel
    :param frame: BGR image
    :return: the original image in BW with all color channel
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def scale_calculation(frame):
    """
    Calculates the conversion factor between pixel and actual distance. This method uses template matching, therefore,
    pre-defined number templates are required
    :param frame: A BGR image contains the right hand side ruler
    :return: The conversion factor
    """

    TEMPLATE_1_PATH = "../marker_templates/Template-1.png"
    TEMPLATE_2_PATH = "../marker_templates/Template-2.png"
    MARKER_PATH = "../marker_templates/marker_template.png"

    midway_height = int(frame.shape[0] / 2)

    marker_width = 40
    useful_frame = frame[146:midway_height, :, :]
    useful_frame_bw = convert_to_bw(useful_frame)

    template_1 = cv2.imread(TEMPLATE_1_PATH)
    template_1_bw = convert_to_bw(template_1)
    template_2 = cv2.imread(TEMPLATE_2_PATH)
    template_2_bw = convert_to_bw(template_2)

    # Searching for matching locations for the templates. After getting locations for those numbers, expanding the
    # region of interest to include tick marks on ruler for more precise distance calculation
    h_1, w_1, _ = template_1_bw.shape
    method = cv2.TM_CCORR_NORMED
    res_1 = cv2.matchTemplate(useful_frame_bw, template_1_bw, method)
    res_2 = cv2.matchTemplate(useful_frame_bw, template_2_bw, method)
    _, _, _, top_left_1 = cv2.minMaxLoc(res_1)
    _, _, _, top_left_2 = cv2.minMaxLoc(res_2)
    bottom_right_1_with_marker = (top_left_1[0] + w_1 + marker_width, top_left_1[1] + h_1)
    bottom_right_2_with_marker = (top_left_2[0] + w_1 + marker_width, top_left_2[1] + h_1)

    # Searching for tick marks on ruler based on template matching, it uses the spacing between top left corners as the
    # measure. This will have more precise measurement
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

    return np.abs(top_left_marker_1_abs[1] - top_left_marker_2_abs[1])


def process_marker_image(file_name):
    """
    Process the marker image on ruler so that the "sample marker" only contains the white portion. This is for better
    template matching result when it comes for scale calculation
    :param file_name: The template file for the marker
    :return: A BGR version of the shrinked BW ruler marker
    """
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


if __name__ == '__main__':
    file_name_list = []
    distance_measurement_list = []
    image_count = 0
    for raw_img in glob.glob(os.path.join(ROOT_DIR, '*.jpg')):
        shorter_img_name = raw_img.split('\\')[-1]
        if 'E' in shorter_img_name:
            image_count += 1
            img_bgr = cv2.imread(raw_img)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = crop_image_for_detection(img_gray)
            img_gray_pooled = skimage.measure.block_reduce(img_gray, (POOLING_SIZE, POOLING_SIZE), np.min)
            img_gray_pooled = np.array(img_gray_pooled, dtype=np.uint8)
            kernel = np.ones((1, 1), np.uint8)
            img_gray_pooled = cv2.morphologyEx(img_gray_pooled, cv2.MORPH_OPEN, kernel)
            bottom_contour_spine, h = measurement.get_bottom_contour(
                cv2.cvtColor(img_gray_pooled, cv2.COLOR_GRAY2BGR), reduction=False, bottom_percentile=60)
            bottom_contour_spine = scale_contour(bottom_contour_spine)
            bottom_contour_spine = np.array(bottom_contour_spine, dtype=np.int32)
            #
            top_contour_spine = bottom_contour_spine + [0, SPINE_CONTOUR_OFFSET]
            try:
                point_of_interest_index = find_point_of_interest_2(top_contour_spine, img_gray)

                point_of_interest = top_contour_spine[point_of_interest_index]
                print(shorter_img_name)
                top_contour = find_top_contour(img_gray)

                top_contour_filled = measurement.fill_contour(top_contour)
                distance_p = measurement.findDistance(point_of_interest, top_contour_filled)
                scale_cm_p = scale_calculation(img_bgr)
                distance = distance_p / scale_cm_p
                file_name_list.append(shorter_img_name)
                distance_measurement_list.append(distance)
                marked_frame = visualize_contour(img_bgr, bottom_contour_spine, top_contour_spine,
                                                 top_contour, point_of_interest_index,
                                                 height_offset=BANNER_HEIGHT, actual_height=distance,
                                                 pixel_height=distance_p, imshow=False)
                marked_frame_name = os.path.join(marked_frame_dir, shorter_img_name)
                cv2.imwrite(marked_frame_name, marked_frame)

            except:
                print('{} detection failed'.format(shorter_img_name))
                # visualize_contour(img_gray, top_contour_spine, None)
    d = {'Img': file_name_list, 'Distance': distance_measurement_list}
    df = pd.DataFrame(d)
    df.to_csv(os.path.join(ROOT_DIR, 'information.csv'), index=False)
    print(image_count)
