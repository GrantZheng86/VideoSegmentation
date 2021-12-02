import cv2
import numpy as np
from Distance_measurement import pixel_by_percentile

TEMPLATE_HEIGHT = 100
TEMPLATE_ASPECT_RATIO = 1.5
TEMPLATE_WIDTH = int(TEMPLATE_ASPECT_RATIO * TEMPLATE_HEIGHT)
TEMPLATE_CENTER_OFFSET = 75


def average_slope(contour_window):
    """
    Calculates average slope of current sliding window
    :param contour_window: The segment of contour in the current slope
    :return:
    """
    end_data = contour_window[-1]
    l = len(contour_window)

    slope_array = []
    for i in range(l - 1):
        curr_slope = (end_data - contour_window[i]) / (l - i - 1)
        slope_array.append(curr_slope)

    return np.average(slope_array)


def detect_feature(contour, window_size=4):
    """
    Finds the region of interest in the contour
    :param contour: The contour used to extract feature
    :param window_size: half of moving window size for low pass filter
    :return: The index on the contour indicating the region of interest
    """
    contour = np.squeeze(contour)
    x_vals = contour[:, 0]
    y_vals = contour[:, 1]

    l = len(x_vals)

    for i in range(l - 2 * window_size):
        back_window = y_vals[i:i + window_size]
        front_window = y_vals[i + window_size + 1:i + window_size * 2 + 1]

        back_window_average = np.average(back_window)
        front_window_average = np.average(front_window)

        back_window_slope = average_slope(back_window)
        front_window_slope = average_slope(front_window)

        curr_value = contour[i + window_size]

        if front_window_average > curr_value[1] and back_window_average > curr_value[
            1] and back_window_slope < 0 < front_window_slope:
            return i + window_size


def crop_template(index_interest_contour, spine_bottom_contour, img):
    """
    Crops the template out from the BGR image without ruler
    :param index_interest_contour: The index of interest on the contour
    :param spine_bottom_contour: The bottom half of the spine
    :param img: BGR image without ruler
    :return: Cropped template, Upper Left Template corner, bottom right template corner, Point of interest on the spine
            contour, and offseted center (actual) center of the template region
    """
    point_of_interest = spine_bottom_contour[index_interest_contour, :]
    center_offset = (point_of_interest[0], point_of_interest[1] - TEMPLATE_CENTER_OFFSET)
    x_min = center_offset[0] - TEMPLATE_WIDTH
    x_max = center_offset[0] + TEMPLATE_WIDTH
    y_min = center_offset[1] - TEMPLATE_HEIGHT
    y_max = center_offset[1] + TEMPLATE_HEIGHT
    template = img[y_min:y_max, x_min:x_max, :]

    return template, (x_min, y_min), (x_max, y_max), (point_of_interest[0], point_of_interest[1]), center_offset

def brightest_region(img, percentile=90):
    """
    Filters the template region by pixel percentile intensity
    :param img: a BGR image of the template
    :param percentile: the percentile above which that the user wants to keep
    :return: The centroid of the largest connected component after thresholding. The original image is also annotated
    """


    image_section = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_value = pixel_by_percentile(image_section, percentile)
    _, th = cv2.threshold(image_section, thresh_value, 255, cv2.THRESH_BINARY)
    largest_cc_centroid = get_largest_cc(th)
    th_color = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    th_color = cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    th_color = cv2.circle(th_color, largest_cc_centroid, 4, (255, 100, 100), -1)
    return largest_cc_centroid

    # cv2.imshow('Brightest', np.hstack((th_color, img)))
    # cv2.waitKey(0)

def get_largest_cc(th_image):
    connectivity = 4
    # Perform the operation
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(th_image, connectivity, cv2.CV_32S)

    largest_cc_area = np.max(stats[1:, -1])
    largest_cc_index = np.where(stats[:, -1] == largest_cc_area)[0][0]
    return (int(centroids[largest_cc_index, 0]), int(centroids[largest_cc_index, 1]))
