import numpy as np

TEMPLATE_HEIGHT = 75
TEMPLATE_ASPECT_RATIO = 2
TEMPLATE_WIDTH = TEMPLATE_ASPECT_RATIO * TEMPLATE_HEIGHT
TEMPLATE_CENTER_OFFSET = 50


def average_slope(contour_window):
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
