import cv2
import numpy as np
import matplotlib.pyplot as plt

BOTTOM_FEATURE_RATIO = 1.7
BOTTOM_PERCENTILE = 60
ASPECT_RATIO = 2
TEMPLATE_HEIGHT = 165


def annotate_frame(frame, corners):
    color = (0, 255, 0)
    thickness = 3
    cv2.rectangle(frame, corners[0], corners[1], color, thickness)
    return frame


def match_template(img, template):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    method = cv2.TM_CCORR_NORMED
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + int(TEMPLATE_HEIGHT * ASPECT_RATIO), top_left[1] + TEMPLATE_HEIGHT)
    center = (top_left[0] + TEMPLATE_HEIGHT, top_left[1] + int(TEMPLATE_HEIGHT / ASPECT_RATIO))

    return top_left, bottom_right, center, max_val


def extract_template(frame):
    img = frame.copy()
    bottom_half_contour, h = get_bottom_contour(img)
    feature_index = detect_feature(bottom_half_contour)
    feature_index += 1
    feature_point = bottom_half_contour[feature_index, :]
    template = crop_image_for_template(feature_point, frame)
    # cv2.imshow("Template", template)
    show_point(feature_point, frame)
    return template, h


def show_point(point, frame):
    point = np.squeeze(point)
    img = frame.copy()
    h = int((img.shape[0] / BOTTOM_FEATURE_RATIO) * (BOTTOM_FEATURE_RATIO - 1))
    x = point[0]
    y = point[1] + h
    center = (x, y)
    rad = 5
    color = (0, 255, 255)
    annotated = cv2.circle(img, center, rad, color, -1)
    cv2.imshow('Point of Interest', annotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_bottom_contour(img, reduction=True):
    original_image = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = int((img.shape[0] / BOTTOM_FEATURE_RATIO) * (BOTTOM_FEATURE_RATIO - 1))
    img = img[h:, :]
    thresh_value = pixel_by_percentile(img, BOTTOM_PERCENTILE)
    _, th = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    th = morph_operation(th, kernel_width=5, kernel_height=7)
    largest_binary = get_largest_cc(th)
    contours, _ = cv2.findContours(largest_binary, 1, 2)
    largest_contour_index = get_longest_contour(contours)
    if reduction:
        largest_contour = contour_reduction(contours[largest_contour_index])
    else:
        largest_contour = get_bottom_half(contours[largest_contour_index])
    contour_img = cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [largest_contour], -1, (0, 255, 0),
                                   2)
    # cv2.imshow("Contour", contour_img)
    # cv2.imshow("Original", original_image)

    return largest_contour, h


def morph_operation(img, kernel_height=3, kernel_width=None):
    if kernel_width is None:
        kernel_width = kernel_height

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def get_largest_cc(binary_img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    stats_copy = stats.copy()
    areas = stats_copy[:, -1]
    areas.sort()
    largest_area = areas[-2]  # -1 is for the black region

    location = np.where(stats == largest_area)
    label = location[0][0]
    largest_only = labels == label
    largest_only = np.array(largest_only, dtype=np.uint8)
    largest_only *= 255

    return largest_only


def pixel_by_percentile(img, percentile):
    img = np.array(img)
    img.flatten()
    non_zero_array = img[img != 0]
    thresh_value = np.percentile(non_zero_array, percentile)
    return thresh_value


def get_longest_contour(contours):
    max_contour_len = 0
    max_contour_index = 0
    counter = 0

    for each_contour in contours:
        if len(each_contour) > max_contour_len:
            max_contour_len = len(each_contour)
            max_contour_index = counter
        counter += 1

    return max_contour_index


def contour_reduction(largest_contour):
    """
    Approximates the contour, and extract the bottom half of it
    :param largest_contour:
    :return:
    """
    ep = 0.007 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, ep, True)
    bottom_half_contour = get_bottom_half(approx)
    # plot_contour_trend(bottom_half_contour)

    return bottom_half_contour


def get_bottom_half(contour):
    contour = np.squeeze(contour)

    contour_x = contour.copy()[:, 0]
    contour_y = contour.copy()[:, 1]

    contour_x_min = np.min(contour_x)
    contour_x_max = np.max(contour_x)

    x_min_index = np.where(contour[:, 0] == contour_x_min)
    x_max_index = np.where(contour[:, 0] == contour_x_max)
    x_min_index = x_min_index[0]
    x_max_index = x_max_index[0]

    if len(x_min_index) != 1:
        beginning_candidate_y = contour_y[x_min_index]
        max_y = np.max(beginning_candidate_y)
        beginning_index = np.where(contour_y == max_y)
        x_set_b = set(x_min_index)
        y_set_b = set(beginning_index[0])
        common_index_b = x_set_b & y_set_b
        beginning_index = common_index_b.pop()
    else:
        beginning_index = x_min_index[0]

    if len(x_max_index) != 1:
        end_candidate_y = contour_y[x_max_index]
        max_y = np.max(end_candidate_y)
        ending_index = np.where(contour_y == max_y)
        x_set_e = set(x_max_index)
        y_set_e = set(ending_index[0])
        common_index_e = x_set_e & y_set_e
        ending_index = common_index_e.pop()
    else:
        ending_index = x_max_index[0]

    return contour[beginning_index:ending_index+1, :]


def plot_contour_trend(contour):
    contour = np.squeeze(contour)
    x_vals = contour[:, 0]
    y_vals = contour[:, 1]

    plt.figure(1)
    plt.plot(x_vals)
    plt.title("X Value Trend")

    plt.figure(2)
    plt.plot(y_vals)
    plt.title("Y Value Trend")

    plt.show()


def detect_feature(contour, window_size=4):
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

        if front_window_average > curr_value[1] and back_window_average > curr_value[1] and back_window_slope < 0 < front_window_slope:
            return i + window_size


def average_slope(contour_window):
    end_data = contour_window[-1]
    l = len(contour_window)

    slope_array = []
    for i in range(l - 1):
        curr_slope = (end_data - contour_window[i]) / (l - i - 1)
        slope_array.append(curr_slope)

    return np.average(slope_array)


def crop_image_for_template(point, frame):
    h = int((frame.shape[0] / BOTTOM_FEATURE_RATIO) * (BOTTOM_FEATURE_RATIO - 1))
    x = point[0]
    y = point[1] + h

    x_min = x - TEMPLATE_HEIGHT
    x_max = x + TEMPLATE_HEIGHT
    y_min = int(y - TEMPLATE_HEIGHT / ASPECT_RATIO)
    y_max = int(y + TEMPLATE_HEIGHT / ASPECT_RATIO)

    template = frame[y_min:y_max, x_min:x_max, :]
    return template


def find_top_bottom_contour(top_portion, reduction=True):
    top_portion = cv2.cvtColor(top_portion, cv2.COLOR_BGR2GRAY)
    value = pixel_by_percentile(top_portion, 60)
    _, th = cv2.threshold(top_portion, value, 255, cv2.THRESH_BINARY)
    largest_binary = get_largest_cc(th)
    contours, _ = cv2.findContours(largest_binary, 1, 2)
    largest_contour_index = get_longest_contour(contours)
    if reduction:
        largest_contour = contour_reduction(contours[largest_contour_index])
    else:
        largest_contour = get_bottom_half(contours[largest_contour_index])
    contour_img = cv2.drawContours(cv2.cvtColor(largest_binary, cv2.COLOR_GRAY2BGR), [largest_contour], -1, (0, 255, 0),
                                   2)
    cv2.imshow('TOP', contour_img)

    return largest_contour


def correct_contour_path(contour, y_offset):
    r, _ = contour.shape
    a = np.zeros((r, 1))
    b = np.ones((r, 1)) * y_offset
    b = np.array(b, dtype=np.uint32)
    a = a.astype(int)
    to_add = np.hstack((a, b))
    contour += to_add

    return contour.astype(int)
