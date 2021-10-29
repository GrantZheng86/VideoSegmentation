import cv2
import numpy as np
import matplotlib.pyplot as plt
import Case_2_Processing

TEMPLATE_SIZE = 150
TOP_IMG_CROP = None
BOTTOM_DETECTION_RATIO = 2
ASPECT_RATIO = 2
SPINE_OFFSET_VALUE = 75
SPINE_THICKNESS = 30


def get_spine_bottom_contour(img, absolute_position=False, connect_component=True):
    original_img = img.copy()
    morph_closed = bottom_thresholding(img)
    largest_connected = get_largest_connected_comp(morph_closed)
    bottom_contour = findBottomContour(largest_connected)
    good_contour = check_landmark_bottom_contour(bottom_contour, original_img)

    if not good_contour and connect_component:
        bottom_contour = connect_broken_contour(bottom_contour, morph_closed)

    if absolute_position:
        r, _ = bottom_contour.shape
        a = np.zeros((r, 1))
        b = np.ones((r, 1)) * img.shape[0] / BOTTOM_DETECTION_RATIO
        b = np.array(b, dtype=np.uint32)
        to_add = np.hstack((a, b))
        bottom_contour = bottom_contour + to_add

    return bottom_contour.astype(int)


def findLandMarkFeature(img):
    original_img = img.copy()
    bottom_contour = get_spine_bottom_contour(img)
    plt.figure(1)
    plt.plot(bottom_contour[:, 1])

    point = findPointOfInterest(bottom_contour[:, 1])
    point_location = bottom_contour[point, :]
    template = crop_image_for_feature(point_location, original_img)
    cv2.imshow("Template to track", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return template

    # cv2.imshow("Location", interesting_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def bottom_half_segmentation(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    h = round(gray.shape[0] / BOTTOM_DETECTION_RATIO)
    gray = gray[h:]


def check_landmark_bottom_contour(bottom_contour, original_img):
    """
    Checks if there's breakage for the bottom curve. If the contour begins after 1/3 of the width or ends before 1/3 of
    the width, it is broken.
    :param bottom_contour:
    :param original_img:
    :return: True if not broken, False otherwise
    """
    x_size = original_img.shape[1]
    x_start_lim = x_size / 3
    x_end_lim = 2 * x_start_lim

    x_init = bottom_contour[0][0]
    x_end = bottom_contour[-1][0]

    if x_init > x_start_lim or x_end < x_end_lim:
        return False
    else:
        return True


def connect_broken_contour(largest_cc_contour, bottom_binary_image):
    second_largest = get_largest_connected_comp(bottom_binary_image, -3)  # -3 indicates the 2nd largest component
    second_bottom_contour = findBottomContour(second_largest)

    second_x_start = second_bottom_contour[0][0]
    first_x_start = largest_cc_contour[0][0]

    if second_x_start > first_x_start:
        to_return = np.concatenate((largest_cc_contour, second_bottom_contour))
    else:
        to_return = np.concatenate((second_bottom_contour, largest_cc_contour))

    return to_return


def annotate_frame(frame, corners):
    color = (0, 255, 0)
    thickness = 3;
    cv2.rectangle(frame, corners[0], corners[1], color, thickness)
    return frame


def match_template(img, template):
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + TEMPLATE_SIZE * 2, top_left[1] + TEMPLATE_SIZE)
    center = (top_left[0] + TEMPLATE_SIZE, top_left[1] + int(TEMPLATE_SIZE / 2))

    return top_left, bottom_right, center, max_val


def crop_image_for_feature(point, img, template_y=TEMPLATE_SIZE):
    y_offset = img.shape[0] / 2
    y_center = point[1] + y_offset
    x_center = point[0]

    y_min = int(y_center - template_y / ASPECT_RATIO)
    y_max = int(y_center + template_y / ASPECT_RATIO)
    x_min = x_center - template_y
    x_max = x_center + template_y

    template = img[y_min:y_max, x_min:x_max, :]
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(template)
    # plt.show()
    return template


def bottom_thresholding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = round(gray.shape[0] / BOTTOM_DETECTION_RATIO)
    gray = gray[h:]
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = morph_operation(th_otsu)
    return closed


def morph_operation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def extend_contour(contour, img):
    h, w, _ = img.shape
    beg_x = contour[0, 0]
    beg_y = contour[0, 1]
    end_x = contour[-1, 0]
    end_y = contour[-1, 1]

    if (beg_x > end_x):
        contour = np.insert(contour, 0, [w, beg_y], axis=0)
    else:
        contour = np.append(contour, [[w, end_y]], axis=0)

    return contour


def bottom_inpainting(contour, frame):
    img = frame.copy()
    h, w, _ = img.shape
    beg_x = contour[0, 0]
    end_x = contour[-1, 0]

    left_corner = [0, h]
    right_corner = [w, h]
    if beg_x > end_x:
        contour = np.insert(contour, 0, right_corner, axis=0)
        contour = np.append(contour, [left_corner], axis=0)
    else:
        contour = np.insert(contour, 0, left_corner, axis=0)
        contour = np.append(contour, [right_corner], axis=0)

    img = cv2.drawContours(img, [contour], -1, (0, 0, 0), -1)
    return img


def get_spine_top_contour(img, bottom_contour):
    img_height, _, _ = img.shape
    y_min = np.min(bottom_contour[:, 1])
    crop_height = y_min - SPINE_OFFSET_VALUE
    img_crop = img[crop_height:, :, :]
    img_crop_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    frame_array = np.array(img_crop_gray)
    frame_array.flatten()
    non_zero_array = frame_array[frame_array != 0]
    thresh_value = np.percentile(non_zero_array, 70)

    _, th = cv2.threshold(img_crop_gray, thresh_value, 255, cv2.THRESH_BINARY)
    sorted_binary_cc_list = Case_2_Processing.sort_component_by_area(Case_2_Processing.get_binary_cc(th))
    largest = sorted_binary_cc_list[-1]
    top_contour = largest.get_contour_top()
    top_contour[:, 1] = top_contour[:, 1] + crop_height
    a = cv2.polylines(img, [top_contour], False, (255, 0, 0), 2)
    cv2.imshow('a', a)
    return top_contour


def get_largest_connected_comp(binary_img, component_number=-2):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    stats_copy = stats.copy()
    areas = stats_copy[:, -1]
    areas.sort()
    largest_area = areas[component_number]

    location = np.where(stats == largest_area)
    label = location[0][0]
    largest_only = labels == label
    largest_only = np.array(largest_only, dtype=np.uint8)
    largest_only *= 255

    return largest_only


def findBottomContour(binary_image, imshow=False):
    contours, _ = cv2.findContours(binary_image, 1, 2)

    counter = 0
    max_contour_len = 0
    max_contour_index = 0
    for each_contour in contours:
        if len(each_contour) > max_contour_len:
            max_contour_len = len(each_contour)
            max_contour_index = counter
        counter += 1

    largest_contour = np.squeeze(contours[max_contour_index])
    contour_x = largest_contour.copy()[:, 0]
    contour_y = largest_contour.copy()[:, 1]

    contour_x_min = np.min(contour_x)
    contour_x_max = np.max(contour_x)

    begin_index = np.where(largest_contour[:, 0] == contour_x_min)
    end_index = np.where(largest_contour[:, 0] == contour_x_max)
    begin_index = begin_index[0][0]
    end_index = end_index[0][0]
    bottom = largest_contour[begin_index:end_index, :]

    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    cv2.polylines(color_image, [bottom], False, (0, 255, 0), 3)

    if imshow:
        # cv2.drawContours(color_image, contours, max_contour_index , color=(0, 255, 0), thickness=5)
        cv2.imshow("Contours", color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return bottom


def find_y_derivative(interval, bottom_curve):
    l = bottom_curve.shape[0]
    derivative_list = []
    deri_smooth_list = []

    for i in range(l - interval):
        front = bottom_curve[i, 1]
        end = bottom_curve[i + interval, 1]
        deri_smppth = smooth_derivative(bottom_curve[i:i + 50, 1])
        deri_smooth_list.append(deri_smppth)
        deri = (end - front) / interval
        derivative_list.append(deri)
    plt.figure(2)
    plt.plot(derivative_list)
    plt.plot(deri_smooth_list)
    plt.show()

    return derivative_list


def smooth_derivative(point_interval, ratio_to_count=0.5):
    l = len(point_interval)
    stop_index = int(ratio_to_count * l)

    slope_list = []
    # for i in range(l - stop_index):
    #     diff = point_interval[-1] - point_interval[i]
    #     slope = diff / (l - i - 1)
    #     slope_list.append(slope)

    # Low Pass Filter
    for i in range(40):
        slope = point_interval[l - i - 1] - point_interval[l - i - 2]
        slope_list.append(slope)

    return np.average(slope_list)


def find_y_second_derivative(derivative_list, interval):
    l = len(derivative_list)
    second_derivative_list = []
    for i in range(l - interval):
        front = derivative_list[i]
        end = derivative_list[i + interval]
        second_derivative_list.append((front - end) / interval)

    plt.plot(second_derivative_list)
    plt.show()
    return second_derivative_list


def analyze_window(window):
    window = np.array(window)
    avg = np.average(window)

    first_value = window[0]
    slope_list = []
    for i in range(20, len(window) - 1):
        curr_value = window[i + 1]
        slope = (curr_value - first_value) / (i + 1)
        slope_list.append(slope)

    avg_slope = np.average(slope_list)
    return avg, avg_slope


def findPointOfInterest(bottom_contour, window_size=50):
    """
    Front and back windows needs to be extracted and compared for slopes
    Front slope need to be smaller than 0, back slope needs to be greater than 0
    Both front and back window need to have greater average value than current
    :param bottom_contour:
    :param points_list:
    :param window_size:
    :return:
    """

    total_stats = []
    l = len(bottom_contour)
    for i in range(l - 2 * window_size):

        curr_index = i + window_size
        front_window = bottom_contour[i:curr_index]
        back_window = bottom_contour[(curr_index + 1):(curr_index + 1 + window_size)]
        front_slope = (front_window[-1] - front_window[1]) / window_size
        back_slope = (back_window[-1] - back_window[1]) / window_size
        front_avg = np.average(front_window)
        back_avg = np.average(back_window)

        curr_stats = [front_avg, front_slope, bottom_contour[curr_index], back_avg, back_slope]
        total_stats.append(curr_stats)

        if (curr_index == 290):
            print()

        if front_slope < 0 and back_slope > 0 and bottom_contour[curr_index] < front_avg and bottom_contour[
            curr_index] < back_avg:
            return int(curr_index + window_size / 2)


def top_half_sesgmentation(img):
    h = int(img.shape[0] / 4)
    frame = img[0:h, :, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_array = np.array(frame)
    frame_array.flatten()
    non_zero_array = frame_array[frame_array != 0]
    thresh_value = np.percentile(non_zero_array, 75)
    _, th = cv2.threshold(frame, thresh_value, 255, cv2.THRESH_BINARY)
    return th


def fill_contour(contour):
    contour_x = contour[:, 0]
    contour_y = contour[:, 1]

    toReturn_x = []
    toReturn_y = []

    for i in range(len(contour_x) - 1):
        curr_x = contour_x[i]
        next_x = contour_x[i + 1]
        curr_y = contour_y[i]
        next_y = contour_y[i + 1]

        if abs(next_x - curr_x) > 1:
            x_array = [curr_x, next_x]
            y_array = [curr_y, next_y]
            z = np.polyfit(x_array, y_array, 1)
            f = np.poly1d(z)

            for x in range(curr_x, next_x):
                y = f(x)
                toReturn_x.append(x)
                toReturn_y.append(y)

        else:
            toReturn_x.append(curr_x)
            toReturn_y.append(curr_y)

    toReturn_x.append(contour_x[-1])
    toReturn_y.append(contour_y[-1])

    toReturn = np.row_stack((toReturn_x, toReturn_y))
    toReturn = np.transpose(toReturn.astype(np.uint32))
    return toReturn


def findDistance(center, bottom_contour, height_offset=0):
    """
    Find the distance from the template center (spine bump) to the top bottom contour.
    :param center: The center location for the template, i.e. the spine bump
    :param bottom_contour: The bottom contour of the upper portion
    :param height_offset:
    :return:
    """
    x = center[0]
    bottom_contour = fill_contour(bottom_contour)
    bottom_contour_x = bottom_contour[:, 0]
    bottom_contour_y = bottom_contour[:, 1]

    max_x = np.max(bottom_contour_x)

    if (x > max_x):
        return -1

    range_beginning = 8

    in_range_index = np.where((x <= bottom_contour_x) & (bottom_contour_x < x + range_beginning))
    while len(in_range_index[0]) <= 5:
        range_beginning += 1
        in_range_index = np.where((x <= bottom_contour_x) & (bottom_contour_x < x + range_beginning))

    in_range_y = bottom_contour_y[in_range_index]
    avg_y = np.average(in_range_y) + height_offset
    distance = center[1] - avg_y

    return distance


def find_area_enclosed(top_contour, bottom_contour, original_image):
    img = original_image.copy()
    encirclement = np.concatenate((np.flip(top_contour, 0), bottom_contour))
    cv2.drawContours(original_image, [encirclement], -1, (0, 255, 225), 1)
    gray_img = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2GRAY)
    zeros = np.zeros(gray_img.shape)
    cv2.drawContours(zeros, [encirclement], -1, 255, -1)
    # cv2.imshow('Lines', zeros)

    return np.sum(zeros)
