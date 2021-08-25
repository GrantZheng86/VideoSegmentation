import cv2
from Case_1_binary_component import Class_1_binary_cc
import numpy as np

THRESHOLD_VAL = 110
MINIMUM_AREA = 5000


def extract_contour(gray_frame):
    _, bw = cv2.threshold(gray_frame, THRESHOLD_VAL, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    sorted_cc_list = create_binary_cc_list(bw)
    bottom_cc = get_lowest_binary_cc(sorted_cc_list)

    top_hull = bottom_cc.get_hull_top()
    bottom_hull = bottom_cc.get_hull_bottom()

    top_contour = bottom_cc.get_contour_top()
    bottom_contour = bottom_cc.get_contour_bottom()

    return top_contour, bottom_contour


def correct_extended_contour(middle_top_contour, middle_bottom_contour, frame_width):
    pixel_offset_value = 10

    x = np.arange(0, frame_width)

    for i in x:
        top_index = np.where(middle_top_contour[:, 0] == i)[0]
        bottom_index = np.where(middle_bottom_contour[:, 0] == i)[0]

        for each_index in top_index:
            top_y = middle_top_contour[each_index, 1]

            for each_bt_index in bottom_index:
                bottom_y = middle_bottom_contour[each_bt_index, 1]
                if bottom_y - top_y <= pixel_offset_value:
                    middle_top_contour[top_index, 1] = bottom_y - pixel_offset_value


def create_binary_cc_list(binary_image):
    """
    Creates binary connected component list whose area is greater than the minimum requirement
    :param binary_image:
    :return:
    """
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    binary_cc_list = []
    for i in range(num_cc - 1):
        curr_index = i + 1
        curr_stats = stats[curr_index, :]
        curr_area = curr_stats[-1]

        if curr_area > MINIMUM_AREA:
            curr_binary_cc = Class_1_binary_cc(labels, curr_index)
            binary_cc_list.append(curr_binary_cc)

    return binary_cc_list


def get_lowest_binary_cc(cc_list):
    lowest_y = 0
    toReturn = None
    for each_cc in cc_list:
        y = each_cc.centroid[1]
        area = each_cc.area

        if y > lowest_y and area > MINIMUM_AREA:
            lowest_y = y
            toReturn = each_cc

    return toReturn


def post_process_bottom_contour(contour):
    end_left = np.min(contour[:, 0])
    end_right = np.max(contour[:, 0])

    left_index_x = np.where(contour[:, 0] == end_left)
    right_index_x = np.where(contour[:, 0] == end_right)

    end_left_lower = np.max(contour[left_index_x, 1])
    end_right_lower = np.max(contour[right_index_x, 1])

    left_index_y = np.where(contour[:, 1] == end_left_lower)
    right_index_y = np.where(contour[:, 1] == end_right_lower)

    left_index = np.intersect1d(left_index_x, left_index_y)[0]
    right_index = np.intersect1d(right_index_x, right_index_y)[0]

    if right_index > left_index:
        larger_index = right_index
        smaller_index = left_index
    else:
        larger_index = left_index
        smaller_index = right_index

    processed_contour = contour[smaller_index:larger_index, :]
    return processed_contour[10:-10, :]


def post_process_top_contour(contour):
    end_left = np.min(contour[:, 0])
    end_right = np.max(contour[:, 0])

    left_index_x = np.where(contour[:, 0] == end_left)
    right_index_x = np.where(contour[:, 0] == end_right)

    end_left_upper = np.min(contour[left_index_x, 1])
    end_right_upper = np.min(contour[right_index_x, 1])

    left_index_y = np.where(contour[:, 1] == end_left_upper)
    right_index_y = np.where(contour[:, 1] == end_right_upper)

    left_index = np.intersect1d(left_index_x, left_index_y)[0]
    right_index = np.intersect1d(right_index_x, right_index_y)[0]

    if right_index > left_index:
        larger_index = right_index
        smaller_index = left_index
    else:
        larger_index = left_index
        smaller_index = right_index

    processed_contour = contour[smaller_index:larger_index, :]
    return processed_contour[10:-10, :]


def fill_contour(contour, frame_width):
    contour_bound_1 = contour[0, :]
    contour_bound_2 = contour[-1, :]

    if contour_bound_1[0] > contour_bound_2[0]:
        left_bound = contour_bound_2
        right_bound = contour_bound_1
        left_to_right = False
    else:
        right_bound = contour_bound_2
        left_bound = contour_bound_1
        left_to_right = True

    if left_bound[0] != 0 or right_bound[0] != frame_width - 1:
        x = [left_bound[0], right_bound[0]]
        y = [left_bound[1], right_bound[1]]
        linear_approx = np.polyfit(x, y, 1)
        linear_approx = np.poly1d(linear_approx)

    if left_bound[0] != 0:
        x_left_range = np.arange(0, left_bound[0])
        y_fit_left = linear_approx(x_left_range)
        contour_fit_left = np.transpose(np.vstack((x_left_range, y_fit_left)))
    if right_bound[0] != frame_width - 1:
        x_right_range = np.arange(right_bound[0], frame_width)
        y_fit_right = linear_approx(x_right_range)
        contour_fit_right = np.transpose(np.vstack((x_right_range, y_fit_right)))

    if left_to_right:
        filled_contour = np.concatenate((contour_fit_left, contour, contour_fit_right), axis=0)
    else:
        contour_fit_left = np.flipud(contour_fit_left)
        contour_fit_right = np.flipud(contour_fit_right)
        filled_contour = np.concatenate((contour_fit_right, contour, contour_fit_left), axis=0)

    return filled_contour.astype(np.int32)


def remove_bump(half_contour):
    # Idea: Using cross product to determine if there's vectors deviating from the course in the looking forward window
    # NOTE: DOES NOT SEEM TO WORK BASED ON PRINTING

    deviating_threshold = 0.1

    l = len(half_contour)
    forward_window = int(l / 10)

    average_list = []
    for i in range(l - forward_window):

        curr_point = half_contour[i, :]
        vector_list = []
        for j in range(forward_window):
            next_point = half_contour[i + j + 1, :]
            curr_vector = next_point - curr_point
            curr_vector = curr_vector / np.linalg.norm(curr_vector)
            vector_list.append(curr_vector)

        base_vector = np.append(vector_list[0], [0])
        vector_cross_list = []
        for j in range(len(vector_list) - 1):
            curr_vector = np.append(vector_list[j + 1], [0])
            cross_product = np.cross(base_vector, curr_vector)
            vector_cross_list.append(np.linalg.norm(cross_product))

        cross_average = np.average(vector_cross_list)
        average_list.append(cross_average)

    max_avg_cross_product = np.max(average_list)
    print("Maximum Cross Product Value {}".format(max_avg_cross_product))
    print()
