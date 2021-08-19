import cv2
import numpy as np
from Case_1_binary_component import Class_1_binary_cc
from Case_2_Processing import sort_component_by_area, pixel_by_percentile
from skimage import measure

BOTTOM_STARTING_HEIGHT = 300
THRESHOLD_VALUE = 30
SMALL_AREA_THRESH = 10000
LARGE_AREA_THRESH = SMALL_AREA_THRESH * 20
BOTTOM_CUTOFF_PERCENTILE = 80


def rearrange_hull_contour(top_hull):
    hull_1 = top_hull[0]
    hull_2 = top_hull[-1]

    if hull_1[1] > hull_2[1]:
        top_hull = np.flipud(top_hull)
        return top_hull, True

    return top_hull, False


def extend_top_contour(top_contour, top_hull, flipped, gray_frame):
    if flipped:
        end_pixel_count = 0
        begin_pixel_count = gray_frame.shape[1] - 1
    else:
        end_pixel_count = gray_frame.shape[1] - 1
        begin_pixel_count = 0
    rearranged_top_contour, _ = rearrange_hull_contour(top_contour)

    if len(top_hull) == 2:
        # Case of only 2 points on top convex hull
        contour_x_beg = rearranged_top_contour[0, 0]
        contour_x_end = rearranged_top_contour[-1, 0]

        beg_full = True
        end_full = True
        if begin_pixel_count != contour_x_beg:
            beg_full = False
        if end_pixel_count != contour_x_end:
            end_full = False

        linear_fit_all = np.polyfit(top_hull, deg=1)
        linear_fit_all = np.poly1d(linear_fit_all)

        if not beg_full:
            begin_filler = generate_linear_fit_points(begin_pixel_count, contour_x_beg, linear_fit_all)
        if not end_full:
            end_filler = generate_linear_fit_points(end_pixel_count, contour_x_end, linear_fit_all)

    else:
        rearranged_top_hull, _ = rearrange_hull_contour(top_hull)
        valid_final_slope = detect_truncation(rearranged_top_hull)
        # TODO: Truncate the contour and fill it up if the final slope is not valid
        wanted_top_contour = get_good_portion(top_hull, top_contour, flipped)

    filled_contour = np.concatenate((begin_filler, top_contour, end_filler))
    return filled_contour


def get_good_portion(top_hull, top_contour, flipped):
    """
    This method is intended to delete final portions that has slope significantly different from the major slope
    Both top_hull and top_contour needs to be rearranged. I.E. Y value must range from small to large.

    :param top_hull:
    :param top_contour:
    :param flipped:
    :return:
    """
    top_contour, _ = rearrange_hull_contour(top_contour)
    top_hull_end = top_hull[-2, 0]
    top_hull_begin = top_hull[0, 0]

    if not flipped:
        valid_contour_mask = np.where(top_contour[:, 0] > top_hull_end)
        contour_left = np.squeeze(top_contour[valid_contour_mask, :])
        valid_contour_mask = np.where(contour_left[:, 0] <= top_hull_begin)
        contour_left = np.squeeze(contour_left[valid_contour_mask, :])
    else:
        valid_contour_mask = np.where(top_contour[:, 0] < top_hull_end)
        contour_left = np.squeeze(top_contour[valid_contour_mask, :])
        valid_contour_mask = np.where(contour_left[:, 0] >= top_hull_begin)
        contour_left = np.squeeze(contour_left[valid_contour_mask, :])

    return contour_left


def generate_linear_fit_points(begin_num, end_num, fit_equation):
    flipped = False
    if begin_num > end_num:
        smaller = end_num
        larger = begin_num
        flipped = True
    else:
        smaller = begin_num
        larger = end_num

    x_list = np.arange(smaller, larger)
    y_list = fit_equation(x_list)

    toReturn = np.transpose(np.vstack((x_list, y_list)))
    if flipped:
        np.flipud(toReturn)
    return toReturn


def detect_truncation(top_hull):
    """
    Determine if a end truncation is necessary. Top Hull needs to be a rearranged top_hull, I.E. y value from small to
    large
    :param top_hull:
    :return:
    """
    weighted_slope_list = hull_slope_with_weight(top_hull)
    max_weight = np.max(weighted_slope_list[:, 1])
    max_weight_index = np.where(weighted_slope_list[:, 1] == max_weight)[0][0]
    max_slope = weighted_slope_list[max_weight_index, 0]
    final_slope = weighted_slope_list[-1, 0]

    abs_difference = abs(abs(max_slope) - abs(final_slope))
    if abs_difference > 0.2:
        return True
    else:
        return False



def hull_slope_with_weight(top_hull):
    l = len(top_hull)

    weighted_slope_list = []
    for i in range(l - 1):
        curr_point = top_hull[i]
        next_point = top_hull[i + 1]

        delta_x = next_point[0] - curr_point[0]
        delta_y = next_point[1] - curr_point[1]

        slope = delta_y / delta_x
        weight = np.linalg.norm([delta_x, delta_y])
        weighted_slope_list.append([slope, weight])

    return np.array(weighted_slope_list)


def post_processing_bottom_segmentation(gray_frame, segmentation_height):
    h, w = gray_frame.shape
    cutoff_height = h - segmentation_height
    bottom_frame = gray_frame[cutoff_height:, :]
    cutoff_pixel = pixel_by_percentile(bottom_frame, BOTTOM_CUTOFF_PERCENTILE)
    _, bw = cv2.threshold(bottom_frame, cutoff_pixel, 255, cv2.THRESH_BINARY)

    cv2.imshow('post processing', bw)


def self_multiply(gray_frame):
    gray_frame = gray_frame.astype(np.int32)
    gray_frame = np.multiply(gray_frame, gray_frame)
    gray_frame = gray_frame / np.amax(gray_frame)
    gray_frame = gray_frame * 255
    gray_frame = gray_frame.astype(np.uint8)
    cv2.imshow('Self Multiply', gray_frame)
    return gray_frame


def bottom_segmentation_recursion_helper(gray_frame, segment_height=BOTTOM_STARTING_HEIGHT):
    h, w = gray_frame.shape

    if segment_height >= h / 3 * 2:
        return -1, None, None

    segment_start = h - segment_height
    image_to_analyze = gray_frame[segment_start:, :]
    _, binary_img = cv2.threshold(image_to_analyze, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    sorted_cc_list = create_sorted_binary_cc_list(binary_img)

    if len(sorted_cc_list) == 0:
        # Cutting section is too small, no element in the list
        return bottom_segmentation_recursion_helper(gray_frame, segment_height=segment_height + 1)

    largest_cc = sorted_cc_list[-1]
    aspect_ratio = largest_cc.minimum_bounding_rectangle()

    if not largest_cc.valid_test():
        # If the current capturing a section other than the interested
        return bottom_segmentation_recursion_helper(gray_frame, segment_height=segment_height + 1)

    if largest_cc.area_cutoff:
        return bottom_segmentation_recursion_helper(gray_frame, segment_height=segment_height + 1)
    else:
        largest_cc.visualize_with_contour(False)
        return segment_height, largest_cc.get_hull_top(), largest_cc.get_contour_top()


def create_sorted_binary_cc_list(binary_image):
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    binary_cc_list = []
    for i in range(num_cc - 1):
        curr_index = i + 1
        curr_stats = stats[curr_index, :]
        curr_area = curr_stats[-1]

        if LARGE_AREA_THRESH > curr_area > SMALL_AREA_THRESH:
            curr_binary_cc = Class_1_binary_cc(labels, curr_index)
            binary_cc_list.append(curr_binary_cc)

    sorted_cc_list = sort_component_by_area(binary_cc_list)
    return sorted_cc_list
