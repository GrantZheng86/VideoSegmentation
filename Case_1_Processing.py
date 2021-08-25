import cv2
import numpy as np
from Case_1_binary_component import Class_1_binary_cc
from Case_2_Processing import sort_component_by_area, pixel_by_percentile
from skimage import measure

BOTTOM_STARTING_HEIGHT = 300
THRESHOLD_VALUE = 25
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
    frame_width = gray_frame.shape[1]
    rearranged_top_contour, _ = rearrange_hull_contour(top_contour)

    if len(top_hull) == 2:
        pass
    else:
        rearranged_top_hull, _ = rearrange_hull_contour(top_hull)
        truncate_final_portion = detect_truncation(rearranged_top_hull)
        if truncate_final_portion:
            truncated_top_contour = truncate_top_contour(rearranged_top_contour, rearranged_top_hull)
            extended_top_contour = fill_truncated_top_contour(rearranged_top_hull, truncated_top_contour, frame_width)
            extended_top_contour = remove_loop_contour(extended_top_contour)
            return extended_top_contour

    rearranged_top_hull, _ = rearrange_hull_contour(top_hull)
    extended_top_contour = fill_untruncated_top_contour(rearranged_top_hull, rearranged_top_contour, frame_width)
    return extended_top_contour


def remove_loop_contour(extended_contour):
    flipped = False
    if extended_contour[0, 0] > extended_contour[-1, 0]:
        flipped = True

    valid_index_list = []
    recorded_x = extended_contour[0, 0]
    for i in range(len(extended_contour) - 1):
        curr_x = extended_contour[i, 0]
        next_x = extended_contour[i + 1, 0]

        if flipped:
            delta_recorded = curr_x - recorded_x
            if delta_recorded <= 0:
                recorded_x = curr_x
            delta_main = next_x - recorded_x
            if delta_main <= 0:
                valid_index_list.append(i)
        else:
            delta_recorded = curr_x - recorded_x
            if delta_recorded >= 0:
                recorded_x = curr_x
            else:
                print()
            delta_main = next_x - recorded_x
            if delta_main >= 0:
                valid_index_list.append(i)
    valid_index_list.append(i + 1)

    # contour_x_set = set()
    # valid_index_list = []
    # for i in range(extended_contour):
    #     curr_x = extended_contour[i, 0]
    #     duplicated = curr_x in contour_x_set
    #
    #     if not duplicated:
    #         contour_x_set.add(curr_x)
    #         valid_index_list.append(i)

    valid_extended_contour = extended_contour[valid_index_list, :]

    return valid_extended_contour


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


def truncate_top_contour(top_contour, top_hull):
    """
    Truncates the top contour elements that are in the final section of top_hull. Note: both top hull and top contour
    needs to be in rearranged form
    :param top_contour:
    :param top_hull:
    :return: a 2D truncated top_contour array
    """
    flipped = False
    top_hull_x_beg = top_hull[0, 0]
    top_hull_x_end = top_hull[-1, 0]
    if top_hull_x_beg > top_hull_x_end:
        flipped = True

    top_hull_end_point = top_hull[-2, :]
    top_hull_beg_point = top_hull[0, :]
    if flipped:
        inclusion_index_end = get_first_encounter(top_contour, top_hull_end_point[0])
        inclusion_index_end = np.arange(0, inclusion_index_end)
        inclusion_index_beg = get_first_encounter(top_contour, top_hull_beg_point[0])
        inclusion_index_beg = np.arange(inclusion_index_beg, len(top_contour))
        # inclusion_index_end = np.where(top_contour[:, 0] >= top_hull_end_point[0])[0]
        # inclusion_index_beg = np.where(top_contour[:, 0] <= top_hull_beg_point[0])[0]
    else:
        inclusion_index_end = get_first_encounter(top_contour, top_hull_end_point[0])
        inclusion_index_end = np.arange(0, inclusion_index_end)
        inclusion_index_beg = get_first_encounter(top_contour, top_hull_beg_point[0])
        inclusion_index_beg = np.arange(inclusion_index_beg, len(top_contour))
        # inclusion_index_end = np.where(top_contour[:, 0] <= top_hull_end_point[0])[0]
        # inclusion_index_beg = np.where(top_contour[:, 0] >= top_hull_beg_point[0])[0]

    inclusion_index = list(set(inclusion_index_beg) & set(inclusion_index_end))
    inclusion_contour = top_contour[inclusion_index, :]
    return inclusion_contour


def get_first_encounter(contour_list, condition):
    l = len(contour_list)

    for i in range(l):
        if contour_list[i, 0] == condition:
            return i

    raise Exception('Invalid Condition')


def black_out_bottom_part(bottom_top_contour, frame):
    """
    Since the direction of muscle is not a straight line. Just cutting of the frame horizontally might crop some of the
    top parts. This method blacks out the bottom part under the contour
    :param bottom_top_contour:
    :param frame: Must be a grayscale 2D matrix
    :return: a frame with the bottom part blacked out
    """
    frame_height, frame_width = frame.shape
    bottom_top_contour_beg = bottom_top_contour[0]
    bottom_top_contour_end = bottom_top_contour[-1]

    if bottom_top_contour_beg[0] > bottom_top_contour_end[0]:
        bottom_top_contour = np.flipud(bottom_top_contour)

    lr_point = [frame_width - 1, frame_height - 1]
    ll_point = [0, frame_height - 1]
    bottom_top_contour = np.append(bottom_top_contour, [lr_point], axis=0)
    bottom_top_contour = np.append(bottom_top_contour, [ll_point], axis=0)
    # bottom_top_contour.append(lr_point)
    # bottom_top_contour.append(ll_point)

    cv2.drawContours(frame, [bottom_top_contour], -1, color=(0, 0, 0), thickness=cv2.FILLED)
    return frame


def fill_untruncated_top_contour(top_hull, top_contour, frame_width):
    hull_last = top_hull[-1, :]
    hull_last_2 = top_hull[-2, :]
    x = [hull_last[0], hull_last_2[0]]
    y = [hull_last[1], hull_last_2[1]]
    if len(top_hull) >= 3:
        hull_last_3 = top_hull[-3, :]
        x.append(hull_last_3[0])
        y.append(hull_last_3[1])

    linear_fit = np.polyfit(x, y, 1)
    linear_fit = np.poly1d(linear_fit)

    flipped = False
    top_hull_x_beg = top_hull[0, 0]
    top_hull_x_end = top_hull[-1, 0]
    if top_hull_x_beg > top_hull_x_end:
        flipped = True

    x_end_stop = top_contour[-1, 0]
    x_beg_stop = top_contour[0, 0]

    if flipped:
        x_fill_range = np.arange(0, x_end_stop)
        x_beg_fill_range = np.arange(x_beg_stop + 1, frame_width)
    else:
        x_fill_range = np.arange(x_end_stop + 1, frame_width)
        x_beg_fill_range = np.arange(0, x_beg_stop)

    y_fill_range = linear_fit(x_fill_range)
    y_beg_fill_range = linear_fit(x_beg_fill_range)
    end_filled_range = np.transpose(np.vstack((x_fill_range, y_fill_range)))
    beg_fill_range = np.transpose(np.vstack((x_beg_fill_range, y_beg_fill_range)))
    if flipped:
        end_filled_range = np.flipud(end_filled_range)
        beg_fill_range = np.flipud(beg_fill_range)
    filled_top_contour = np.concatenate((beg_fill_range, top_contour, end_filled_range))
    return filled_top_contour.astype(np.int32)


def fill_truncated_top_contour(top_hull, truncated_top_contour, frame_width):
    """
    Extends both left and right of the truncated top contour. Truncation means that the ending of the top contour is
    undesirable
    :param top_hull:
    :param truncated_top_contour:
    :param frame_width:
    :return:
    """
    flipped = False
    top_hull_x_beg = top_hull[0, 0]
    top_hull_x_end = top_hull[-1, 0]
    if top_hull_x_beg > top_hull_x_end:
        flipped = True

    top_hull_end_1 = top_hull[-2, :]
    top_hull_end_2 = top_hull[-3, :]
    top_hull_end_x = [top_hull_end_1[0], top_hull_end_2[0]]
    top_hull_end_y = [top_hull_end_1[1], top_hull_end_2[1]]
    linear_fit = np.polyfit(top_hull_end_x, top_hull_end_y, 1)
    linear_fit = np.poly1d(linear_fit)
    x_end_stop = truncated_top_contour[-1, 0]
    x_beg_stop = truncated_top_contour[0, 0]

    if flipped:
        x_fill_range = np.arange(0, x_end_stop)
        x_beg_fill_range = np.arange(x_beg_stop + 1, frame_width)
    else:
        x_fill_range = np.arange(x_end_stop + 1, frame_width)
        x_beg_fill_range = np.arange(0, x_beg_stop)

    y_fill_range = linear_fit(x_fill_range)
    y_beg_fill_range = linear_fit(x_beg_fill_range)
    end_filled_range = np.transpose(np.vstack((x_fill_range, y_fill_range)))
    beg_fill_range = np.transpose(np.vstack((x_beg_fill_range, y_beg_fill_range)))
    if flipped:
        end_filled_range = np.flipud(end_filled_range)
        beg_fill_range = np.flipud(beg_fill_range)
    filled_top_contour = np.concatenate((beg_fill_range, truncated_top_contour, end_filled_range))
    return filled_top_contour.astype(np.int32)


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
    Determine if a end truncation is necessary. Top Hull needs to be a rearranged top_hull, I.E. y value from small
    to large :param top_hull: :return: A boolean. True if the slope difference is greater than 0.2, False otherwise.
    A True return signals that the final part of the slope need to be truncated
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

    # cv2.imshow('post processing', bw)


def self_multiply(gray_frame):
    gray_frame = gray_frame.astype(np.int32)
    gray_frame = np.multiply(gray_frame, gray_frame)
    gray_frame = gray_frame / np.amax(gray_frame)
    gray_frame = gray_frame * 255
    gray_frame = gray_frame.astype(np.uint8)
    # cv2.imshow('Self Multiply', gray_frame)
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
    """
    Creates sorted binary connected component by area
    :param binary_image:
    :return:
    """
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
