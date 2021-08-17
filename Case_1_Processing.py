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

    if segment_height >= h / 4 * 3:
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
        return segment_height, largest_cc.get_hull_top(), aspect_ratio


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
