import cv2
import numpy as np
import Case2_binary_component
from Case_2_Processing import sort_component_by_area

STARTING_HEIGHT = 200
THRESHOLD_VALUE = 100


def get_top_element_bottom_contour(frame):
    segment_height = segment_top(frame)
    if segment_height == -1:
        return None, None
    segmented_frame = frame[0:segment_height, :, :]
    gray_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_frame, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=4)
    binary_cc_list = []
    for i in range(num_cc):
        each_binary_cc = Case2_binary_component.Case2TopBinaryComponent(labels, i)
        binary_cc_list.append(each_binary_cc)

    sorted_binary_cc = sort_component_by_area(binary_cc_list)
    largest_cc = sorted_binary_cc[-2]
    bottom_contour = largest_cc.get_contour_bottom()
    return bottom_contour, segment_height
    # hull = np.squeeze(largest_cc.get_contour_bottom())
    # cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
    # cv2.imshow("Convex Hull", frame)


def segment_top(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = segmentation_recursion_helper(gray_frame, STARTING_HEIGHT)
    # segmented_frame = frame[0:height, :, :]
    return height
    # cv2.imshow('Top Segmentation', segmented_frame)


def segmentation_recursion_helper(gray_frame, height):

    if height >= (gray_frame.shape[0]/2 - 50):
        return -1

    segmented_frame = gray_frame[0:height, :]
    _, thresholded = cv2.threshold(segmented_frame, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=4)
    binary_cc_list = []
    for i in range(num_cc):
        each_binary_cc = Case2_binary_component.Case2TopBinaryComponent(labels, i)
        binary_cc_list.append(each_binary_cc)

    sorted_binary_cc = sort_component_by_area(binary_cc_list)
    largest_cc = sorted_binary_cc[-2]

    if largest_cc.area_cutoff:
        return segmentation_recursion_helper(gray_frame, height + 1)
    else:
        return height
