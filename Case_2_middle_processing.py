import cv2
import numpy as np
import Case2_binary_component
from Case_2_Processing import sort_component_by_area, pixel_by_percentile


def partition_middle_part(middle_frame):
    middle_frame = cv2.blur(middle_frame, (5, 5))
    gray_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2GRAY)
    pixel_cutoff = pixel_by_percentile(gray_frame, 75)
    _, binary = cv2.threshold(gray_frame, pixel_cutoff, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    bw = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
    binary_cc_list = []
    for i in range(num_cc):
        each_binary_cc = Case2_binary_component.Case2MiddleBinaryComponent(labels, i)
        binary_cc_list.append(each_binary_cc)

    sorted_binary_cc = sort_component_by_area(binary_cc_list)
    largest_cc = sorted_binary_cc[-2]
    top_contour, bottom_contour = largest_cc.linear_approximation()

    # cv2.polylines(middle_frame, [top_contour], False, (0, 255, 255), 1)
    # cv2.polylines(middle_frame, [bottom_contour], False, (0, 255, 255), 1)
    # cv2.imshow("Middle_frame", middle_frame)

    return top_contour, bottom_contour
