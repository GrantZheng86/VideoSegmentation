import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import Case2_binary_component
import xlsxwriter

BOTTOM_STARTING_HEIGHT = 275
DEBUG_FILE_NAME = "debug.xlsx"
THRESHOLDING_CUTOFF = 100


def get_binary_cc(binary_image):
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    binary_component_list = []
    for i in range(num_cc - 1):
        curr_binary_component = Case2_binary_component.Case2BinaryComponent(labels, label_num=i + 1)

        if not curr_binary_component.invalid_component:
            binary_component_list.append(curr_binary_component)
    binary_component_list = sort_component_by_area(binary_component_list)

    return binary_component_list
    # largest_component = binary_component_list[-1]
    # second_largest = binary_component_list[-2]


def sort_component_by_area(component_list):
    l = len(component_list)

    for i in range(l):

        min_idx = i

        for j in range(i + 1, l):
            if component_list[min_idx].area > component_list[j].area:
                min_idx = j

        component_list[i], component_list[min_idx] = component_list[min_idx], component_list[i]

    return component_list


def get_bottom_two_parts(img, frame_num=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bottom_height = bottom_two_parts_recursion_helper(gray, BOTTOM_STARTING_HEIGHT, frame_num)
    valid_frame = False
    # 413 is the cutoff threshold for non-detection
    if bottom_height != -1:
        h, w = gray.shape
        line_y_position = h - bottom_height
        start_pt = (0, line_y_position)
        end_pt = (w, line_y_position)
        img = cv2.line(img, start_pt, end_pt, (0, 255, 0), 3)
        valid_frame = True
    else:
        img = cv2.putText(img, "No Segmentation", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return img, bottom_height, valid_frame


def bottom_two_parts_recursion_helper(full_gray_img, cutoff_height, frame_number=0):
    # 1. Crop the bottom image out. If the cutoff line is more than half way up to the image. Then return -1 indicate
    # this is not a good image
    kernel = np.ones((7, 7), np.uint8)
    h, w = full_gray_img.shape
    if cutoff_height >= h / 2:
        bottom_gray = full_gray_img[h - cutoff_height:, :]

        _, bw = cv2.threshold(bottom_gray, THRESHOLDING_CUTOFF, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        # if frame_number != 0:
        #     cv2.imwrite("Debugging Frame {}.jpg".format(frame_number), cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))
        cv2.imshow('Unsegmented', bw)
        return -1

    bottom_gray = full_gray_img[h - cutoff_height:, :]

    # 2. Thresholding the image
    _, bw = cv2.threshold(bottom_gray, THRESHOLDING_CUTOFF, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # 3. get binary connected component list
    binary_cc_list = get_binary_cc(bw)
    if len(binary_cc_list) < 2:
        return bottom_two_parts_recursion_helper(full_gray_img, cutoff_height + 1, frame_number)

    # 4. determine stop condition
    stop = recursion_stop(binary_cc_list)

    if stop:
        # if frame_number != 0:
        #     cv2.imwrite("Debugging Frame {}.jpg".format(frame_number), cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))
        return cutoff_height
    else:
        cutoff_height += 1
        return bottom_two_parts_recursion_helper(full_gray_img, cutoff_height, frame_number)


def recursion_stop(binary_cc_list):
    largest = binary_cc_list[-1]
    second_largest = binary_cc_list[-2]

    area_list = []
    for each_cc in binary_cc_list:
        area_list.append(each_cc.area)

    mean_area = np.average(area_list)
    stdv_area = np.std(area_list)
    large_area_cutoff = 1750

    large_area_criteria = second_largest.area > large_area_cutoff

    largest_x = largest.centroid[0]
    second_x = second_largest.centroid[0]

    largest_y = largest.centroid[1]
    second_y = second_largest.centroid[1]

    # position_constraint_x = second_x > largest_x
    # position_constraint_y = second_y > largest_y
    position_constraint = np.abs(largest_y - second_y) < 80

    if largest.area_cutoff == False and second_largest.area_cutoff == False:
        top_cutoff_criteria = False
    else:
        top_cutoff_criteria = True

    if large_area_criteria and not top_cutoff_criteria and position_constraint:
        return True
    else:
        return False


def pixel_by_percentile(img, percentile):
    img = np.array(img)
    img.flatten()
    non_zero_array = img[img != 0]
    thresh_value = np.percentile(non_zero_array, percentile)
    return thresh_value


def partition_bottom_frame(bt_frame):
    gray_bt_frame = cv2.cvtColor(bt_frame, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray_bt_frame, THRESHOLDING_CUTOFF, 255, cv2.THRESH_BINARY)
    binary_cc_list = get_binary_cc(bw)
    sorted_binary_cc_list = sort_component_by_area(binary_cc_list)

    largest = sorted_binary_cc_list[-1]
    second = sorted_binary_cc_list[-2]

    largest_x = largest.centroid[0]
    second_x = second.centroid[0]

    if largest_x < second_x:
        femur = second
        pelvis = largest
    else:
        femur = largest
        pelvis = second

    femur_top_contour = femur.get_contour_top()
    pelvis_top_contour = pelvis.get_contour_top()

    cv2.polylines(bt_frame, [femur_top_contour], False, (0, 255, 0), 2)
    cv2.imshow('Top Contour', bt_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_convolution(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kernel = np.array([[0, 1, 0, -1, 0],
    #                    [1, 2, 0, -2, -1],
    #                    [2, 3, 0, -3, -2],
    #                    [1, 2, 0, -2, -1],
    #                    [0, 1, 0, -1, 0]])

    # kernel = np.array([[1, 2, 1],
    #                    [0, 0, 0],
    #                    [-1, 2, -1]])

    kernel = np.array([[0, 1, 2, 1, 1],
                       [1, 2, 3, 2, 1],
                       [0, 0, 0, 0, 0],
                       [-1, -2, -3, -2, -1],
                       [0, -1, 2, -1, 0]])

    # kernel = np.array([[1, 0, -1],
    #                    [2, 0, -2],
    #                    [1, 0, -1]])

    convolved = ndimage.convolve(img, kernel, mode='constant', cval=0.0)
    cv2.imshow('conv', convolved)
    print()


def image_gradient(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    lap = np.uint8(np.absolute(lap))
    # Below code convert image gradient in x direction
    sobelx = cv2.Scharr(image, 0, dx=1, dy=0)
    sobelx = np.uint8(np.absolute(sobelx))
    # Below code convert image gradient in y direction
    sobely = cv2.Scharr(image, 0, dx=0, dy=1)
    sobely = np.uint8(np.absolute(sobely))

    # kernel = np.ones((7, 7), np.uint8)
    kernel = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    sobely = cv2.morphologyEx(sobely, cv2.MORPH_CLOSE, kernel)
    _, sobely = cv2.threshold(sobely, 200, 255, cv2.THRESH_BINARY)

    added = sobelx + sobely
    # plt.imshow(added, cmap='gray')
    # plt.show()

    to_show = np.hstack((lap, sobely, sobelx))
    cv2.imshow('Gradient', to_show)
