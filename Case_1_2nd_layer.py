import cv2
import numpy as np
from Case_2_Processing import pixel_by_percentile, sort_component_by_area
from Case_1_binary_component import Class_1_binary_cc

LAYER_THICKNESS = 10
PERCENTILE = 80
STARTING_HEIGHT = 70
LARGE_AREA_THRESHOLD = 5000


def extract_contours(img, crop_location):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img[0:crop_location, :]

    im_h, im_w = img.shape

    pixel_threshold = pixel_by_percentile(img, PERCENTILE)
    _, bw = cv2.threshold(img, pixel_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    seg_height, top_info, bottom_info = recursion_helper(bw, height=STARTING_HEIGHT)

    top_contour = top_info[1]
    top_contour[:, 1] = top_contour[:, 1] + (im_h - seg_height)
    remove_bumps(top_contour)

    cv2.polylines(bgr_img, [top_info[1]], False, (0, 255, 0))
    cv2.imshow('bw', bgr_img)


def remove_bumps(contour):
    STARTING_INDEX = 10
    starting_point = contour[STARTING_INDEX, :]
    end_point = contour[-STARTING_INDEX, :]

    starting_x = starting_point[0]
    starting_y = starting_point[1]
    ending_x = end_point[0]
    ending_y = end_point[1]

    del_y = (ending_y - starting_y)
    del_x = (ending_x - starting_x)

    general_angle = np.arctan2(del_y, del_x)
    angle_array = create_angle_array(contour, STARTING_INDEX)
    import matplotlib.pyplot as plt
    plt.plot(angle_array)
    plt.show()
    print()


def create_angle_array(contour, starting_index):
    toReturn = []

    block_size = 10
    for i in range(len(contour) - 3 * starting_index - 1):
        base_point = contour[starting_index + i + 1]

        angle_list = []
        for j in range(block_size):
            curr_point = contour[starting_index + i + j + 2]
            del_x = curr_point[0] - base_point[0]
            del_y = curr_point[1] - base_point[1]
            angle = np.arctan2(del_y, del_x)
            angle = np.rad2deg(angle)
            angle_list.append(angle)
        toReturn.append(np.average(angle_list))

    return toReturn


def recursion_helper(bw_img, height):
    h, w = bw_img.shape
    if height > h * 3 / 4:
        return -1
    upper_stop = h - height
    bw_img_crop = bw_img[upper_stop:, :]
    sorted_binary_cc = get_binary_cc(bw_img_crop)

    # if there's too few elements in the array
    if len(sorted_binary_cc) < 2:
        return recursion_helper(bw_img, height=height + 1)

    largest_cc = sorted_binary_cc[-1]
    # If the largest cc has been cutoff on top
    if largest_cc.area_cutoff:
        return recursion_helper(bw_img, height=height + 1)
    if largest_cc.area < LARGE_AREA_THRESHOLD:
        return recursion_helper(bw_img, height=height + 1)
    else:
        top_info = [largest_cc.get_hull_top(), largest_cc.get_contour_top()]
        bottom_info = [largest_cc.get_hull_bottom(), largest_cc.get_contour_bottom()]
        return height, top_info, bottom_info


def find_image_boundary(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw_inverse = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_inverse, connectivity=4)

    binary_cc_list = []
    for i in range(num_cc - 1):
        curr_index = i + 1
        curr_binary_cc = Class_1_binary_cc(labels, curr_index)
        binary_cc_list.append(curr_binary_cc)

    upper_left_cc = find_upper_left_component(binary_cc_list)
    upper_right_cc = find_upper_right_component(binary_cc_list)

    right_hypotenuse = get_hypotenuse(upper_right_cc)
    left_hypotenuse = get_hypotenuse(upper_left_cc)

    return left_hypotenuse, right_hypotenuse


def find_upper_left_component(cc_list):
    AREA_CRITERIA = 5000

    left_most_centroid_x = 999
    upper_most_centroid_y = 999
    upper_left_corner = -1

    for i in range(len(cc_list)):
        curr_cc = cc_list[i]

        if not curr_cc.invalid_component:
            curr_centroid = curr_cc.centroid
            curr_x = curr_centroid[0]
            curr_y = curr_centroid[1]
            curr_area = curr_cc.area

            if curr_x < left_most_centroid_x and curr_y < upper_most_centroid_y and curr_area > AREA_CRITERIA:
                upper_left_corner = i
                left_most_centroid_x = curr_x
                upper_most_centroid_y = curr_y

    return cc_list[upper_left_corner]


def find_upper_right_component(cc_list):
    AREA_CRITERIA = 1000

    right_most_centroid_x = 0
    upper_most_centroid_y = 999
    upper_right_corner = -1

    for i in range(len(cc_list)):
        curr_cc = cc_list[i]

        if not curr_cc.invalid_component:
            curr_centroid = curr_cc.centroid
            curr_x = curr_centroid[0]
            curr_y = curr_centroid[1]
            curr_area = curr_cc.area

            if curr_x > right_most_centroid_x and curr_y < upper_most_centroid_y and curr_area > AREA_CRITERIA:
                upper_right_corner = i
                right_most_centroid_x = curr_x
                upper_most_centroid_y = curr_y

    return cc_list[upper_right_corner]


def get_binary_cc(binary_image):
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    binary_component_list = []
    for i in range(num_cc - 1):
        curr_binary_component = Class_1_binary_cc(labels, label_number=i + 1)

        if not curr_binary_component.invalid_component:
            binary_component_list.append(curr_binary_component)
    binary_component_list = sort_component_by_area(binary_component_list)

    return binary_component_list


def get_hypotenuse(corner_component):
    hull = np.squeeze(cv2.convexHull(corner_component.contour))

    lower_corner = np.max(hull[:, 1])
    lower_index = np.where(hull[:, 1] == lower_corner)[0]

    lower_x = hull[lower_index, 0]

    position = "R"
    if np.average(lower_x) <= 10:
        position = "L"

    if len(lower_x) > 1:
        lower_y = hull[lower_index, 1]

        if position == "R":
            lower_y = np.max(lower_y)
        else:
            lower_y = np.min(lower_y)
        lower_x = lower_x[0]
    else:
        lower_y = hull[lower_index, 1][0]
        lower_x = lower_x[0]

    lower_corner = [lower_x, lower_y]

    if position == "R":
        upper_corner_x = np.min(hull[:, 0])
    else:
        upper_corner_x = np.max(hull[:, 0])

    upper_corner_index = np.where(hull[:, 0] == upper_corner_x)[0]
    if len(upper_corner_index) > 1:
        upper_corner_y = hull[upper_corner_index, 1]
        upper_corner_y = np.min(upper_corner_y)
        upper_corner = [upper_corner_x, upper_corner_y]
    else:
        upper_corner_y = hull[upper_corner_index, 1]
        upper_corner = [upper_corner_x, upper_corner_y]

    x = [upper_corner[0], lower_corner[0]]
    y = [upper_corner[1], lower_corner[1]]

    linear_fit = np.polyfit(x, y, 1)
    linear_fit = np.poly1d(linear_fit)

    if position == "L":
        x_range = np.arange(lower_corner[0], upper_corner[0])
    else:
        x_range = np.arange(upper_corner[0], lower_corner[0])

    y_range = linear_fit(x_range)

    hypotenuse = np.transpose(np.vstack((x_range, y_range)))
    return hypotenuse.astype(np.int32)
