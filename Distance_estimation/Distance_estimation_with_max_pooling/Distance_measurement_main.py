import skimage.measure
import cv2
import glob
import os
import numpy as np
import Distance_estimation.Distance_measurement
import matplotlib.pyplot as plt

ROOT_DIR = r'C:\Users\Zheng\Documents\Medical_Images\Processed Images\P12P13_files\renamed_images'
BANNER_HEIGHT = 140
RULER_WIDTH = 40
BOTTOM_HEIGHT = 200
POOLING_SIZE = 12
SLOPE_WINDOW = 5
UPPER_RATIO = 1 / 3
MINIMUM_WIDTH = 50


def crop_image_for_detection(img_gray):
    to_return = img_gray[BANNER_HEIGHT:-BOTTOM_HEIGHT, :-RULER_WIDTH]
    return to_return


def scale_contour(contour):
    """
    Scale the pooled contour back to the image size
    :param contour: The contour from the pooled image
    :return: a rescaled contour back into the original image scale
    """
    # contour = Distance_estimation.Distance_measurement.fill_contour(contour)
    contour = contour * POOLING_SIZE
    # contour = Distance_estimation.Distance_measurement.fill_contour(contour)
    return contour


def find_average_slope(contour, imshow=False):
    """
    Calculated a list of averaged contour. The average window is defined as a constant in the head of the file.
    :param contour: The contour in the original image scale
    :param imshow: Whether to show a plot of the calculated slope
    :return: a list of averaged slope
    """

    if (len(contour) < SLOPE_WINDOW):
        raise 'Insufficient contour length'
    l, _ = contour.shape
    avg_slope_list = []

    for i in range(l - SLOPE_WINDOW - 1):
        curr_pt = contour[i]
        slope_list = []
        for j in range(SLOPE_WINDOW):
            next_pt = contour[i + j + 1]
            x_change = float(next_pt[0] - curr_pt[0])
            y_change = float(next_pt[1] - curr_pt[1])
            # Put a small delta to avoid divide by 0 issue
            if x_change == 0:
                x_change = 1
            curr_slope = y_change / x_change
            slope_list.append(curr_slope)
        avg_slope_list.append(np.average(slope_list))

    l = len(avg_slope_list)
    for i in range(SLOPE_WINDOW):
        curr_pt = contour[i + l]
        slope_list = []
        for j in range(SLOPE_WINDOW - i):
            try:
                next_pt = contour[i+l+j]
            except:
                print()
            x_change = float(next_pt[0] - curr_pt[0])
            y_change = float(next_pt[1] - curr_pt[1])
            # Put a small delta to avoid divide by 0 issue
            if x_change == 0:
                x_change = 1
            curr_slope = y_change / x_change
            slope_list.append(curr_slope)
        avg_slope_list.append(np.average(slope_list))

    avg_slope_list.append(avg_slope_list[-1])


    if imshow:
        plt.plot(avg_slope_list)
        plt.show()

    return avg_slope_list


def visualize_contour(img, bottom_contour, top_contour, point_of_interest_index=None):
    """
    Showing the contour of the sping. with critical points for slope calculation, and an optional point of interest
    :param img: The image in the original scale, can be either BW and GRAY
    :param contour: The contour in the original scale
    :param point_of_interest_index: Optional argument, the index of point in the contour
    :return: None, just show image
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_with_line = cv2.polylines(img, [bottom_contour], False, (0, 255, 0), 3)
    # img_with_line = cv2.polylines(img_with_line, [top_contour], False, (255, 255, 0), 3)

    for each_pt in bottom_contour:
        img_with_line = cv2.circle(img_with_line, (each_pt[0], each_pt[1]), 3, (0, 0, 255), 2)
    if point_of_interest_index is not None:
        point_of_interest = bottom_contour[point_of_interest_index]
        img_with_line = cv2.circle(img_with_line, (point_of_interest[0], point_of_interest[1]), 5, (0, 255, 255), 5)

    cv2.imshow('Bottom Contour', img_with_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_point_of_interest(slope_list):
    # TODO: point of interest determination needs to be changed. Currently just taking the middle point of two max
    #  values. This will lead to miss classification. 1. Better methods of finding peaks, instead of max values. 2. Considering pixel intensities
    """
    Finds the index of point of interest from the slope list. This function can also throw exception when the contour
    slope profile does not match the shape we want.
    :param slope_list: A list of slope
    :return: The index at which the point that represents the spine contour
    """
    indices = np.arange(0, len(slope_list), 1, dtype=int)
    slope_with_indices = np.vstack((slope_list, indices))
    slope_with_indices = slope_with_indices.T
    slope_with_indices = slope_with_indices[slope_with_indices[:, 0].argsort()]

    found_bounds = False
    largest_slope = slope_with_indices[-1]
    second_largest_slope = slope_with_indices[-2]
    loop_count = 0
    while not found_bounds:
        second_largest_slope = slope_with_indices[-2 - loop_count]
        if loop_count > 4:
            raise Exception('Looping out of bounds, cannot find the template')
        if np.abs(largest_slope[1] - second_largest_slope[1]) > 4:
            found_bounds = True
        else:
            loop_count += 1

    point_of_interest = int((largest_slope[1] + second_largest_slope[1]) / 2)
    return point_of_interest


def find_point_of_interest_1(contour, img_gray, imshow=False):
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    _, x_max = img_gray.shape

    contour_slope = find_average_slope(contour, imshow=False)
    indices = np.arange(0, len(contour_slope), 1, dtype=int)
    intensity_value = []

    for pt in contour:
        avg = pixel_average_around_point(img_gray, pt, imshow=imshow)
        intensity_value.append(avg)

    intensity_value = np.array(intensity_value)
    contour_indices_intensity = np.vstack((indices, contour_slope, intensity_value))
    contour_indices_intensity = contour_indices_intensity.T

    sorted_by_intensity = contour_indices_intensity[contour_indices_intensity[:, -1].argsort()]

    point_to_return = 0
    meet_spec = False
    while not meet_spec:
        point_to_return -= 1
        curr_slope = sorted_by_intensity[point_to_return][1]
        curr_index = int(sorted_by_intensity[point_to_return][0])
        prev_index = int(sorted_by_intensity[point_to_return][0] - SLOPE_WINDOW)

        prev_slope = None
        if prev_index > 0:
            prev_slope = contour_indices_intensity[prev_index][1]
        if prev_slope is not None and prev_slope <= 0 <= curr_slope:
            point_location = contour[curr_index]
            x = point_location[0]
            if x - MINIMUM_WIDTH > 0 and x + MINIMUM_WIDTH < x_max:
                meet_spec = True



    return int(sorted_by_intensity[point_to_return][0])


def pixel_average_around_point(img, point, imshow=False):
    y_lim, x_lim = img.shape
    BOX_LEN = 50
    BOX_Y_OFFSET = -BOX_LEN / 2
    x = point[0]
    y = point[1]
    x_min = int(np.max((x - BOX_LEN / 2, 0)))
    y_min = int(np.max((y - BOX_LEN / 2, 0)) + BOX_Y_OFFSET)
    x_max = int(np.min((x_min + BOX_LEN, x_lim)))
    y_max = int(np.min((y_min + BOX_LEN, y_lim)))

    if imshow:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_color = cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        img_color = cv2.circle(img_color, (x, y), 3, (0, 0, 255), 2)
        cv2.imshow('box boundary', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    area = img[y_min:y_max, x_min:x_max]
    return np.average(area)


def find_top_contour(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    upper_region = img[0:int(img.shape[0] * UPPER_RATIO), :]
    upper_region_pooled = skimage.measure.block_reduce(upper_region, (POOLING_SIZE, POOLING_SIZE), np.min)
    top_contour, successful_detection = Distance_estimation.Distance_measurement.find_lumbodorsal_bottom(
        upper_region_pooled, imshow=False, reduction=False)

    if successful_detection:
        return scale_contour(top_contour)

    raise Exception('Top contour detection failed')


if __name__ == '__main__':

    for raw_img in glob.glob(os.path.join(ROOT_DIR, '*.jpg')):
        shorter_img_name = raw_img.split('\\')[-1]
        if 'E' in shorter_img_name:
            img_bgr = cv2.imread(raw_img)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = crop_image_for_detection(img_gray)
            img_gray_pooled = skimage.measure.block_reduce(img_gray, (POOLING_SIZE, POOLING_SIZE), np.min)
            img_gray_pooled = np.array(img_gray_pooled, dtype=np.uint8)
            kernel = np.ones((1, 1), np.uint8)
            img_gray_pooled = cv2.morphologyEx(img_gray_pooled, cv2.MORPH_OPEN, kernel)
            bottom_contour, h = Distance_estimation.Distance_measurement.get_bottom_contour(
                cv2.cvtColor(img_gray_pooled, cv2.COLOR_GRAY2BGR), reduction=False, bottom_percentile=60)
            bottom_contour_filled = scale_contour(bottom_contour)
            bottom_contour_filled = np.array(bottom_contour_filled, dtype=np.int32)
            # avg_slope_list = find_average_slope(bottom_contour_filled, imshow=False)

            try:
                # point_of_interest = find_point_of_interest(avg_slope_list)
                point_of_interest = find_point_of_interest_1(bottom_contour_filled, img_gray)
                top_contour = find_top_contour(img_gray)
                visualize_contour(img_gray, bottom_contour_filled, top_contour, point_of_interest)

            except:
                print('{}detection failed'.format(shorter_img_name))
                visualize_contour(img_gray, bottom_contour_filled, None)


