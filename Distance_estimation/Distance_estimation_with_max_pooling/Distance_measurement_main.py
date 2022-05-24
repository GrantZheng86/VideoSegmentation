import skimage.measure
import cv2
import glob
import os
import numpy as np
import Distance_estimation.Distance_measurement
import matplotlib.pyplot as plt

ROOT_DIR = r'C:\Users\Zheng\Documents\Medical_Images\Processed Images\P1P3P8_files\renamed_images'
BANNER_HEIGHT = 140
RULER_WIDTH = 40
BOTTOM_HEIGHT = 200
POOLING_SIZE = 10
SLOPE_WINDOW = 7


def crop_image_for_detection(img_gray):
    to_return = img_gray[BANNER_HEIGHT:-BOTTOM_HEIGHT, :-RULER_WIDTH]
    return to_return


def scale_contour(contour):
    contour = contour * POOLING_SIZE
    # contour = Distance_estimation.Distance_measurement.fill_contour(contour)
    return contour


def find_average_slope(contour,  imshow=False):
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

    if imshow:
        plt.plot(avg_slope_list)
        plt.show()

    return avg_slope_list


def visualize_contour(img, contour, point_of_interest_index=None):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_with_line = cv2.polylines(img, [contour], False, (0, 255, 0), 3)

    for each_pt in contour:
        img_with_line = cv2.circle(img_with_line, (each_pt[0], each_pt[1]), 3, (0, 0, 255), 2)
    if point_of_interest_index is not None:
        point_of_interest = contour[point_of_interest_index]
        img_with_line = cv2.circle(img_with_line, (point_of_interest[0], point_of_interest[1]), 5, (0, 255, 255), 5)


    cv2.imshow('Bottom Contour', img_with_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_point_of_interest(slope_list):
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
        if np.abs(largest_slope[1] - second_largest_slope[1]) > 5:
            found_bounds = True
        else:
            loop_count += 1

    point_of_interest = int((largest_slope[1] + second_largest_slope[1]) / 2)
    return point_of_interest


if __name__ == '__main__':

    for raw_img in glob.glob(os.path.join(ROOT_DIR, '*.jpg')):
        shorter_img_name = raw_img.split('\\')[-1]
        if 'E' in shorter_img_name:
            img_bgr = cv2.imread(raw_img)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = crop_image_for_detection(img_gray)
            img_gray_pooled = skimage.measure.block_reduce(img_gray, (POOLING_SIZE, POOLING_SIZE), np.min)
            img_gray_pooled = np.array(img_gray_pooled, dtype=np.uint8)
            bottom_contour, h = Distance_estimation.Distance_measurement.get_bottom_contour(
                cv2.cvtColor(img_gray_pooled, cv2.COLOR_GRAY2BGR), reduction=False, bottom_percentile=60)
            bottom_contour_filled = scale_contour(bottom_contour)
            bottom_contour_filled = np.array(bottom_contour_filled, dtype=np.int32)
            avg_slope_list = find_average_slope(bottom_contour_filled)

            try:
                point_of_interest = find_point_of_interest(avg_slope_list)
                point_of_interest += int(SLOPE_WINDOW/2)
                visualize_contour(img_gray, bottom_contour_filled, point_of_interest)
            except:
                print('{}detection failed'.format(shorter_img_name))
                visualize_contour(img_gray, bottom_contour_filled)
                plt.plot(avg_slope_list)
                plt.show()

