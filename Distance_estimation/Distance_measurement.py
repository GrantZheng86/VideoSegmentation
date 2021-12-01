import cv2
import numpy as np

BOTTOM_PERCENTILE = 60


def morph_closing(img_bw, kernel_height=3, kernel_width=None):
    """
    Performs a morphological closing. Removes potential breakages in contour
    :param img_bw: A thresholded BW image
    :param kernel_height: Kernel height for the morph operation
    :param kernel_width: Kernel width for the morph operation
    :return: A BW image after the morph closing operation
    """
    if kernel_width is None:
        kernel_width = kernel_height

    assert len(np.unique(img_bw)) == 2, "Input image is not BW"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
    closing = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    return closing


def pixel_by_percentile(img_gray, percentile):
    """
    Determine the pixel intensity by percentile, all black pixels are excluded
    :param img_gray: a BW image
    :param percentile: The percentile the user wish to achieve in the image
    :return: an int value for the pixel
    """
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    assert len(img_gray.shape) == 2, "Error in image channels"
    img_gray = np.array(img_gray)
    img_gray.flatten()
    non_zero_array = img_gray[img_gray != 0]
    thresh_value = np.percentile(non_zero_array, percentile)
    return thresh_value


def get_largest_cc(img_bw):
    """
    Separates out the largest connected component in the image, and turns every other pixel values to be 0
    :param img_bw: a BW image for separation
    :return: a BW image, in the original size, that only contains the largest connected component
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bw, connectivity=8)
    stats_copy = stats.copy()
    areas = stats_copy[:, -1]
    areas.sort()
    largest_area = areas[-2]  # -1 is for the black background region

    location = np.where(stats == largest_area)
    label = location[0][0]
    largest_only = labels == label
    largest_only = np.array(largest_only, dtype=np.uint8)
    largest_only *= 255

    return largest_only


def get_longest_contour(contours):
    """
    Finds the longest contour within a contour list. This methods should be used to separate out the outer most contour
    from the inner contour of a component
    :param contours: a list of contours
    :return: index for the longest contour in the list
    """
    max_contour_len = 0
    max_contour_index = 0
    counter = 0

    for each_contour in contours:
        if len(each_contour) > max_contour_len:
            max_contour_len = len(each_contour)
            max_contour_index = counter
        counter += 1

    return max_contour_index


def contour_reduction(largest_contour):
    """
    Obtain an approximation, or LPF, of the contour, and then separate out the bottom section of it. Half separation is
    determined by the min and max x coordinates.
    :param largest_contour:
    :return: The bottom half of the approximated contour
    """
    ep = 0.007 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, ep, True)
    bottom_half_contour = get_bottom_half(approx)
    # plot_contour_trend(bottom_half_contour)

    return bottom_half_contour


def get_bottom_half(contour):
    """
    Gets the bottom half of the the contour
    :param contour:
    :return:
    """
    contour = np.squeeze(contour)

    contour_x = contour.copy()[:, 0]
    contour_y = contour.copy()[:, 1]

    contour_x_min = np.min(contour_x)
    contour_x_max = np.max(contour_x)

    x_min_index = np.where(contour[:, 0] == contour_x_min)
    x_max_index = np.where(contour[:, 0] == contour_x_max)
    x_min_index = x_min_index[0]
    x_max_index = x_max_index[0]

    if len(x_min_index) != 1:
        beginning_candidate_y = contour_y[x_min_index]
        max_y = np.max(beginning_candidate_y)
        beginning_index = np.where(contour_y == max_y)
        x_set_b = set(x_min_index)
        y_set_b = set(beginning_index[0])
        common_index_b = x_set_b & y_set_b
        beginning_index = common_index_b.pop()
    else:
        beginning_index = x_min_index[0]

    if len(x_max_index) != 1:
        end_candidate_y = contour_y[x_max_index]
        max_y = np.max(end_candidate_y)
        ending_index = np.where(contour_y == max_y)
        x_set_e = set(x_max_index)
        y_set_e = set(ending_index[0])
        common_index_e = x_set_e & y_set_e
        ending_index = common_index_e.pop()
    else:
        ending_index = x_max_index[0]

    return contour[beginning_index:ending_index + 1, :]


def get_bottom_contour(img, reduction=True, bottom_feature_ratio=1.7, show=False):
    """
    This method is specifically designed for case 4 to get the bottom contour.
    :param img:  A BGR image without any annotation or markers
    :param reduction: Whether to use approximation for bottom contour separation, this is like a LPF
    :param bottom_feature_ratio: the amount of frame that the bottom spine occupies
    :return: The bottom half (approximated by default) contour of the spine, and the beginning height of the cropped
            image region
    """
    original_image = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = int((img.shape[0] / bottom_feature_ratio) * (bottom_feature_ratio - 1))
    img = img[h:, :]
    thresh_value = pixel_by_percentile(img, BOTTOM_PERCENTILE)
    _, th = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    th = morph_closing(th, kernel_width=5, kernel_height=7)
    largest_binary = get_largest_cc(th)
    contours, _ = cv2.findContours(largest_binary, 1, 2)
    largest_contour_index = get_longest_contour(contours)
    if reduction:
        largest_contour = contour_reduction(contours[largest_contour_index])
    else:
        largest_contour = get_bottom_half(contours[largest_contour_index])

    largest_contour[:, 1] += h
    if show:
        contour_img = cv2.drawContours(original_image, [largest_contour], -1, (0, 255, 0),2)
        cv2.imshow("Contour", contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return largest_contour, h
