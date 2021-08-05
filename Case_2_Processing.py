import cv2
import numpy as np

BOTTOB_STARTING_HEIGHT = 100

def get_bottom_two_parts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bottom_two_parts_recursion_helper(gray):
    thresh_val = pixel_by_percentile(gray, 60)

def pixel_by_percentile(img, percentile):
    img = np.array(img)
    img.flatten()
    non_zero_array = img[img != 0]
    thresh_value = np.percentile(non_zero_array, percentile)
    return thresh_value