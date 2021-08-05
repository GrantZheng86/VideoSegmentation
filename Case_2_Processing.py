import cv2
import numpy as np

BOTTOM_STARTING_HEIGHT = 400


def get_bottom_two_parts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bottom_gary = gray[h - BOTTOM_STARTING_HEIGHT:, :]
    bottom_two_parts_recursion_helper(bottom_gary, gray)


def bottom_two_parts_recursion_helper(gray, full_gray_img):
    thresh_val = pixel_by_percentile(gray, 80)
    _, bw = cv2.threshold(gray, thresh_val,255,cv2.THRESH_BINARY)
    ad_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 15)
    edges = cv2.Canny(gray, 1.5, 50)
    bw = np.vstack((bw, edges))
    cv2.imshow('bw', bw)


def pixel_by_percentile(img, percentile):
    img = np.array(img)
    img.flatten()
    non_zero_array = img[img != 0]
    thresh_value = np.percentile(non_zero_array, percentile)
    return thresh_value
