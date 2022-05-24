import skimage.measure
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import Distance_estimation.Distance_measurement


ROOT_DIR = r'C:\Users\Zheng\Documents\Medical_Images\Processed Images\P1DUS_files\renamed_images'
BANNER_HEIGHT = 140
RULER_WIDTH = 40
BOTTOM_HEIGHT = 200
POOLING_SIZE = 8

def crop_image_for_detection(img_gray):
    to_return = img_gray[BANNER_HEIGHT:-BOTTOM_HEIGHT, :-RULER_WIDTH]
    return to_return

def scale_contour(contour):
    contour = contour*8
    contour = Distance_estimation.Distance_measurement.fill_contour(contour)
    return contour

if __name__ == '__main__':

    for raw_img in glob.glob(os.path.join(ROOT_DIR, '*.jpg')):
        img_bgr = cv2.imread(raw_img)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = crop_image_for_detection(img_gray)
        img_gray_pooled = skimage.measure.block_reduce(img_gray, (POOLING_SIZE, POOLING_SIZE), np.min)
        bottom_contour, h = Distance_estimation.Distance_measurement.get_bottom_contour(cv2.cvtColor(img_gray_pooled, cv2.COLOR_GRAY2BGR), reduction=False)
        bottom_contour_filled = scale_contour(bottom_contour)
        bottom_contour_filled = np.array(bottom_contour_filled, dtype=np.int32)
        img_bgr = cv2.drawContours(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), [bottom_contour_filled], 0, (0, 255, 0), 3)
        cv2.imshow('a', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.imshow(img_gray, cmap='gray')
        # plt.show()
