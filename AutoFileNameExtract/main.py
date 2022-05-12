import cv2
import numpy as np
import glob
import os
import pytesseract

IMAGE_FOLDER = 'RawImages'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def color_filter(image, debug=False):
    LOWER_BOUND = (38, 80, 10)
    UPPER_BOUND = (40, 255, 255)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, LOWER_BOUND, UPPER_BOUND)
    filtered = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    if debug:
        cv2.namedWindow("Color filtered HSV", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color filtered HSV", 600, 700)
        cv2.imshow('Color filtered HSV', cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


if __name__ == '__main__':
    for image_name in glob.glob('{}/*.png'.format(IMAGE_FOLDER)):
        image_bgr = cv2.imread(image_name)
        filtered_image = color_filter(image_bgr, debug=True)
        # config = r'--textord_space_size_is_variable 1'
        print(pytesseract.image_to_string(filtered_image))
