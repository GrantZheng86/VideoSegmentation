import numpy as np
import cv2


class Case2BinaryComponent:

    def __init__(self, connected_component_label_image, label_num):
        self.label_image = connected_component_label_image
        self.label_num = label_num
        self.area = -1
        self.area_cutoff = True
        self.contour = None
        self.separate_component()

        if self.area != 1:
            self.get_contour()
            self.determine_cutoff()
            # print(self.area_cutoff)
            self.invalid_component = False
            # self.visualize_with_contour()
        else:
            self.invalid_component = True

    def separate_component(self):
        """
        separate the desired component from the total label image.
        :return:
        """
        idx = np.where(self.label_image == self.label_num)
        self.area = len(idx[0])
        # self.visualize(idx)
        # print()

    def determine_cutoff(self):
        """
        Determine if the current component is cut by the crop from the original image.
        :return:
        """
        x, y, w, h = cv2.boundingRect(self.contour)
        if y == 0:
            self.area_cutoff = True
        else:
            self.area_cutoff = False

    def visualize(self, idx):
        background = np.zeros(self.label_image.shape)
        background[idx] = 255
        background = background.astype(np.uint8)

        cv2.imshow('connected Component', background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_with_contour(self):
        background = np.zeros(self.label_image.shape)
        idx = np.where(self.label_image == self.label_num)
        background[idx] = 255
        background = background.astype(np.uint8)

        color_background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_background, [self.contour], -1, (0, 255, 0), 2)
        cv2.imshow('contour', color_background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_contour(self):
        """
        Gets the major contour surrounding the connected component. Ignores other contours that might be voids inside
        the major connected component
        :return:
        """
        idx = np.where(self.label_image == self.label_num)
        background = np.zeros(self.label_image.shape)
        background[idx] = 255
        background = background.astype(np.uint8)

        contours, _ = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        actual_contour_id = -1
        longest_contour_size = -1
        counter = 0

        for each_contour in contours:
            l = each_contour.shape[0]
            if l > longest_contour_size:
                actual_contour_id = counter
                longest_contour_size = l
            counter += 1
        self.contour = np.squeeze(contours[actual_contour_id])

        # color_background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(color_background, contours, -1, (0, 255, 0), 2)
        # cv2.imshow('contour', color_background)
        # print()
