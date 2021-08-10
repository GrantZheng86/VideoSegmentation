import numpy as np
import cv2


class Case2BinaryComponent:

    def __init__(self, connected_component_label_image, label_num):
        self.label_image = connected_component_label_image
        self.label_num = label_num
        self.area = -1
        self.area_cutoff = True
        self.contour = None
        self.centroid = None
        self.separate_component()

        if self.area != 1:
            self.get_contour()
            self.determine_cutoff()
            # print(self.area_cutoff)
            self.invalid_component = False
            self.fill_centroid()
            # self.visualize_with_contour()
        else:
            self.invalid_component = True

    def fill_centroid(self):
        background = np.zeros(self.label_image.shape)
        idx = np.where(self.label_image == self.label_num)
        background[idx] = 255
        background = background.astype(np.uint8)
        _, _, _, centroid = cv2.connectedComponentsWithStats(background)
        self.centroid = centroid[-1]

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

    def get_contour_top(self):
        contour_x = self.contour[:, 0]
        left_most_x = np.min(contour_x)
        right_most_x = np.max(contour_x)

        left_candidate_index = np.where(contour_x == left_most_x)[0]
        right_candidate_index = np.where(contour_x == right_most_x)[0]

        left_candidate = np.squeeze(self.contour[left_candidate_index, :])
        right_candidate = np.squeeze(self.contour[right_candidate_index, :])

        if len(left_candidate_index) > 1:
            left_candidate_y = left_candidate[:, 1]
            min_y = np.min(left_candidate_y)
            upper_index = np.where(self.contour[:, 1] == min_y)
            upper_left_index = np.intersect1d(upper_index, left_candidate_index)
        else:
            upper_left_index = left_candidate_index

        if len(right_candidate_index) > 1:
            right_candidate_y = right_candidate[:, 1]
            min_y = np.min(right_candidate_y)
            upper_index = np.where(self.contour[:, 1] == min_y)[0]
            upper_right_index = np.intersect1d(upper_index, right_candidate_index)
        else:
            upper_right_index = right_candidate_index

        front_half = self.contour[0:upper_left_index[0], :]
        front_half = np.flipud(front_half)
        back_half = self.contour[upper_right_index[0]:, :]
        back_half = np.flipud(back_half)
        toReturn = np.concatenate((front_half, back_half))

        return np.array(toReturn.tolist())




