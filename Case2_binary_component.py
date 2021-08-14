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

    def get_contour_top(self, pelvis=False):
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

        if upper_right_index[0] > upper_left_index[0]:
            smaller_index = upper_left_index[0]
            larger_index = upper_right_index[0]
        else:
            smaller_index = upper_right_index[0]
            larger_index = upper_left_index[0]

        contour_1 = self.contour[smaller_index:larger_index + 1, :]
        contour_2_1 = self.contour[larger_index:, :]
        contour_2_2 = self.contour[0:smaller_index + 1, :]
        contour_2 = np.concatenate((contour_2_1, contour_2_2))
        contour_1_y = contour_1[:, 1]
        contour_2_y = contour_2[:, 1]

        if np.average(contour_1_y) > np.average(contour_2_y):
            top_contour = contour_2
        else:
            top_contour = contour_1

        if top_contour[0, 0] > top_contour[-1, 0]:
            top_contour = np.flipud(top_contour)

        return top_contour

    @staticmethod
    def get_upper_right_corner(cornerPoints):

        toReturn = {}
        center_pos = (np.average(cornerPoints[:, 0]), np.average(cornerPoints[:, 1]))
        for eachPoint in cornerPoints:
            if eachPoint[0] < center_pos[0]:
                if eachPoint[1] < center_pos[1]:
                    toReturn[0] = eachPoint
                else:
                    toReturn[1] = eachPoint
            else:
                if eachPoint[1] > center_pos[1]:
                    toReturn[2] = eachPoint
                else:
                    toReturn[3] = eachPoint

        return toReturn[3]


class Case2TopBinaryComponent(Case2BinaryComponent):

    def __init__(self, connected_component_label_image, label_num):
        super().__init__(connected_component_label_image, label_num)

    def determine_cutoff(self):
        r, c = self.label_image.shape

        x, y, w, h = cv2.boundingRect(self.contour)
        if y + h == r:
            self.area_cutoff = True
        else:
            self.area_cutoff = False

    def get_contour_bottom(self):
        hull = np.squeeze(cv2.convexHull(self.contour))
        hull_x = hull[:, 0]
        hull_y = hull[:, 1]

        max_hull_x = np.max(hull_x)
        min_hull_x = np.min(hull_x)

        max_index = np.where(hull_x == max_hull_x)[0]
        min_index = np.where(hull_x == min_hull_x)[0]

        if len(max_index) > 1:
            candidate_y = hull_y[max_index]
            lower_y = np.max(candidate_y)
            lower_y_index = np.where(hull_y == lower_y)
            max_index = np.intersect1d(max_index, lower_y_index)[0]
        else:
            max_index = max_index[0]

        if len(min_index) > 1:
            candidate_y = hull_y[min_index]
            lower_y = np.max(candidate_y)
            lower_y_index = np.where(hull_y == lower_y)
            min_index = np.intersect1d(min_index, lower_y_index)[0]
        else:
            min_index = min_index[0]

        smaller_index = min_index
        larger_index = max_index

        if min_index > max_index:
            smaller_index = max_index
            larger_index = min_index

        contour_1 = hull[smaller_index:larger_index + 1]

        contour_2_1 = hull[larger_index:, :]
        contour_2_2 = hull[0:smaller_index + 1, :]
        contour_2 = np.concatenate((contour_2_1, contour_2_2))

        contour_1_y = np.average(contour_1[:, 1])
        contour_2_y = np.average(contour_2[:, 1])

        if contour_1_y > contour_2_y:
            return contour_1
        else:
            return contour_2


class Case2MiddleBinaryComponent(Case2TopBinaryComponent):
    def __init__(self, connected_component_label_image, label_num):
        super().__init__(connected_component_label_image, label_num)
        self.thickness = 20

    def linear_approximation(self):
        top_contour = self.get_contour_top()
        bottom_contour = self.get_contour_bottom()

        top_x = top_contour[:, 0]
        top_y = top_contour[:, 1]
        bottom_x = bottom_contour[:, 0]
        bottom_y = bottom_contour[:, 1]

        linear_top = np.polyfit(top_x, top_y, 1)
        linear_bottom = np.polyfit(bottom_x, bottom_y, 1)

        fxn_top = np.poly1d(linear_top)
        fxn_bottom = np.poly1d(linear_bottom)

        centroid_y = self.centroid[1]
        x = np.arange(0, self.label_image.shape[1])
        approx_y_top = fxn_top(x)
        approx_y_bottom = fxn_bottom(x)

        approx_y_top = approx_y_top.astype(np.int32)
        approx_y_bottom = approx_y_bottom.astype(np.int32)

        approx_top_contour = np.transpose(np.vstack((x, approx_y_top)))
        approx_bottom_contour = np.transpose(np.vstack((x, approx_y_bottom)))

        return approx_top_contour, approx_bottom_contour

    def get_contour_hull(self):
        hull = cv2.convexHull(self.contour)
        return np.squeeze(hull)

    def get_contour_top(self):
        hull = self.get_contour_hull()
        hull_x = hull[:, 0]
        hull_y = hull[:, 1]

        max_hull_x = np.max(hull_x)
        min_hull_x = np.min(hull_x)

        max_index = np.where(hull_x == max_hull_x)[0]
        min_index = np.where(hull_x == min_hull_x)[0]

        if len(max_index) > 1:
            candidate_y = hull_y[max_index]
            lower_y = np.max(candidate_y)
            lower_y_index = np.where(hull_y == lower_y)
            max_index = np.intersect1d(max_index, lower_y_index)[0]
        else:
            max_index = max_index[0]

        if len(min_index) > 1:
            candidate_y = hull_y[min_index]
            lower_y = np.max(candidate_y)
            lower_y_index = np.where(hull_y == lower_y)
            min_index = np.intersect1d(min_index, lower_y_index)[0]
        else:
            min_index = min_index[0]

        smaller_index = min_index
        larger_index = max_index

        if min_index > max_index:
            smaller_index = max_index
            larger_index = min_index

        contour_1 = hull[smaller_index:larger_index + 1]

        contour_2_1 = hull[larger_index:, :]
        contour_2_2 = hull[0:smaller_index + 1, :]
        contour_2 = np.concatenate((contour_2_1, contour_2_2))

        contour_1_y = np.average(contour_1[:, 1])
        contour_2_y = np.average(contour_2[:, 1])

        if contour_1_y > contour_2_y:
            return contour_2
        else:
            return contour_1
