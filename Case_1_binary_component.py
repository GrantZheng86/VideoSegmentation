from Case2_binary_component import Case2BinaryComponent
import numpy as np
import cv2


class Class_1_binary_cc(Case2BinaryComponent):

    def __init__(self, cc_image, label_number):
        super().__init__(cc_image, label_number)

    def convert_to_convex_hull(self):
        hull = np.squeeze(cv2.convexHull(self.contour))
        return hull

    def get_hull_top(self, **kwargs):
        """
        Gets the top half of the convex hull. This might need some improvement because of non-clear boundary condition
        :param **kwargs:
        :return:
        """
        hull = self.convert_to_convex_hull()
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

        contour_1 = hull[smaller_index+1:larger_index]

        contour_2_1 = hull[larger_index+1:, :]
        contour_2_2 = hull[0:smaller_index, :]
        contour_2 = np.concatenate((contour_2_1, contour_2_2))

        contour_1_y = np.average(contour_1[:, 1])
        contour_2_y = np.average(contour_2[:, 1])

        if contour_1_y > contour_2_y:
            return contour_2
        else:
            return contour_1

    def valid_test(self):
        """
        Test if the current connected component is valid. This component needs to stretch a certain distance to cover
        the entire distance
        :return:
        """
        hull = self.convert_to_convex_hull()
        min_x = np.min(hull[:, 0])
        max_x = np.max(hull[:, 0])
        h, w = self.label_image.shape

        if min_x > w/3 or max_x < w/3*2:
            return False

        return True

    def visualize_with_contour(self, hold_on=False):
        background = np.zeros(self.label_image.shape)
        idx = np.where(self.label_image == self.label_num)
        background[idx] = 255
        background = background.astype(np.uint8)

        color_background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_background, [self.convert_to_convex_hull()], -1, (0, 255, 0), 2)
        cv2.imshow('contour', color_background)
        if hold_on:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def minimum_bounding_rectangle(self):
        cnt = self.contour
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w/h
        return aspect_ratio
