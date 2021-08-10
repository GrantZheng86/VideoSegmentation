import cv2
import numpy as np
import Case_2_Processing

FILE_NAME = "Debugging Frame 208.jpg"

if __name__ == "__main__":
    img = cv2.imread(FILE_NAME)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img[50:, :]

    # kernel = np.ones((7, 7), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('om', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    binary_cc = Case_2_Processing.get_binary_cc(img)
    sorted_binary_cc = Case_2_Processing.sort_component_by_area(binary_cc)

    largest = sorted_binary_cc[-1]
    second_largest = sorted_binary_cc[-2]

    largest.visualize_with_contour()
    second_largest.visualize_with_contour()

    largest_x = largest.centroid[0]
    second_x = second_largest.centroid[0]

    largest_y = largest.centroid[1]
    second_y = second_largest.centroid[1]

    position_constraint_x = second_x > largest_x
    position_constraint_y = second_y > largest_y
    position_constraint = position_constraint_x & position_constraint_y

    area_list = []
    for each_cc in sorted_binary_cc:
        area_list.append(each_cc.area)

    mean_area = np.average(area_list)
    stdv_area = np.std(area_list)
    large_area_cutoff = 2500

    print("Mean Area {}".format(mean_area))
    print("Standard Deviation {}".format(stdv_area))
    print("Large Area Criteria {}".format(large_area_cutoff))
    print("Largest {}".format(largest.area))
    print("Second {}".format(second_largest.area))

    print("Largest Area Cutoff {}".format(largest.area_cutoff))
    print("Second Area Cutoff {}".format(second_largest.area_cutoff))

    large_area_criteria = second_largest.area > large_area_cutoff
    top_cutoff_criteria = largest.area_cutoff & second_largest.area_cutoff
    print("Large Area: {} ||| Top Cutoff {} ||| position_constraint {}".format(large_area_criteria, top_cutoff_criteria,
                                                                            position_constraint))

    if large_area_criteria and not top_cutoff_criteria:
        print( True)
    else:
        print(False)
