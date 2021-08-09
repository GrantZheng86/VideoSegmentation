import cv2
import numpy as np
import Case_2_Processing

FILE_NAME = "Segmentation Trial 2/Debugging Frame 5.jpg"

if __name__ == "__main__":
    img = cv2.imread(FILE_NAME)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[10:, :]
    cv2.imshow('om', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    binary_cc = Case_2_Processing.get_binary_cc(img)
    sorted_binary_cc = Case_2_Processing.sort_component_by_area(binary_cc)

    largest = sorted_binary_cc[-1]
    second_largest = sorted_binary_cc[-2]

    largest.visualize_with_contour()
    second_largest.visualize_with_contour()

    area_list = []
    for each_cc in sorted_binary_cc:
        area_list.append(each_cc.area)

    mean_area = np.average(area_list)
    stdv_area = np.std(area_list)
    large_area_cutoff = mean_area + 2*stdv_area

    print("Mean Area {}".format(mean_area))
    print("Standard Deviation {}".format(stdv_area))
    print("Large Area Criteria {}".format(large_area_cutoff))
    print("Largest {}".format(largest.area))
    print("Second {}".format(second_largest.area))

    print("Largest Area Cutoff {}".format(largest.area_cutoff))
    print("Second Area Cutoff {}".format(second_largest.area_cutoff))

    large_area_criteria = second_largest.area > large_area_cutoff
    top_cutoff_criteria = largest.area_cutoff & second_largest.area_cutoff
    print("Large Area: {} ||| Top Cutoff {}".format(large_area_criteria, top_cutoff_criteria))

    if large_area_criteria and not top_cutoff_criteria:
        print( True)
    else:
        print(False)
