import glob
import cv2
import numpy as np
import Distance_measurement
import Feature_Extraction

IMAGE_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
             "ES & LM images & data Oct2021\\Images"


def crop_annotation(image, ruler=True):
    """
    Crop out unwanted annotations that's majorly location at the top and bottom. Keep rulers at the right side.
    x and y locations are adjustable. Default size is set in the default parameters
    :param ruler: Whether to include ruler at the right size of the image. Default to include.
    :param image: A BGR image. Must have 3 dimensions
    :return: The cropped in the desired shape
    """

    assert len(list(image.shape)) == 3, "Image must be the original BGR, not grayscale"
    if ruler:
        x_start = 147
        x_end = 926
        y_start = 285
        y_end = 903
    else:
        x_start = 147
        x_end = 926
        y_start = 285
        y_end = 863

    return image[x_start:x_end, y_start:y_end, :]


def convert_to_bw(frame):
    """
    Converts the current BGR frame to a BW image with BGR Channel
    :param frame: BGR image
    :return: the original image in BW with all color channel
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def scale_calculation(frame):
    """
    Calculates the conversion factor between pixel and actual distance. This method uses template matching, therefore,
    pre-defined number templates are required
    :param frame: A BGR image contains the right hand side ruler
    :return: The conversion factor
    """

    TEMPLATE_1_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data " \
                      "Oct2021\\ES & LM images & data Oct2021\\Template-1.png"
    TEMPLATE_2_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data " \
                      "Oct2021\\ES & LM images & data Oct2021\\Template-2.png"
    MARKER_PATH = "C:\\Users\\Grant\\OneDrive - Colorado School of Mines\VideoSegmentation\\ES & LM images & data Oct2021\\" \
                  "ES & LM images & data Oct2021\\marker_template.png"

    marker_width = 40
    useful_frame = frame[146:366, :, :]
    useful_frame_bw = convert_to_bw(useful_frame)

    template_1 = cv2.imread(TEMPLATE_1_PATH)
    template_1_bw = convert_to_bw(template_1)
    template_2 = cv2.imread(TEMPLATE_2_PATH)
    template_2_bw = convert_to_bw(template_2)

    # Searching for matching locations for the templates. After getting locations for those numbers, expanding the
    # region of interest to include tick marks on ruler for more precise distance calculation
    h_1, w_1, _ = template_1_bw.shape
    method = cv2.TM_CCORR_NORMED
    res_1 = cv2.matchTemplate(useful_frame_bw, template_1_bw, method)
    res_2 = cv2.matchTemplate(useful_frame_bw, template_2_bw, method)
    _, _, _, top_left_1 = cv2.minMaxLoc(res_1)
    _, _, _, top_left_2 = cv2.minMaxLoc(res_2)
    bottom_right_1_with_marker = (top_left_1[0] + w_1 + marker_width, top_left_1[1] + h_1)
    bottom_right_2_with_marker = (top_left_2[0] + w_1 + marker_width, top_left_2[1] + h_1)

    # Searching for tick marks on ruler based on template matching, it uses the spacing between top left corners as the
    # measure. This will have more precise measurement
    marker_template = process_marker_image(MARKER_PATH)
    marker_section_1 = useful_frame_bw[top_left_1[1]:bottom_right_1_with_marker[1],
                       top_left_1[0]:bottom_right_1_with_marker[0], :]
    marker_section_2 = useful_frame_bw[top_left_2[1]:bottom_right_2_with_marker[1],
                       top_left_2[0]:bottom_right_2_with_marker[0], :]
    res_marker_1 = cv2.matchTemplate(marker_section_1, marker_template, method)
    res_marker_2 = cv2.matchTemplate(marker_section_2, marker_template, method)
    _, _, _, top_left_marker_1 = cv2.minMaxLoc(res_marker_1)
    _, _, _, top_left_marker_2 = cv2.minMaxLoc(res_marker_2)

    top_left_marker_1_abs = (top_left_1[0], top_left_1[1] + top_left_marker_1[1])
    top_left_marker_2_abs = (top_left_2[0], top_left_2[1] + top_left_marker_2[1])

    return np.abs(top_left_marker_1_abs[1] - top_left_marker_2_abs[1])


def process_marker_image(file_name):
    """
    Process the marker image on ruler so that the "sample marker" only contains the white portion. This is for better
    template matching result when it comes for scale calculation
    :param file_name: The template file for the marker
    :return: A BGR version of the shrinked BW ruler marker
    """
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(img, 1, 2)
    contour = np.squeeze(contour[0])
    left = np.min(contour[:, 0])
    right = np.max(contour[:, 0])
    up = np.min(contour[:, 1])
    down = np.max(contour[:, 1])

    marker_image = img[up:down, left:right]
    return cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":

    imshow = False

    saving_dict = {}
    total_files = 0
    successful = 0
    unsuccessful = 0

    for file_name in glob.glob('{}/*.png'.format(IMAGE_PATH)):
        image_name = file_name.split('\\')[-1]
        frame = cv2.imread(file_name)
        frame_with_ruler = crop_annotation(frame, ruler=True)
        frame_wo_ruler = crop_annotation(frame, ruler=False)

        if "E" in image_name:  # Case Erector Spine
            spine_bottom_contour_for_show, h = Distance_measurement.get_bottom_contour(frame_wo_ruler, reduction=False,
                                                                                       show=False)
            spine_bottom_contour_for_detection, _ = Distance_measurement.get_bottom_contour(frame_wo_ruler,
                                                                                            reduction=True,
                                                                                            show=False)
            total_files += 1
            try:  # Try to look for features on the bottom of the spine
                index_of_interest = Feature_Extraction.detect_feature(spine_bottom_contour_for_detection) + 1
                template, ul, br, center, center_offset = Feature_Extraction.crop_template(index_of_interest,
                                                                                           spine_bottom_contour_for_detection,
                                                                                           frame_wo_ruler)

                if imshow:
                    no_ruler_copy = frame_wo_ruler.copy()
                    no_ruler_copy = cv2.rectangle(no_ruler_copy, ul, br, (255, 0, 0), 2)
                    no_ruler_copy = cv2.circle(no_ruler_copy, thickness=1, color=(0, 0, 255), center=center_offset, radius=2)
                    no_ruler_copy = cv2.circle(no_ruler_copy, thickness=1, color=(0, 255, 255), center=center,
                                               radius=2)
                    cv2.imshow('Template', template)
                    cv2.imshow('Region of Interest', no_ruler_copy)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                successful += 1
            except TypeError:
                print("{} Template Detection Unsuccessful".format(image_name))
                unsuccessful += 1

    print("Successful detection {}, Unsuccessful detection {}, total{}".format(successful, unsuccessful, total_files))