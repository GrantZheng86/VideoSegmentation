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


def post_processing_words(words_list):
    # # 4 meaning all of the has been detected
    # if len(words_list) == 4:
    #     return words_list
    #
    # if len(words_list) == 3:
    #     if valid_word(words_list):
    #         return words_list

    if valid_file_name(words_list):
        return words_list

    # Contraction / Relaxation Cases
    last_word_first_letter = words_list[-1][0]
    if 'R' != last_word_first_letter and 'C' != last_word_first_letter:
        last_word = words_list[-1][-2:]
        second_list_word = words_list[-1][:-2]
        words_list.pop()
        words_list.append(second_list_word)
        words_list.append(last_word)
        return post_processing_words(words_list)

    # Test if LM is present in the second last word
    second_last_word = words_list[-2]
    if 'LM' in second_last_word and len(second_last_word) > 2:
        third_last_word = second_last_word[:-2]
        second_last_word = second_last_word[-2:]
        assert second_last_word == 'LM'
        insert_index = len(words_list) - 2
        del words_list[-2]

        words_list.insert(insert_index, second_last_word)
        words_list.insert(insert_index, third_last_word)
        return post_processing_words(words_list)

    # Test if the first word is properly split
    first_word = words_list[0]
    if 'BL' in first_word or 'ES' in first_word:
        true_first_word = first_word[:-2]
        second_word = first_word[-2:]
        assert second_word == 'BL' or second_word == 'ES'
        del words_list[0]

        words_list.insert(0, second_word)
        words_list.insert(0, true_first_word)
        return post_processing_words(words_list)

    words_list.insert(0, 'Manual Annotation Required')
    return words_list


def valid_word(word_list):
    # Each single word should not contain more than 3 char
    for word in word_list:
        if len(word) > 3:
            return False

    return True


def valid_file_name(word_list):

    if not valid_word(word_list):
        return False

    # Test first position
    first_word = word_list[0]

    if first_word[0] != 'P':
        return False
    if not first_word[1:].isnumeric():
        return False

    # Test Second position
    second_word_candidate = ['BL', 'ES', '2WK', '4WK']
    second_word = word_list[1]
    if second_word not in second_word_candidate:
        return False

    # If the file has 4 words:
    if len(word_list) == 4:
        third_word = word_list[2]
        if third_word != 'LM':
            return False

    last_word = word_list[-1]
    if len(last_word) != 2:
        return False
    last_word_first_letter = last_word[0]
    last_word_second_letter = last_word[1]
    if last_word_first_letter != 'R' and last_word_first_letter != 'C':
        return False
    if not last_word_second_letter.isnumeric():
        return False

    return True


if __name__ == '__main__':
    saving_directory = r'renamed_images'
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    for image_name in glob.glob('{}/*.png'.format(IMAGE_FOLDER)):
        image_bgr = cv2.imread(image_name)
        filtered_image = color_filter(image_bgr, debug=False)
        detected_text = pytesseract.image_to_string(filtered_image)
        detected_text = detected_text[:-1]
        try:
            post_processed_text = post_processing_words(detected_text.split(' '))
            post_processed_text = " ".join(post_processed_text)
        except:
            post_processed_text = "Manual Annotation Required {}".format(detected_text[:-1])

        new_image_name = post_processed_text + '.jpg'
        new_image_path = os.path.join(saving_directory, new_image_name)
        cv2.imwrite(new_image_path, image_bgr)



