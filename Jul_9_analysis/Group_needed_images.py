import os
import glob
import shutil
import pandas as pd

ROOT_DIRECTORY = r'C:\Users\Grant\Downloads\OneDrive_2022-07-08\Processed Images'
LABEL_PATH = 'Patient_labels.xlsx'
IMAGE_FORMAT = '*.jpg'
REGROUPED_PATH = 'Regrouped_images'

if __name__ == '__main__':
    if os.path.exists(REGROUPED_PATH):
        shutil.rmtree(REGROUPED_PATH)
    os.makedirs(REGROUPED_PATH)

    for sub_folder in os.listdir(ROOT_DIRECTORY):
        dir_to_loop = os.path.join(ROOT_DIRECTORY, sub_folder, 'renamed_images')
        for image_name in glob.glob(os.path.join(dir_to_loop, IMAGE_FORMAT)):
            if 'ES' in image_name:
                shutil.copy(image_name, REGROUPED_PATH)
