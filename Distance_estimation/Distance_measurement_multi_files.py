import sys
import os
from tqdm import tqdm

if __name__ == '__main__':
    script_desc = open('Main.py')
    script_main = script_desc.read()

    root_directory = '/home/grant/Documents/Medical_images/processed_images'
    for each_dir in tqdm(os.listdir(root_directory)):
        renamed_image_dir = os.path.join(root_directory, each_dir, 'renamed_images')
        sys.argv = ['Main.py', '--folder_location', renamed_image_dir]
        exec(script_main)
    script_desc.close()
