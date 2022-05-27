import cv2
import glob
import os
import shutil

if __name__ == '__main__':
    X_MIN = 278
    X_MAX = X_MIN+627
    Y_MIN = 0
    Y_MAX = Y_MIN + 1080

    saving_dir = 'cropped_imgs'
    if os.path.exists(saving_dir):
        shutil.rmtree(saving_dir)
    os.makedirs(saving_dir)

    for img_name in glob.glob('*.png'):
        img = cv2.imread(img_name)
        img_resized = img[Y_MIN:Y_MAX, X_MIN:X_MAX, :]
        img_name = img_name.split('.')[0]
        img_resized_path = os.path.join(saving_dir, '{}.jpg'.format(img_name))
        cv2.imwrite(img_resized_path, img_resized)

        # cv2.imshow('resized', img_resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()