import cv2 as cv
import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from scipy import stats
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def getRandomVideoFrame(fileName):
    cwd = os.getcwd()
    cap = cv.VideoCapture(fileName)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frameNum = randint(0, length)

    success, image = cap.read()
    saved = False

    count = 0
    while success and not saved:
        if count == frameNum:
            fileNameToSave = "{} extracted.jpg".format(fileName)
            fileNameToSave = os.path.join(cwd, fileNameToSave)
            cv.imwrite(fileNameToSave, cropDesiredImage(image))
            saved = True
            return
        else:
            success, image = cap.read()
            count += 1


def cropDesiredImage(img):
    # Crops the top of the image where letters are
    cropped = img[143:, 0:575, :]
    return cropped


def threholdingImage(img):
    img = cv.GaussianBlur(img, (3, 3), 0)
    _, th = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
    _, th_ostu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    th_otsu_closing = cv.morphologyEx(th_ostu, cv.MORPH_CLOSE, kernel)
    th_otsu_opening = cv.morphologyEx(th_ostu, cv.MORPH_OPEN, kernel)

    toshow = np.hstack((img, th_ostu, th_otsu_closing, th_otsu_opening))

    # return th_otsu_opening
    return th_otsu_opening


def thresholdingVideo(fileName):
    cap = cv.VideoCapture(fileName)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    cwd = os.getcwd()
    out_name = "segmented_video.avi"
    out_name = os.path.join(cwd, out_name)
    out = cv.VideoWriter(out_name, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (2300, 937))

    while cap.isOpened():
        success, image = cap.read()

        if success:
            image = cropDesiredImage(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            currFrame = threholdingImage(image)
            cv.imshow("Thresholding", currFrame)
            currFrame = cv.cvtColor(currFrame, cv.COLOR_GRAY2RGB)
            out.write(currFrame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()


def gamma_correction(alpha, beta, image):
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    return new_image


# Press the green button in the gutter to run the script.

# def drawContours(cnt, images):
#     for curr_cnt in cnt:
#         epsilon = 0.01 * cv.arcLength(curr_cnt, True)
#
#         if len(curr_cnt) > 4:
#             # approx = cv.fitEllipse(np.squeeze(curr_cnt))
#             # cv.ellipse(images, approx, (0, 255, 0), 2)
#             hull = cv.convexHull(np.squeeze(curr_cnt))
#             ellipse = cv.fitEllipse(hull)
#             # cv.drawContours(images, [hull], -1, (0, 255, 0), 2)
#             cv.ellipse(images, ellipse, (0, 255, 0), 2)
#
#     # approx = cv.fitEllipse(np.squeeze(cnt[10]))
#     # cv.ellipse(images, approx, (0, 255, 0), 2)
#     cv.imshow("Coutours", images)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def fitEllipseToConvexHull(hull, img, show_image=False):
    ellipse_list = []

    for each_hull in hull:
        ellipse = cv.fitEllipse(each_hull)
        ellipse_list.append(ellipse)

    if show_image:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for each_ellipse in ellipse_list:
            cv.ellipse(img, each_ellipse, (0, 255, 0), 2)

        cv.imshow('Ellipses', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return ellipse_list


def fitEllipseToContours(contours, img, show_image=False):
    ellipse_list = []

    for each_cnt in contours:
        ellipse = cv.fitEllipse(each_cnt)
        ellipse_list.append(ellipse)

    if show_image:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for each_ellipse in ellipse_list:
            cv.ellipse(img, each_ellipse, (0, 255, 0), 2)

        cv.imshow('Ellipses', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return ellipse_list

def draw_contours(cnt, images):
    img = cv.cvtColor(images, cv.COLOR_GRAY2BGR)
    cv.drawContours(img, cnt, -1, (0, 255, 0), 2)
    cv.imshow('Contours', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_convex_hull(cnt, img):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    hull_list = []

    for each_cnt in cnt:
        hull = cv.convexHull(np.squeeze(each_cnt))
        hull_list.append(hull)

    cv.drawContours(img, hull_list, -1, (0, 255, 0), 2)
    cv.imshow('Convex Hull', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return hull_list
def filterSmallContours(contours):
    contour_size_list = []
    for cnt in contours:
        contour_size_list.append(len(cnt))

    contour_mean = np.average(contour_size_list)
    contour_stdv = np.std(contour_size_list)

    filtered_contours = []

    for cnt in contours:
        if len(cnt) > contour_mean:
            filtered_contours.append(cnt)

    return filtered_contours

def findAveragePixelIntensity(cnt,img):
    cimg = np.zeros_like(img)
    cv.drawContours(cimg, cnt, -1, color=255, thickness=-1)
    pts = np.where(cimg == 255)
    intensities = np.sum(img[pts], dtype=np.float)
    pixel_count = np.sum(cimg) / 255.0

    return intensities / pixel_count

def getConvexHullAndStats(binary_img, original_image):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_img, connectivity=8)
    contours, hierarchy = cv.findContours(binary_img, 1, 2)
    filtered_contours = filterSmallContours(contours)

    return_ellipse = []
    return_intensity = []
    return_hull = []
    for each_cnt in filtered_contours:
        label = getEnclosedAreaLabel(labels, each_cnt, binary_img)
        label = label[0]

        if label != 0:
            ellipse = cv.fitEllipse(each_cnt)
            average_intensity = findAveragePixelIntensity([each_cnt], original_image)
            hull = cv.convexHull(np.squeeze(each_cnt))

            return_ellipse.append(ellipse)
            return_intensity.append(average_intensity)
            return_hull.append(hull)
            print(average_intensity)
            print(label)

    return return_ellipse, return_intensity, return_hull


def contourEncirclementTest(cnt1, cnt2):
    l1 = len(cnt1)
    l2 = len(cnt2)

    large_cnt = None
    small_cnt = None
    if l1 > l2:
        large_cnt = cnt1
        small_cnt = cnt2
    else:
        large_cnt = cnt2
        small_cnt = cnt1

    for point in small_cnt:
        inside = cv.pointPolygonTest(large_cnt, (point[0], point[1]), False)

        if inside > 0:
            return True

    return False

def getEnclosedAreaLabel(label_img, cnt, thresh_img=None):
    cimg = np.zeros_like(label_img)
    cv.drawContours(cimg, [cnt], -1, color=255, thickness=-1)

    if thresh_img is not None:
        th_img_color = cv.cvtColor(thresh_img, cv.COLOR_GRAY2RGB)
        cv.drawContours(th_img_color, [cnt], -1, color=(0, 255, 0), thickness=1)
        # cv.imshow('DEBUG', th_img_color)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    # plt.imshow(cimg)
    # plt.show()
    pts = np.where(cimg == 255)
    toReturn = label_img[pts[0], pts[1]]
    toReturn = stats.mode(toReturn, axis=None)
    return toReturn[0]





def getIntensityInBoundary(img, contours):
    lst_intensities = []

    for i in range(len(contours)):
        cimg = np.zeros_like(img)
        cv.drawContours(cimg, contours, i, color=255, thickness=-1)
        pts = np.where(cimg == 255)
        lst_intensities.append(img[pts[0], pts[1]])

    return lst_intensities

def drawEllipseInformation(ellipse, original_image):

    (xc,yc),(d1,d2),angle_org = ellipse

    rmajor = max(d1, d2) / 2
    if angle_org > 90:
        angle = angle_org - 90
    else:
        angle = angle_org + 90

    xtop = xc + math.cos(math.radians(angle)) * rmajor
    ytop = yc + math.sin(math.radians(angle)) * rmajor
    xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
    ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
    cv.line(original_image, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 0, 255), 2)

    rminor = min(d1, d2) / 2
    if angle_org > 90:
        angle = angle_org
    else:
        angle = angle_org

    xtop = xc + math.cos(math.radians(angle)) * rminor
    ytop = yc + math.sin(math.radians(angle)) * rminor
    xbot = xc + math.cos(math.radians(angle + 180)) * rminor
    ybot = yc + math.sin(math.radians(angle + 180)) * rminor

    cv.line(original_image, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 0, 255), 2)
    cv.circle(original_image, (int(xc), int(yc)), 3, (255, 255, 255), -1)

    return original_image

def comprehensiveVisualization(ellipse, intensity, hull, original_image):

    l = len(ellipse)
    original_image = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
    font = cv.FONT_HERSHEY_SIMPLEX
    intensity_color = (255, 0, 255)
    label_color = (255, 255, 0)
    fontScale = 0.5
    thickness = 1

    for i in range(l):
        curr_ellipse = ellipse[i]
        xc, yc = curr_ellipse[0]
        xc = np.int(xc)
        yc = np.int(yc)
        org = (xc, yc)
        returned_img = drawEllipseInformation(ellipse[i], original_image)
        cv.drawContours(returned_img, [hull[i]], -1, (0, 255, 0), 2)
        returned_img = cv.putText(returned_img, str(intensity[i])[:5], org, font,
                            fontScale, intensity_color, thickness, cv.LINE_AA)
        cv.imshow('Ellipse_info', returned_img)
        cv.waitKey(0)
        cv.destroyAllWindows()



if __name__ == '__main__':
    # getRandomVideoFrame("Vid_1.mp4")
    img = cv.imread('Vid_1.mp4 extracted.jpg')
    img = img.astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gamma_corrected = gamma_correction(1.5, 0, np.expand_dims(img, axis=2))
    gamma_corrected = np.squeeze(gamma_corrected)
    gamma_comp = np.hstack((gamma_corrected, img))
    plt1 = plt.figure(1)
    plt.imshow(gamma_comp, cmap='gray')

    gamma_thresh = threholdingImage(gamma_corrected)
    ellipse, intensity, hull = getConvexHullAndStats(gamma_thresh, img)
    comprehensiveVisualization(ellipse, intensity, hull, img)
    # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(gamma_thresh, connectivity=8)
    # contours, hierarchy = cv.findContours(gamma_thresh, 1, 2)
    # filtered_contours = filterSmallContours(contours)
    # draw_contours(filtered_contours, gamma_thresh)
    # hull_list = draw_convex_hull(filtered_contours, gamma_thresh)
    # intensity_list = getIntensityInBoundary(img, filtered_contours)
    # ellipse_list = fitEllipseToConvexHull(hull_list, gamma_thresh, True)
    # ellipse_list_cnt = fitEllipseToContours(filtered_contours, gamma_thresh, True)
    # # drawContours(contours, cv.cvtColor(gamma_thresh, cv.COLOR_GRAY2RGB))
    # img_thresh = threholdingImage(img)
    # plt2 = plt.figure(2)
    # plt.imshow(np.hstack((gamma_thresh, img_thresh)), cmap='gray')
    # plt.show()




















