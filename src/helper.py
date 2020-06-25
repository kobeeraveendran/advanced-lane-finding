import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pickle

from calibrate_camera import extract_points


def perspective_transform(image, mtx, dist, src, dest):

    M = cv2.getPerspectiveTransform(src, dest)
    M_inv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags = cv2.INTER_LINEAR)

    return warped, M, M_inv

def curvature(left_fit, right_fit, y):

    ym_per_pix = 30 / 720
    #ym_per_pix = 3 / 200

    y_eval = np.max(y)
    y_eval *= ym_per_pix

    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** (3/2)) / abs(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** (3/2)) / abs(2 * right_fit[0])

    return left_curverad, right_curverad

def offset(car_center, lane_center):

    xm_per_pix = 3.7 / 700

    if lane_center - car_center > 0:
        return "{}m left of center".format(round((lane_center - car_center) * xm_per_pix, 2))

    elif lane_center - car_center < 0:
        return "{}m right of center".format(round((lane_center - car_center) * -xm_per_pix, 2))

    else:
        return "centered"

def draw_lines(image, polygon):

    for i in range(len(polygon)):

        cv2.line(image, tuple(polygon[i]), tuple(polygon[(i + 1) % len(polygon)]), (0, 0, 255), thickness = 2)

if __name__ == "__main__":

    #image = cv2.imread("../test_images/test5.jpg")
    image = mpimg.imread("../test_images/test5.jpg")

    src = np.float32([
        [250, 678], # bottom left
        [585, 456], # top left
        [698, 456], # top right
        [1057, 678] # bottom right
    ])

    img_copy = np.copy(image)


    dest = np.float32([
        [386, 719], # bottom left
        [386, 0],   # top left
        [894, 0],   # top right
        [894, 719]  # bottom right
    ])

    #mtx, dist = extract_points(glob.glob("../camera_cal/*.jpg"))
    dist_pickle = pickle.load(open("dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undist = cv2.undistort(img_copy, mtx, dist)

    draw_lines(undist, src)

    warped = perspective_transform(image, mtx, dist, src, dest)

    draw_lines(warped, dest)

    # cv2.imshow("undist", undist)
    # cv2.waitKey()

    # cv2.imshow("warped", warped)
    # cv2.waitKey()

    plt.imshow(undist)
    plt.show()
    plt.imshow(warped)
    plt.show()