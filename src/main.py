import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

from calibrate_camera import extract_points, camera_cal
from transforms import perspective_transform, draw_lines
from detection import threshold, base_lane_lines, fit_poly
from detection import Line

# for one image
def find_lane_lines(image, left_lane_line, right_lane_line):

    undist = cv2.undistort(image, mtx, dist)

    thresholded = threshold(undist)

    warped = perspective_transform(thresholded, mtx, dist, src, dest)

    #leftx, lefty, rightx, righty, out_img = base_lane_lines(warped)

    left_fit, right_fit, out_img = fit_poly(warped)
    
    plt.imshow(out_img, cmap = "gray")

    # plt.subplot(2, 1, 1)
    # plt.plot(peaks)
    # plt.title("Histogram peaks")

    # plt.subplot(2, 1, 2)
    # plt.imshow(warped, cmap = "gray")
    
    plt.show()

    return warped

if __name__ == "__main__":

    # calibrate camera using chessboard images
    # obtain camera matrix and distortion coefficients
    #mtx, dist = extract_points(glob.glob("../camera_cal/*"))
    dist_pickle = pickle.load(open("dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    src = np.float32([
        [250, 678], # bottom left
        [585, 456], # top left
        [698, 456], # top right
        [1057, 678] # bottom right
    ])

    dest = np.float32([
        [386, 719], # bottom left
        [386, 0],   # top left
        [894, 0],   # top right
        [894, 719]  # bottom right
    ])

    image = mpimg.imread("../test_images/test5.jpg")

    result = find_lane_lines(image)

    # plt.imshow(image)
    # plt.show()
    
    # plt.imshow(result, cmap = "gray")
    # plt.show()

    # cv2.imshow("Original", image)
    # cv2.waitKey()

    # cv2.imshow("Result", result)
    # cv2.waitKey()