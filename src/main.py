import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

from calibrate_camera import extract_points, camera_cal
from helper import perspective_transform, draw_lines, curvature, offset
from detection import threshold, base_lane_lines, fit_poly
from detection import Line

# for one image
def find_lane_lines(image, left_lane_line, right_lane_line):

    undist = cv2.undistort(image, mtx, dist)

    thresholded = threshold(undist)

    warped, M, M_inv = perspective_transform(thresholded, mtx, dist, src, dest)

    #leftx, lefty, rightx, righty, out_img = base_lane_lines(warped)

    if not (left_lane_line.detected or right_lane_line.detected):
        leftx, lefty, rightx, righty, car_center, lane_center = base_lane_lines(warped)
        left_fit, right_fit, left_fitx, right_fitx, y = fit_poly(warped.shape, leftx, lefty, rightx, righty)

        left_curve, right_curve = curvature(left_fit, right_fit, y)

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    points_left = np.array([np.transpose(np.vstack([left_fitx, y]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y])))])
    points = np.hstack((points_left, points_right))

    cv2.fillPoly(color_warp, np.int_([points]), (0, 255, 0))

    new_warp = cv2.warpPerspective(color_warp, M_inv, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)

    offset_text = offset(car_center, lane_center)

    cv2.putText(
        result, 
        "Radius of Curvature: {}m".format(round(min(left_curve, right_curve), 1)), 
        (100, 100), 
        cv2.FONT_HERSHEY_COMPLEX, # use complex
        1, 
        (0, 255, 0), 
        2
    )

    cv2.putText(
        result, 
        "Vehicle is {}".format(offset_text), 
        (100, 150), 
        cv2.FONT_HERSHEY_COMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )

    #plt.imshow(warped, cmap = "gray")
    plt.imshow(result)
    
    plt.show()

    return result

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

    # for image in glob.glob("../test_images/*.jpg"):
    #     image = mpimg.imread(image)

    #     left_lane_line = Line()
    #     right_lane_line = Line()

    #     result = find_lane_lines(image, left_lane_line, right_lane_line)

    image = mpimg.imread("../test_images/straight_lines1.jpg")

    left_lane_line = Line()
    right_lane_line = Line()

    result = find_lane_lines(image, left_lane_line, right_lane_line)

    # plt.imshow(image)
    # plt.show()
    
    # plt.imshow(result, cmap = "gray")
    # plt.show()

    # cv2.imshow("Original", image)
    # cv2.waitKey()

    # cv2.imshow("Result", result)
    # cv2.waitKey()