import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

from moviepy.editor import VideoFileClip
import os

from calibrate_camera import extract_points, camera_cal
from helper import perspective_transform, curvature, draw_lane_lines, offset
from detection import threshold, base_lane_lines, fit_poly, prior_frame_search
from detection import Line

# for one image
#def find_lane_lines(image, left_lane_line, right_lane_line):
def find_lane_lines(image):

    undist = cv2.undistort(image, mtx, dist)

    thresholded = threshold(undist)

    warped, M, M_inv = perspective_transform(thresholded, mtx, dist, src, dest)

    plt.imshow(thresholded, cmap = "gray")
    plt.show()

    #if not (left_lane_line.detected or right_lane_line.detected):
    leftx, lefty, rightx, righty, car_center, lane_center = base_lane_lines(warped)
    left_fit, right_fit, left_fitx, right_fitx, y = fit_poly(warped.shape, leftx, lefty, rightx, righty)

    # print(left_fit)
    # print(right_fit)

    # update Line objects
    # left_lane_line.detected = True
    # right_lane_line.detected = True

    if abs(offset(car_center, lane_center)) > 0.35:

        leftx, lefty, rightx, righty, car_center, lane_center = prior_frame_search(warped, 100, left_lane_line.current_fit, right_lane_line.current_fit)
        left_fit, right_fit, left_fitx, right_fitx, y = fit_poly(warped.shape, leftx, lefty, rightx, righty)

    # sanity checks
    #if abs(lane_center - car_center) > 0.3:


    left_curve, right_curve = curvature(left_fit, right_fit, y)

    # update Line objects
    left_lane_line.current_fit = left_fit
    right_lane_line.current_fit = right_fit
    left_lane_line.radius_of_curvature = left_curve
    right_lane_line.radius_of_curvature = right_curve

    # leftx, lefty, rightx, righty, car_center, lane_center = base_lane_lines(warped)
    # left_fit, right_fit, left_fitx, right_fitx, y = fit_poly(warped.shape, leftx, lefty, rightx, righty)

    # left_curve, right_curve = curvature(left_fit, right_fit, y)

    result = draw_lane_lines(image, undist, warped, left_fitx, right_fitx, y, M_inv, car_center, lane_center, left_curve, right_curve)

    #plt.imshow(warped, cmap = "gray")
    # plt.imshow(result)
    
    # plt.show()

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

    for image in glob.glob("../test_images/*.jpg"):
        image = mpimg.imread(image)

        left_lane_line = Line()
        right_lane_line = Line()

        #result = find_lane_lines(image, left_lane_line, right_lane_line)
        result = find_lane_lines(image)

        plt.imshow(result)
        plt.show()


    # begin comment (for video generation)
    # left_lane_line = Line()
    # right_lane_line = Line()

    # os.makedirs("../output_videos/", exist_ok = True)
    
    # input_vid = VideoFileClip("../project_video.mp4")
    # process_clip = input_vid.fl_image(find_lane_lines)
    # process_clip.write_videofile("../output_videos/project_video_output.mp4", audio = False)
    # end video generation comment

    # image = mpimg.imread("../test_images/straight_lines1.jpg")

    # left_lane_line = Line()
    # right_lane_line = Line()

    # result = find_lane_lines(image, left_lane_line, right_lane_line)

    # plt.imshow(image)
    # plt.show()
    
    # plt.imshow(result, cmap = "gray")
    # plt.show()

    # cv2.imshow("Original", image)
    # cv2.waitKey()

    # cv2.imshow("Result", result)
    # cv2.waitKey()