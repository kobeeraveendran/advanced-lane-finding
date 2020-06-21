import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

from calibrate_camera import extract_points, camera_cal
from transforms import perspective_transform, draw_lines

# for one image
def find_lane_lines(image):

    undist = cv2.undistort(image, mtx, dist)

    warped = perspective_transform(undist, mtx, dist, src, dest)



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