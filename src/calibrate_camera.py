import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import sys
import os
import pickle


def extract_points(images):

    '''
    args:
        - images: list of strings containing the filenames of the calibration image set
    
    returns:
        - mtx: camera calibration matrix
        - dist: distortion coefficients
    '''

    obj = np.zeros((6 * 9, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    for filename in images:

        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            obj_points.append(obj)
            img_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist

    pickle.dump(dist_pickle, open("dist_pickle.p", "wb"))

    return mtx, dist


def camera_cal(image, mtx, dist, filename = None, save = False):

    '''
    args:
        - filename: filename (paths also accepted)
        - mtx: camera matrix from `extract_points()`, or loaded from saved file
        - dist: distortion coefficients from `extract_points()`, or loaded from saved file

    returns:
        - dst: undistorted image
    '''
    if filename:
        image = mpimg.imread(filename)
        

    # undistort image
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    # write to new image for checking purposes
    if save:
        split = filename.split('.')
        new_filename = filename.split('.')[-2].split('/')[-1]
        cv2.imwrite("../undistorted/{}_undist.{}".format(new_filename, split[-1]), dst)

    return dst

if __name__ == "__main__":

    if len(sys.argv) > 1:

        # preferably a path without a trailing '/'
        image_list = glob.glob(sys.argv[1] + "/*")

    else:
        image_list = glob.glob("../camera_cal/*")

    mtx, dist = extract_points(image_list)

    os.makedirs("../undistorted/", exist_ok = True)

    dst = camera_cal("../camera_cal/calibration1.jpg", mtx, dist)