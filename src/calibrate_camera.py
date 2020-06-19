import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

import sys
import os

def extract_points(images):

    '''
    args:
        - images: list of strings containing the filenames of the calibration image set
    
    returns:
        - obj_points: list of points in 3-D; i.e. in real-world space
        - img_points: list of points in 2d; i.e. in planar image space
    '''

    obj = np.zeros((6 * 9, 3), np.float32)
    obj[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    for i, filename in enumerate(images):

        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            obj_points.append(obj)
            img_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obs_points, img_points, gray.shape[::-1], None, None)

    return gray, obj_points, img_points


def calibrate_camera(filename, mtx, dist):

    '''
    args:
        - image: grayscale image (converted from the BGR original)
        - obj_points: list of points in 3-D; i.e. in real-world space
        - img_points: list of points in 2d; i.e. in planar image space

    returns:
        - ret: bool
        - mtx: camera matrix
        - dist: distortion coefficients
        - rvecs: rotation vectors
        - tvecs: translation vectors
    '''

    split = filename.split('.')

    image = cv2.imread(filename)


    # undistort image
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    # write to new image for checking purposes
    cv2.imwrite("{}_undist.{}".format(''.join(split[0:-1]), split[-1]), dst)

    return dst

if __name__ == "__main__":

    if len(sys.argv > 1):

        # preferably a path without a trailing '/'
        image_list = glob.glob(sys.argv[1] + "/*")

    else:
        image_list = glob.glob("../camera_cal/*")

    _, mtx, dist, _, _ = extract_points(image_list)

    #dst = calibrate_camera()