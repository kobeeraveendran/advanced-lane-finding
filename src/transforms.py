import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pickle

from calibrate_camera import extract_points

def perspective_transform(image, mtx, dist, src, dest):

    undist = cv2.undistort(image, mtx, dist)

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    except:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags = cv2.INTER_LINEAR)

    return warped

if __name__ == "__main__":

    image = cv2.imread("../test_images/straight_lines1.jpg")

    print("image shape: ", image.shape)

    src = np.float32([
        [212, 719], # bottom left
        [591, 456], # top left
        [687, 456], # top right
        [1096, 719] # bottom right
    ])

    dest = np.float32([
        [386, 719], # bottom left
        [386, 0],   # top left
        [894, 0],   # top right
        [894, 719]  # bottom right
    ])

    mtx, dist = extract_points(glob.glob("../camera_cal/*.jpg"))

    warped = perspective_transform(image, mtx, dist, src, dest)

    cv2.imshow("image", image)
    cv2.waitKey()

    cv2.imshow("warped", warped)
    cv2.waitKey()