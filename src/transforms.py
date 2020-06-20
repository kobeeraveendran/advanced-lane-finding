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

def draw_lines(image, quad):

    print("Quad: \n", quad)

    for i in range(len(quad)):

        cv2.line(image, tuple(quad[i]), tuple(quad[(i + 1) % len(quad)]), (0, 0, 255), thickness = 2)

if __name__ == "__main__":

    image = cv2.imread("../test_images/straight_lines1.jpg")

    print("image shape: ", image.shape)

    src = np.float32([
        [250, 678], # bottom left
        [585, 456], # top left
        [698, 456], # top right
        [1057, 678] # bottom right
    ])

    img_copy = np.copy(image)

    draw_lines(img_copy, src)

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

    warped = perspective_transform(image, mtx, dist, src, dest)

    draw_lines(warped, dest)

    cv2.imshow("image", img_copy)
    cv2.waitKey()

    cv2.imshow("warped", warped)
    cv2.waitKey()