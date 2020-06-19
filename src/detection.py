import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

def threshold(image):

    # convert to HLS color space
    try:
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    except:
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    s_channel = hls[:, :, 2]

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    except:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    sobelx_binary = np.zeros_like(scaled_sobel)
    sobelx_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(s_binary == 1) | (sobelx_binary)] = 1

    return combined_binary


if __name__ == "__main__":

    combined_binary = threshold("../test_images/straight_lines1.jpg")