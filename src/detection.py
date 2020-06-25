import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

class Line():

    def __init__(self):

        # whether the line was detected in the previous time-step
        self.detected = False

        # x vals of the most recent n line fits
        self.recent_xfitted = []

        # avg x vals from the last n fits
        self.bestx = None

        # polynomial coeffs avg'd from last n fits
        self.best_fit = None

        # polynomial coeffs from the last fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line
        self.radius_of_curvature = None

        # offset from line to vehicle center
        self.offset = None

        # diff between prev and current line of best fit coefficients
        self.diffs = np.array([0, 0, 0], dtype = 'float')

        # x values of detected line pixels
        self.all_x = None

        # y values of detected line pixels
        self.all_y = None

        

def threshold(image):

    # convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray, cmap = "gray")
    # plt.show()

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    sobelx_binary = np.zeros_like(scaled_sobel)
    sobelx_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    # plt.imshow(sobelx_binary, cmap = "gray")
    # plt.show()

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1
    
    # plt.imshow(s_binary, cmap = "gray")
    # plt.show()

    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(s_binary == 1) | (sobelx_binary == 1)] = 1

    # plt.imshow(combined_binary, cmap = "gray")
    # plt.show()

    return combined_binary

def histogram_peaks(image):
    hist = np.sum(image[image.shape[0] // 2:, :], axis = 0)

    car_center = np.int(hist.shape[0] // 2)
    left_base = np.argmax(hist[:car_center])
    right_base = np.argmax(hist[car_center:]) + car_center

    lane_center = (left_base + right_base) // 2

    return left_base, right_base, car_center, lane_center

def base_lane_lines(image):

    #plt.imshow(image, cmap = "gray")

    # histogram = np.sum(image[image.shape[0] // 2:, :], axis = 0)

    # #out_img = np.dstack((image, image, image))

    # # identify x coord of left and right lane lines
    # mid = np.int(histogram.shape[0] // 2)
    # left_base = np.argmax(histogram[:mid])
    # right_base = np.argmax(histogram[mid:]) + mid

    left_base, right_base, car_center, lane_center = histogram_peaks(image)

    # set up sliding window procedure
    num_windows = 9
    margin = 100
    min_pix = 50

    win_height = np.int(image.shape[0] // num_windows)

    # all activated pixels
    nonzero = image.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    left_current = left_base
    right_current = right_base

    left_lane_indices = []
    right_lane_indices = []

    for window in range(num_windows):
        leftx_start = left_current - margin
        rightx_start = right_current - margin
        leftx_end = left_current + margin
        rightx_end = right_current + margin

        y_start = image.shape[0] - (window + 1) * win_height
        y_end = image.shape[0] - window * win_height

        # cv2.rectangle(out_img, (leftx_start, y_start), (leftx_end, y_end), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (rightx_start, y_start), (rightx_end, y_end), (0, 255, 0), 2)

        # extract nonzero pixels in windows
        left_inds = ((nonzero_y >= y_start) & (nonzero_y < y_end) & (nonzero_x >= leftx_start) & (nonzero_x < leftx_end)).nonzero()[0]
        right_inds = ((nonzero_y >= y_start) & (nonzero_y < y_end) & (nonzero_x >= rightx_start) & (nonzero_x < rightx_end)).nonzero()[0]

        left_lane_indices.append(left_inds)
        right_lane_indices.append(right_inds)

        if len(left_inds) > min_pix:
            left_current = np.int(np.mean(nonzero_x[left_inds]))
        
        if len(right_inds) > min_pix:
            right_current = np.int(np.mean(nonzero_x[right_inds]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    return left_x, left_y, right_x, right_y, car_center, lane_center

def fit_poly(img_shape, leftx, lefty, rightx, righty):

    # leftx, lefty, rightx, righty, out_img = base_lane_lines(warped_image)

    left_fit = np.polyfit(lefty, leftx, deg = 2)
    right_fit = np.polyfit(righty, rightx, deg = 2)
    
    y = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_fitx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_fitx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # plt.plot(left_fitx, y, color = "yellow")
    # plt.plot(right_fitx, y, color = "yellow")

    return left_fit, right_fit, left_fitx, right_fitx, y

# def prior_frame_search(warped, margin):

if __name__ == "__main__":

    combined_binary = threshold(cv2.imread("../test_images/straight_lines1.jpg"))
    #cv2.imshow("combined binary", combined_binary)
    #cv2.waitKey()

    # plt.imshow(combined_binary, cmap = "gray")
    # plt.show()