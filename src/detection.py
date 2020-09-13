import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

class Line():

    def __init__(self):

        # whether the line was detected in the previous frame
        self.detected = False

        # x vals of the last n fits of the line
        self.recent_xfitted = []

        # avg x vals from the last n fits
        self.bestx = None

        # polynomial coeffs avg'd from last n fits
        self.best_fit = None

        # polynomial coeffs from previous fit
        self.prev_fit = [np.array([False])]

        # x and y vals from the current and previous detections
        self.x = None
        self.y = None
        self.prev_x = None
        self.prev_y = None

        # x and y fits from current and previous poly. fit
        self.fit_x = None
        self.fit_y = None
        self.prev_fit_x = None
        self.prev_fit_y = None

        # polynomial coeffs of current fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line
        self.radius_of_curvature = None

        # line position in image (not the distance from line to car as suggested; this will later be calculated using the base pos of L+R lane lines)
        self.line_base_pos = None

        # diff between prev and current line of best fit coefficients
        self.diffs = np.array([0, 0, 0], dtype = 'float')

        

def threshold(image):
    # convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    sobelx_binary = np.zeros_like(scaled_sobel)
    sobelx_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 100) & (s_channel <= 255)] = 1

    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(s_binary == 1) | (sobelx_binary == 1)] = 1

    # generate region of interest mask
    vertices = np.array([[(0, combined_binary.shape[0]), (550, 456), (730, 456), (combined_binary.shape[1], combined_binary.shape[0])]], dtype = np.int32)
    mask = np.zeros_like(combined_binary)

    if len(combined_binary.shape) > 2:
        ignore_mask_color = (255,) * combined_binary.shape[2]
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_combined_binary = cv2.bitwise_and(combined_binary, mask)

    return masked_combined_binary

def histogram_peaks(image):
    hist = np.sum(image[image.shape[0] // 2:, :], axis = 0)

    car_center = np.int(hist.shape[0] // 2)
    left_base = np.argmax(hist[:car_center])
    right_base = np.argmax(hist[car_center:]) + car_center

    lane_center = (left_base + right_base) // 2

    return left_base, right_base, car_center, lane_center

def sliding_window(image, left_lane_line, right_lane_line):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis = 0)

    out_img = np.dstack((image, image, image))

    mid = np.int(histogram.shape[0] // 2)
    left_base = np.argmax(histogram[:mid])
    right_base = np.argmax(histogram[mid:]) + mid
    car_center = mid
    lane_center = (left_base + right_base) // 2

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

        cv2.rectangle(out_img, (leftx_start, y_start), (leftx_end, y_end), (0, 255, 0), 2)
        cv2.rectangle(out_img, (rightx_start, y_start), (rightx_end, y_end), (0, 255, 0), 2)

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

    left_lane_line.detected = True
    right_lane_line.detected = True

    left_lane_line.prev_x = left_lane_line.x
    left_lane_line.x =  left_x
    left_lane_line.prev_y = left_lane_line.y
    left_lane_line.y =  left_y

    right_lane_line.prev_x = right_lane_line.x
    right_lane_line.x = right_x
    right_lane_line.prev_y = right_lane_line.y
    right_lane_line.y = right_y

    left_lane_line, right_lane_line, y, out_img = fit_poly(image.shape, left_lane_line, right_lane_line, out_img)

    return left_lane_line, right_lane_line, car_center, lane_center, y, out_img

def prior_frame_search(warped, margin, left_fit, right_fit):

    _, _, car_center, lane_center = histogram_peaks(warped)

    nonzero = warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    poly_left = nonzeroy ** 2 * left_fit[0]  + nonzeroy * left_fit[1] + left_fit[2]
    poly_right = nonzeroy ** 2 * right_fit[0] + nonzeroy * right_fit[1] + right_fit[2]

    left_lane_inds = ((nonzerox >= poly_left - margin) & (nonzerox < poly_left + margin))
    right_lane_inds = ((nonzerox >= poly_right - margin) & (nonzerox < poly_right + margin))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, car_center, lane_center

# fit the detected lane pixels a polynomial
def fit_poly(img_shape, left_lane_line, right_lane_line, out_img):

    left_fit = np.polyfit(left_lane_line.y, left_lane_line.x, deg = 2)
    right_fit = np.polyfit(right_lane_line.y, right_lane_line.x, deg = 2)

    y = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_fitx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_fitx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

    out_img[left_lane_line.y, left_lane_line.x] = [255, 0, 0]
    out_img[right_lane_line.y, right_lane_line.x] = [0, 0, 255]

    left_lane_line.prev_fit_x = left_lane_line.fit_x
    left_lane_line.fit_x = left_fit

    right_lane_line.prev_fit_x = right_lane_line.fit_x
    right_lane_line.fit_x = right_fit

    left_lane_line.current_fit.append(left_fit)
    right_lane_line.current_fit.append(right_fit)

    left_lane_line.fit_y = y
    right_lane_line.fit_y = y

    return left_lane_line, right_lane_line, y, out_img

if __name__ == "__main__":

    combined_binary = threshold(cv2.imread("../test_images/straight_lines1.jpg"))
    #cv2.imshow("combined binary", combined_binary)
    #cv2.waitKey()

    # plt.imshow(combined_binary, cmap = "gray")
    # plt.show()