import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pickle

from calibrate_camera import extract_points


def perspective_transform(image, mtx, dist, src, dest):

    M = cv2.getPerspectiveTransform(src, dest)
    M_inv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags = cv2.INTER_LINEAR)

    return warped, M, M_inv

# def curvature(left_fit, right_fit, y):

#     xm_per_pix = 3.7 / 700
#     ym_per_pix = 30 / 720
#     #ym_per_pix = 3 / 200

#     y_eval = np.max(y) * ym_per_pix

#     #left_fit = np.polyfit(y, left_fit, 2)
#     #right_fit = np.polyfit(y, right_fit, 2)

#     left_fit *= xm_per_pix
#     right_fit *= xm_per_pix

#     left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#     right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

#     return left_curverad, right_curverad

def curvature(fit_x, fit_y):

    xm_per_pix = 3.7 / 700
    ym_per_pix = 30 / 720

    y_eval = 456 * ym_per_pix
    # y_eval = np.max(fit_y)

    fit_rw = np.polyfit(fit_y * ym_per_pix, fit_x * xm_per_pix, 2)

    curve_rad = ((1 + (2*fit_rw[0]*y_eval*ym_per_pix + fit_rw[1])**2)**1.5) / np.absolute(2*fit_rw[0])

    return curve_rad

def offset(car_center, lane_center):

    xm_per_pix = 3.7 / 700

    return round((lane_center - car_center) * xm_per_pix, 2)

def draw_lane_lines(image, undist, warped, left_fitx, right_fitx, y, M_inv, car_center, lane_center, left_curve, right_curve):

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    points_left = np.array([np.transpose(np.vstack([left_fitx, y]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y])))])
    points = np.hstack((points_left, points_right))

    cv2.fillPoly(color_warp, np.int_([points]), (0, 255, 0))

    new_warp = cv2.warpPerspective(color_warp, M_inv, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)

    if lane_center - car_center > 0:
        offset_text = "{}m left of center".format(offset(car_center, lane_center))

    elif lane_center - car_center < 0:
        offset_text = "{}m right of center".format(-offset(car_center, lane_center))

    else:
        offset_text = "centered"

    cv2.putText(
        result, 
        "Radius of Curvature: {}m".format(round(min(left_curve, right_curve), 1)), 
        (100, 100), 
        cv2.FONT_HERSHEY_COMPLEX, 
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

    return result

def draw_lines(image, polygon):

    for i in range(len(polygon)):

        cv2.line(image, tuple(polygon[i]), tuple(polygon[(i + 1) % len(polygon)]), (0, 0, 255), thickness = 2)

if __name__ == "__main__":

    #image = cv2.imread("../test_images/test5.jpg")
    image = mpimg.imread("../test_images/test5.jpg")

    src = np.float32([
        [250, 678], # bottom left
        [585, 456], # top left
        [698, 456], # top right
        [1057, 678] # bottom right
    ])

    img_copy = np.copy(image)


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

    undist = cv2.undistort(img_copy, mtx, dist)

    draw_lines(undist, src)

    warped = perspective_transform(image, mtx, dist, src, dest)

    draw_lines(warped, dest)

    # cv2.imshow("undist", undist)
    # cv2.waitKey()

    # cv2.imshow("warped", warped)
    # cv2.waitKey()

    plt.imshow(undist)
    plt.show()
    plt.imshow(warped)
    plt.show()