import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from calibrate_camera import extract_points, camera_cal

# calibrate camera using chessboard images
# obtain camera matrix and distortion coefficients
mtx, dist = extract_points(glob.glob("../camera_cal/*"))

