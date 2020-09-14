## Advanced Lane Finding

This program is part of a project in the Udacity SDC nanodegree and part of my research at UCF. The goal of the program is to be a robust pipeline for detecting lane lines, calculating vehicle offset from the center of the lane, determining the road's curvature, and eventually predicting the steering angle of the vehicle. Below are the steps used to process the input video before it is finally returned (in a separate video file), complete with drawn lane markings and other useful information.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Required packages

To run the code, you'll need the following packages (with a Python3 installation):

* numpy
* cv2 (opencv-python)
* matplotlib
* moviepy (you may need to install ffmpeg as well if you do not have it on your system)

### Usage
I've segmented this project into two mediums of usage: a Jupyter notebook and standalone Python code (both available in `src/`). To run the Jupyter notebook, make sure you have Jupyter notebook installed and simply run the following:

```
cd src/ && jupyter notebook
```

From the browser window, open `lf-pipeline.ipynb` and execute each cell from the top using `Shift + Enter` to see the pipeline in action.

The Jupyter notebook is recommended since it has more up-to-date code, and is easier to visualize each step of the pipeline. Converting the updated pipeline to compartmentalized Python files is a work in progress. However, if you wish, you can still run the standalone Python version by installing the required packages and executing `python main.py` in the `src/` folder.