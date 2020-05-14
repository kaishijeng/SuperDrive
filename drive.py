#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

#
# SuperDrive
# a live processing capable, clean(-ish) implementation of lane &
# path detection based on comma.ai's SuperCombo neural network model
#

# ============================================================================ #
# Parse arguments
import os
import warnings
import argparse

apr = argparse.ArgumentParser(description = "Predicts lane line and vehicle path using the SuperCombo neural network!")
apr.add_argument("--input", type=str, dest="inputFile", help="Input capture device or video file", required=True)
apr.add_argument("--disable-gpu", dest="disableGPU", action="store_true", help="Disables the use of GPU for inferencing")
apr.add_argument("--disable-warnings", dest="disableWarnings", action="store_true", help="Disables console warning messages")
apr.add_argument("--show-opencv-window", dest="showOpenCVVisualization", action="store_true", help="Shows OpenCV frame visualization")
apr.add_argument("--show-matplotlib-window", dest="showMatPlotLibVisualization", action="store_true", help="Shows Matplotlib output visualization (WARNING: TANKS PERFORMANCE)")

args = apr.parse_args()

# Where are we reading from?
CAMERA_DEVICE = str(args.inputFile)

# Do we want to disable GPU?
if args.disableGPU == True:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Do we want to disable warning messages?
if args.disableWarnings == True:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    warnings.filterwarnings("ignore")
# ============================================================================ #

import cv2
import sys
import time
import pathlib
import matplotlib
import numpy as np
import tensorflow as tf
from parser import parser
import savitzkygolay as sg
import matplotlib.pyplot as plt
from undistort.undistort import undistort
from timeit import default_timer as timer

# OpenPilot transformations (needed to get the model to output correct results)
from common.transformations.model import medmodel_intrinsics
from common.transformations.camera import transform_img, eon_intrinsics

# Are we running TF on GPU?
if tf.test.is_gpu_available() == True:
    isGPU = True
    tfDevice = "GPU"
else:
    isGPU = False
    tfDevice = "CPU"

# Initialize undistort
undist = undistort(frame_width=560, frame_height=315)

# Initialize OpenCV capture and set basic parameters
cap = cv2.VideoCapture(CAMERA_DEVICE)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# Load Keras model for lane detection
#
# path = [y_pos of path plan along x=range(0,192) |
#        std of y_pos of path plan along x=range(0,192) |
#        how many meters it can see]
# 12 * 128 * 256 is 2 consecutive imgs in YUV space of size 256 * 512
lanedetector = tf.keras.models.load_model(str(pathlib.Path(__file__).parent.absolute()) + "/supercombo.keras")

# We need a place to keep two separate consecutive image frames
# since that's what SuperCombo uses
fr0 = np.zeros((384, 512), dtype=np.uint8)
fr1 = np.zeros((384, 512), dtype=np.uint8)

# SuperCombo requires a feedback of state after each prediction
# (to improve accuracy?) so we'll allocate space for that
state = np.zeros((1, 512))

# Additional inputs to the steering model
#
# "Those actions are already there, we call it desire.
#  It's how the lane changes work" - @Willem from Comma
#
# Note: not implemented in SuperDrive (yet)
desire = np.zeros((1, 8))

# We want to keep track of our FPS rate, so here's
# some variables to do that
fpsActual = 0;
fpsCounter = 0;
fpsTimestamp = 0;

# Main loop here
while True:

    # Get frame start time
    t_frameStart = timer()

    # FPS counter logic
    fpsCounter += 1
    if int(time.time()) > fpsTimestamp:
        fpsActual = fpsCounter
        fpsTimestamp = int(time.time())
        fpsCounter = 0

    # Read frame
    (ret, frame) = cap.read()

    # Resize incoming frame to smaller size (to save resource in undistortion)
    frame = cv2.resize(frame, (560, 315))

    # Undistort incoming frame
    # This is standard OpenCV undistortion using a calibration matrix.
    # In this case, a Logitech C920 is used (default for undistortion helper).
    # Just perform chessboard calibration to get the matrices!
    frame = undist.frame(frame)

    # Crop the edges out and try to get to (512,256), since that's what
    # the SuperCombo model uses. Note that this is skewed a bit more
    # to the sky, since my camera can "see" the hood and that probably won't
    # help us in the task of lane detection, so we crop that out
    frame = frame[14:270, 24:536]

    # Then we want to convert this to YUV
    frameYUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

    # Use Comma's transformation to get our frame into a format that SuperCombo likes
    frameYUV = transform_img(frameYUV, from_intr=eon_intrinsics,
                             to_intr=medmodel_intrinsics, yuv=True,
                             output_size=(512, 256)).astype(np.float32) \
        / 128.0 - 1.0

    # We want to push our image in fr1 to fr0, and replace fr1 with
    # the current frame (to feed into the network)
    fr0 = fr1
    fr1 = frameYUV

    # SuperCombo input shape is (12, 128, 256): two consecutive images
    # in YUV space. We concatenate fr0 and fr1 together to get to that
    networkInput = np.concatenate((fr0, fr1))

    # We then want to reshape this into the shape the network requires
    networkInput = networkInput.reshape((1, 12, 128, 256))

    # Build actual input combination
    input = [networkInput, desire, state]

    # Then, we can run the prediction!
    # TODO: this is somehow very slow(?)
    networkOutput = lanedetector.predict(input)

    # Parse output and refeed state
    parsed = parser(networkOutput)
    state = networkOutput[-1]

    # Now we have all the points!
    # These correspond to points with x = <data in here>, y = range from
    # 0 to 192 (output of model)
    leftLanePoints = parsed["lll"][0]
    rightLanePoints = parsed["rll"][0]
    pathPoints = parsed["path"][0]

    # We may also want to smooth this out
    leftLanePoints = sg.savitzky_golay(leftLanePoints, 51, 3)
    rightLanePoints = sg.savitzky_golay(rightLanePoints, 51, 3)
    pathPoints = sg.savitzky_golay(pathPoints, 51, 3)

    # Compute position on current lane
    currentPredictedPos = (-1) * pathPoints[0]

    # Compute running time
    p_totalFrameTime = round((timer() - t_frameStart) * 1000, 2)

    print("Frame processed on " + tfDevice + " \t" + str(p_totalFrameTime) + " ms\t" + str(fpsActual) + " fps")

    # Output (enlarged) frame with text overlay
    if args.showOpenCVVisualization == True:
        canvas = frame.copy()
        canvas = cv2.resize(canvas, ((1024, 512)))
        cv2.putText(canvas, "Vision processing time: " + str(p_totalFrameTime) + " ms (" + str(fpsActual) + " fps)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(canvas, "Device: " + tfDevice, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(canvas, "Position: " + str(round(currentPredictedPos, 3)) + " m off centerline", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        cv2.imshow("Frame", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Draw plots
    if args.showMatPlotLibVisualization == True:
        plt.clf()
        plt.title("Lane/Path Prediction")

        # Interactive mode on to not steal output
        plt.ion()

        # Left/right lane and predicted path
        plt.plot(leftLanePoints, range(0, 192), "b-", linewidth=1)
        plt.plot(rightLanePoints, range(0, 192), "r-", linewidth=1)
        plt.plot(pathPoints, range(0, 192), "g-", linewidth=1)

        # Constrain X-axis so that our plot won't dance around
        plt.xlim(-5, 5)

        # Invert X-axis
        plt.gca().invert_xaxis()
        plt.pause(0.001)
