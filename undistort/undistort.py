#
# Frame undistort processor
#
# @NamoDev, May 2020
#
import os
import cv2
import pickle
import pathlib

class undistort():

    # Init
    def __init__(self, camera_model = "c920", frame_width = 1280, frame_height = 720):

        # Get absolute path to here
        absPath = pathlib.Path(__file__).parent.absolute()

        # Get path to camera data
        camDataPath = str(absPath) + "/cameras/" + camera_model.lower() + "/"

        # Load camera and distortion matrices
        with open(camDataPath + "matrix.pkl", "rb") as f:
            self.cameraMatrix = pickle.load(f)

        with open(camDataPath + "distortion.pkl", "rb") as f:
            self.distortionMatrix = pickle.load(f)

        # Calculate optimal camera matrix
        self.frameWidth = frame_width
        self.frameHeight = frame_height
        self.newCMatrix, self.roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distortionMatrix, (frame_width, frame_height), 1, (frame_width, frame_height))

        # We are go!
        self.ready = True

    # Undistort a frame
    def frame(self, frame):

        if self.ready != True:
            raise RuntimeError("Undistort hasn't finished initializing yet!")

        # We need to have a valid frame to work with
        if frame is not None:
            # See if frame size == what we initialized with
            # if so, we don't need to compute optimal camera matrix again
            frameH, frameW = frame.shape[:2]

            if frameH == self.frameHeight and frameW == self.frameWidth:
                ncm = self.newCMatrix
                roi = self.roi
            else:
                # Oh no, different frame size, let's recompute
                ncm, roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distortionMatrix, (frameW, frameH), 1, (frameW, frameH))
                self.newCMatrix = ncm
                self.roi = roi
                self.frameHeight = frameH
                self.frameWidth = frameW

            # Now we can undistort!
            undistorted = cv2.undistort(frame, self.cameraMatrix, self.distortionMatrix, None, ncm)
            return undistorted

        else:
            # Error: invalid frame (NoneType)
            raise ValueError("Frame cannot be none!")
