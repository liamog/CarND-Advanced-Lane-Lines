"""Camera calibration."""
import glob
import os.path
import pickle
import sys

import matplotlib.image as mpimg
import numpy as np

import cv2


class Camera():
    """Camera calibration class."""

    def __init__(self, calibration_image_path):
        """Initializer."""
        # cache file to store camera calibration data.
        self._calibration_filename = "calibration_data.p"

        # Calibration images output for debugging
        self.callibration_chess_boards = []

        # Camera Calibration support
        self._calibration_mtx = None
        self._calibration_dist = None
        self._calibration_rvecs = None
        self._calibration_tvecs = None
        self._init_calibrate_camera(calibration_image_path)

    def process_image(self, img):
        """Process the image."""
        return cv2.undistort(img, self._mtx,
                             self._dist, None, self._mtx)

    def _calibrate_camera_from_image(self,
                                     img,
                                     num_x,
                                     num_y,
                                     objpoints,
                                     imgpoints):
        objp = np.zeros((num_y * num_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (num_x, num_y), corners, ret)
            self.callibration_chess_boards.append(img)
        return shape

    def _init_calibrate_camera(self, calibration_image_path):
        if os.path.isfile(self._calibration_filename):
            with open(self._calibration_filename, 'rb') as calibration_file:
                if sys.version_info[0] < 3:
                    calibration_data = pickle.load(calibration_file)
                else:
                    calibration_data = pickle.load(
                        calibration_file, encoding='latin1')
                self._mtx = calibration_data["mtx"]
                self._dist = calibration_data["dist"]
                self._rvecs = calibration_data["rvecs"]
                self._tvecs = calibration_data["tvecs"]
                return

        num_x = int(9)
        num_y = int(6)

        objpoints = []
        imgpoints = []
        print(calibration_image_path)

        for file in glob.glob(calibration_image_path + '/*.jpg'):
            print(file)
            img = mpimg.imread(file)
            shape = self._calibrate_camera_from_image(
                img, num_x, num_y, objpoints, imgpoints)

        ret = cv2.calibrateCamera(
            objpoints, imgpoints, shape, None, None)
        if ret[0]:
            print("Successfully Calibrated Camera")
            self._mtx = ret[1]
            self._dist = ret[2]
            self._rvecs = ret[3]
            self._tvecs = ret[4]
            calibration_data = {"mtx": self._mtx,
                                "dist": self._dist,
                                "rvecs": self._rvecs,
                                "tvecs": self._tvecs}
            pickle.dump(calibration_data,
                        open(self._calibration_filename, "wb"),
                        protocol=2)
        else:
            print("Failed to Calibrate Camera")
