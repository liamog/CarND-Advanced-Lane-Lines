import os.path
import pickle
import sys
from enum import Enum

import numpy as np

import cv2


class SourceType(Enum):
    S_CHANNEL = 1
    R_CHANNEL = 2
    GRAY = 3
 
class BinaryImage():
    def __init__(self, calibration_image_path, thresholds_file, source_type):
        # cache file to store camera calibration data.
        self._calibration_filename = "calibration_data.p"

        # image undistortion variables
        self._mtx = None
        self._dist = None

        # perspective transform matrix
        self.perspective_transform = None

        # perspective transform inverse matrix
        self.perspective_inverse = None

        # threshold values for the gradient in the X direction
        self.grad_x_threshold = (0, 0)
        self._sobel_x_kernel = 9
        # threshold values for the gradient in the Y direction
        self.grad_y_threshold = (0, 0)
        self._sobel_y_kernel = 9

        # threshold values for the gradient magnitude
        self.mag_threshold = (0, 0)
        # threshold values for the gradient direction in rads
        self.dir_threshold = (0.0, 0.0)
        # binary image after processing.
        self.binary_warped = None

        self._init_perspective_transform_matrices()
        self._init_calibrate_camera(calibration_image_path)
        self.load_thresholds(thresholds_file)
        self._source_type = source_type

    def load_thresholds(self, thresholds_file):
        '''Loads previously saved threshold configuration.'''
        if os.path.exists(thresholds_file):
            with open(thresholds_file, 'rb') as config_file:
                if sys.version_info[0] < 3:
                    config = pickle.load(config_file)
                else:
                    config = pickle.load(config_file, encoding='latin1')

                self.grad_x_threshold = config["grad_x_threshold"]
                self.grad_y_threshold = config["grad_y_threshold"]
                self.mag_threshold = config["mag_threshold"]
                self.dir_threshold = config["dir_threshold"]

    def _init_perspective_transform_matrices(self):
        src = np.float32([[230.0, 700.0], [531.0, 495.0],
                          [762.5, 495.0], [1080.0, 700.0]])
        dst = np.float32([[230.0, 700.0], [230.0, 495.0], [
            1080.0, 495.0], [1080.0, 700.0]])
        self.perspective_transform = cv2.getPerspectiveTransform(src, dst)
        self.perspective_inverse = cv2.getPerspectiveTransform(dst, src)

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
        if (os.path.isfile(self._calibration_filename)):
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

    def _build_gradients_for_source(self):
        self._sobelx = cv2.Sobel(self.source_channel, cv2.CV_64F,
                                 1, 0, ksize=self._sobel_x_kernel)
        self._sobely = cv2.Sobel(self.source_channel, cv2.CV_64F,
                                 0, 1, ksize=self._sobel_x_kernel)
        # build X gradient
        self._build_grad_x()
        self._build_grad_y()
        self._build_grad_dir()
        self._build_grad_mag()

    def _build_grad_x(self):
        # Apply threshold
        # Apply x gradient with the OpenCV Sobel() function
        # and take the absolute value
        abs_sobel = np.absolute(self._sobelx)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        self._grad_x_binary = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        self._grad_x_binary[(scaled_sobel >= self.grad_x_threshold[0]) &
                            (scaled_sobel <= self.grad_x_threshold[1])] = 1

    def _build_grad_y(self):
        # Apply threshold
        # Apply y gradient with the OpenCV Sobel() function
        # and take the absolute value
        abs_sobel = np.absolute(self._sobely)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        self._grad_y_binary = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        self._grad_y_binary[(scaled_sobel >= self.grad_y_threshold[0]) &
                            (scaled_sobel <= self.grad_y_threshold[1])] = 1

    def _build_grad_mag(self):
        # Calculate the gradient magnitude
        grad_mag = np.sqrt(self._sobelx**2 + self._sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(grad_mag) / 255
        grad_mag = (grad_mag / scale_factor).astype(np.uint8)

        # Create a binary image of ones where threshold is met, zeros otherwise
        self._grad_mag_binary = np.zeros_like(grad_mag)
        self._grad_mag_binary[(grad_mag >= self.mag_threshold[0]) &
                              (grad_mag <= self.mag_threshold[1])] = 1

    def _build_grad_dir(self):
        # Grayscale
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        abs_grad_dir = np.arctan2(np.absolute(self._sobely),
                                  np.absolute(self._sobelx))
        self._grad_dir_binary = np.zeros_like(abs_grad_dir)
        self._grad_dir_binary[(abs_grad_dir >= self.dir_threshold[0]) &
                              (abs_grad_dir <= self.dir_threshold[1])] = 1

    def prepare_img(self, img):
        img_size = (img.shape[1], img.shape[0])
        undist = cv2.undistort(img, self._mtx,
                               self._dist, None, self._mtx)
        self.warped = cv2.warpPerspective(undist,
                                          self.perspective_transform,
                                          img_size,
                                          flags=cv2.INTER_LINEAR)
                                          
        # Crop out the bottom of the image where part of the car is visible.
        self.warped = self.warped[0:self.warped.shape[0]:, :]

        # Set specified source channel
        if self._source_type == SourceType.R_CHANNEL:
            r_channel = self.warped[:, :, 0]
            self.source_channel = r_channel
        elif self._source_type == SourceType.S_CHANNEL:
            self.hls = cv2.cvtColor(self.warped, cv2.COLOR_RGB2HLS)
            s_channel = self.hls[:, :, 2]
            self.source_channel = s_channel
        elif self._source_type == SourceType.GRAY:
            self.source_channel = cv2.cvtColor(self.warped, cv2.COLOR_RGB2GRAY)
        # Apply each of the thresholding functions
        self._build_gradients_for_source()

        self.binary_warped = np.zeros_like(self.source_channel, np.int8)
        self.binary_warped[
            ((self._grad_x_binary == 1) &
             (self._grad_y_binary == 1)) |
            ((self._grad_mag_binary == 1) &
             (self._grad_dir_binary == 1))] = 1

        return self.binary_warped
