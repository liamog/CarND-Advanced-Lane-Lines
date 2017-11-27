"""Binary Image module."""
import os.path
import pickle
import sys
from enum import Enum

import numpy as np

import cv2


class SourceType(Enum):
    """Type of color conversion to use."""

    S_CHANNEL = 1
    R_CHANNEL = 2
    GRAY = 3


class BinaryImage():
    """
    BinaryImage class.

    Manages color conversion and binary thresholding for the input image.
    """

    def __init__(self, calibration_image_path, thresholds_file, source_type):
        """Initializer."""
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

        # intermediate binary images
        self._grad_x_binary = None
        self._grad_y_binary = None
        self._grad_mag_binary = None
        self._grad_dir_binary = None

        # binary image after processing.
        self.processed_image = None

        self.load_thresholds(thresholds_file)
        self._source_type = source_type

    def load_thresholds(self, thresholds_file):
        """Load previously saved threshold configuration."""
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

    def process_image(self, img):
        """Process the image returning binary image with thresholds applied."""
        incoming_image = np.copy(img)
        # Crop out the bottom of the image where part of the car is visible.
        incoming_image = incoming_image[0:incoming_image.shape[0]:, :]

        # Set specified source channel
        if self._source_type == SourceType.R_CHANNEL:
            r_channel = incoming_image[:, :, 0]
            self.source_channel = r_channel
        elif self._source_type == SourceType.S_CHANNEL:
            self.hls = cv2.cvtColor(incoming_image, cv2.COLOR_RGB2HLS)
            s_channel = self.hls[:, :, 2]
            self.source_channel = s_channel
        elif self._source_type == SourceType.GRAY:
            self.source_channel = cv2.cvtColor(
                incoming_image, cv2.COLOR_RGB2GRAY)
        # Apply each of the thresholding functions
        self._build_gradients_for_source()

        self.processed_image = np.zeros_like(self.source_channel, np.int8)
        self.processed_image[
            ((self._grad_x_binary == 1) &
             (self._grad_y_binary == 1)) |
            ((self._grad_mag_binary == 1) &
             (self._grad_dir_binary == 1))] = 1

        return self.processed_image
