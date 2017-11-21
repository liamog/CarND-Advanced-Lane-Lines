"""Lane Line Module. c Liam O'Gorman."""
import glob
import os.path
import pickle
import sys

import matplotlib.image as mpimg
import numpy as np

import cv2
from line import Line


class LaneLines():
    """The Lane Line Class."""

    def _init_perspective_transform_matrices(self):
        src = np.float32([[279.0, 674.0],
                          [531.0, 495.0],
                          [762.5, 495.0],
                          [1041.0, 674.0]])
        dst = np.float32([[279.0, 674.0],
                          [279.0, 495.0],
                          [1041.0, 495.0],
                          [1041.0, 674.0]])
        self._perspective_transform = cv2.getPerspectiveTransform(src, dst)
        self._perspective_inverse = cv2.getPerspectiveTransform(dst, src)

    def _convert_to_gray(self, img):
        # If we have 
        if len(img.shape) > 2:
            channel_count = img.shape[2]  
            if (channel_count == 3):
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _calibrate_camera_from_image(self, img, nx, ny, objpoints, imgpoints):
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
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

        nx = int(9)
        ny = int(6)

        objpoints = []
        imgpoints = []
        print(calibration_image_path)

        for file in glob.glob(calibration_image_path + '/*.jpg'):
            print(file)
            img = mpimg.imread(file)
            shape = self._calibrate_camera_from_image(
                img, nx, ny, objpoints, imgpoints)

        ret = cv2.calibrateCamera(
            objpoints, imgpoints, shape, None, None)
        if(ret[0]):
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
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        abs_sobel = np.absolute(self._sobelx)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        self._grad_x_binary = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        self._grad_x_binary[(scaled_sobel >= self._grad_x_threshold[0]) &
                            (scaled_sobel <= self._grad_x_threshold[1])] = 1

    def _build_grad_y(self):
        # Apply threshold
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        abs_sobel = np.absolute(self._sobely)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        self._grad_x_binary = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        self._grad_x_binary[(scaled_sobel >= self._grad_y_threshold[0]) &
                            (scaled_sobel <= self._grad_y_threshold[1])] = 1

    def _build_grad_mag(self):
        # Calculate the gradient magnitude
        grad_mag = np.sqrt(self._sobelx**2 + self._sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(grad_mag) / 255
        grad_mag = (grad_mag / scale_factor).astype(np.uint8)

        # Create a binary image of ones where threshold is met, zeros otherwise
        self._grad_mag_binary = np.zeros_like(grad_mag)
        self._grad_mag_binary[(grad_mag >= self._mag_threshold[0]) &
                              (grad_mag <= self._mag_threshold[1])] = 1

    def _build_grad_dir(self):
        # Grayscale
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        abs_grad_dir = np.arctan2(np.absolute(self._sobely),
                                  np.absolute(self._sobelx))
        self._grad_dir_binary = np.zeros_like(abs_grad_dir)
        self._grad_dir_binary[(abs_grad_dir >= self._dir_threshold[0]) &
                              (abs_grad_dir <= self._dir_threshold[1])] = 1

    def write_image(self, name, img):
        """Write an image with the correct color space."""
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            if (channel_count == 3):
                cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(name, img)

    def _weighted_img(self, img, initial_img, alpha=0.8, beta=1., epsilon=0.):
        """`img` is the image to overlay on top of initial img.

        `initial_img` should be the image before any processing.
        The result image is computed as follows:

        initial_img * alpha + img * beta + epsilon
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, alpha, img, beta, epsilon)

    def _draw_lines_between_points(self,
                                   shape,
                                   points,
                                   color=[255, 0, 0],
                                   thickness=10):
        """Draw `lines` on `img` with `color` and `thickness`."""
        img = np.zeros(shape, dtype=np.uint8)
        for idx in range(len(points) - 1):
            cv2.line(img, tuple(points[idx]), tuple(
                points[idx + 1]), color, thickness)
        return img

    def _prepare_img(self):
        undist = cv2.undistort(self.source_img, self._mtx,
                               self._dist, None, self._mtx)
        self.warped = cv2.warpPerspective(undist,
                                          self._perspective_transform,
                                          self.img_size,
                                          flags=cv2.INTER_LINEAR)
        # Crop out the bottom of the image where part of the car is visible.
        self.warped = self.warped[0:self.warped.shape[0]:, :]
        self.hls = cv2.cvtColor(self.warped, cv2.COLOR_RGB2HLS)
        self.s_channel = self.hls[:, :, 2]
        self.gray = cv2.cvtColor(self.warped, cv2.COLOR_RGB2GRAY)
        # Changes this to different sources to try different channels etc.

        self.source_channel = self.s_channel
        # Apply each of the thresholding functions
        self._build_gradients_for_source()

        self.binary_warped = np.zeros_like(self.source_channel, np.int8)
        self.binary_warped[
            ((self._grad_x_binary == 1) &
             (self._grad_y_binary == 1)) |
            ((self._grad_mag_binary == 1) &
             (self._grad_dir_binary == 1))] = 1
        # Smooth the lines by merging the last n frames
        self._images.append(self.binary_warped)
        if len(self._images) > 5:
            del self._images[0]

        # merge image with previous images.
        for img in self._images:
            self.binary_warped += img

    def _find_line_full_search(self):
        # mask top of warped image to remove noise

        self.histogram = np.sum(
            self.binary_warped[self.binary_warped.shape[0] // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(
            (self._warped_y_range[1] - self._warped_y_range[0]) / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self._warped_y_range[1] - \
                (window + 1) * window_height
            win_y_high = self._warped_y_range[1] - \
                window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image

            cv2.rectangle(self.lane_find_visualization,
                          (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high),
                          (0, 255, 0),
                          2)
            cv2.rectangle(self.lane_find_visualization,
                          (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high),
                          (0, 255, 0),
                          2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels,
            # recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        self.left_lane_line.fit_line(leftx, lefty)

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        self.right_lane_line.fit_line(rightx, righty)
        return self._visualize_lanes()

    def _find_lines_from_best_fit(self):
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        self.left_lane_line.fit_line_from_current(nonzerox, nonzeroy, margin)
        self.right_lane_line.fit_line_from_current(nonzerox, nonzeroy, margin)
        return self._visualize_lanes()

    def _visualize_lanes(self):
        """Visualize the lane as an overlay image fill between the lines."""
        self.lane_find_visualization[self.left_lane_line.ally,
                                     self.left_lane_line.allx] = [255, 0, 0]
        self.lane_find_visualization[self.right_lane_line.ally,
                                     self.right_lane_line.allx] = [0, 0, 255]

        # visualize the curve.
        self.left_lane_line.visualize_lane_current_fit(
            self.lane_find_visualization)
        self.right_lane_line.visualize_lane_current_fit(
            self.lane_find_visualization)

        # visualize the lane as a poly fill.
        color_warp = self.left_lane_line.poly_fill_with_other_lane(
            self.right_lane_line, np.shape(self.binary_warped))

        # Warp the blank back to original image space
        # using inverse perspective matrix
        lane_poly = cv2.warpPerspective(
            color_warp, self._perspective_inverse, (self.img_size))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (255, 255, 255)
        line_type = 3

        left_text_pos = (20, 100)
        if (self.left_lane_line.radius_of_curvature > 10000):
            left_text = 'Left CurRad=Straight'
        else:
            left_text = 'Left CurRad={:3.4f}'.format(
                self.left_lane_line.radius_of_curvature)
        left_text += ': lane_dist={:3.4f}'.format(
            self.left_lane_line.line_base_pos)
        cv2.putText(lane_poly,
                    left_text,
                    left_text_pos,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        right_text_pos = (20, 160)
        # If greater than 10000m (10k) assume straight
        if (self.right_lane_line.radius_of_curvature > 10000):
            right_text = 'Right CurRad=Straight'
        else:
            right_text = 'Right CurRad={:3.4f}'.format(
                self.right_lane_line.radius_of_curvature)
        right_text += (': lane_dist={:3.4f}'.format(
            self.right_lane_line.line_base_pos))

        cv2.putText(lane_poly,
                    right_text,
                    right_text_pos,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        signed_dist_from_center = self.right_lane_line.line_base_pos + \
            self.left_lane_line.line_base_pos
        if (signed_dist_from_center > 0):
            side = "Right"
        else:
            side = "Left"

        center_text = 'Car is {:3.4f}m ({}) from lane center'.format(
            signed_dist_from_center, side)
        center_text_pos = (20, 220)

        cv2.putText(lane_poly,
                    center_text,
                    center_text_pos,
                    font,
                    font_scale,
                    font_color,
                    line_type)
        cv2.line(lane_poly,
                 (int(self.img_width / 2), 0),
                 (int(self.img_width / 2), int(self.img_height)),
                 (255, 0, 0), 3)

        # Combine the result with the original image
        return cv2.addWeighted(self.source_img, 1, lane_poly, 0.3, 0)

    def set_thresholds(self,
                       grad_x_threshold,
                       grad_y_threshold,
                       mag_threshold,
                       dir_threshold):
        self._grad_x_threshold = grad_x_threshold
        self._grad_y_threshold = grad_y_threshold
        self._mag_threshold = mag_threshold
        self._dir_threshold = dir_threshold

    def save_thresholds(self):
        '''Save threshold configuration.'''
        config = {
            "grad_x_threshold": self._grad_x_threshold,
            "grad_y_threshold": self._grad_y_threshold,
            "mag_threshold": self._mag_threshold,
            "dir_threshold": self._dir_threshold,
            }
        pickle.dump(config, open(self._thresholds_file_name, "wb"))

    def load_thresholds(self):
        '''Loads previously saved threshold configuration.'''
        if os.path.exists(self._thresholds_file_name):
            with open(self._thresholds_file_name, 'rb') as config_file:
                config = pickle.load(config_file, encoding='latin1')
                self._grad_x_threshold = config["grad_x_threshold"]
                self._grad_y_threshold = config["grad_y_threshold"]
                self._mag_threshold = config["mag_threshold"]
                self._dir_threshold = config["dir_threshold"]
    
    def clear(self):
        self.gray = None
        self.source_channel = None

        self.warped = None
        self.binary_warped = None
        self.lane_find_visualization = None
        self.histogram = None

    def process_image(self, img):
        """Process image and return image with Lane Line drawn."""
        self.clear()
        self.source_img = img
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        self.img_size = (self.img_width, self.img_height)

        self._prepare_img()

        single_channel = self.binary_warped
        self.lane_find_visualization = np.dstack(
            (single_channel * 255,
             single_channel * 255,
             single_channel * 255)).astype(np.uint8)

        # mask top of warped image to remove noise
        self.binary_warped[0:self._warped_y_range[0]:1, ::] = 0
        # mask bottom of warped image to remove noise
        self.binary_warped[self._warped_y_range[1]
            :self.binary_warped.shape[0] - 1:1, ::] = 0
        if (not self.right_lane_line.detected or
                not self.left_lane_line.detected):
            return self._find_line_full_search()
        else:
            return self._find_lines_from_best_fit()


    def __init__(self, calibration_image_path):
        """Initializer."""
        # Configuration file to store threshold settings.
        self._thresholds_file_name = "thresholds.p"
        # cache file to store camera calibration data.
        self._calibration_filename = "calibration_data.p"

        # Set to the image currently being processed
        self._img = None
        self._images = []
        # perspective transform matrix
        self._perspective_transform = None

        # perspective transform inverse matrix
        self._perspective_inverse = None

        # threshold values for the gradient in the X direction
        self._grad_x_threshold = (46, 82)
        self._sobel_x_kernel = 9
        # threshold values for the gradient in the Y direction
        self._grad_y_threshold = (14, 166)
        self._sobel_y_kernel = 9

        # threshold values for the gradient magnitude
        self._mag_threshold = (17, 255)
        # threshold values for the gradient direction in rads
        self._dir_threshold = (0.0, 0.47123889803846897)

        # range along y that we search for lane pixels.
        # Use to ignore noise in the distance and the car bonnet
        self._warped_y_range = (100, 650)

        self._grad_x_binary = None
        self._grad_y_binary = None
        self._grad_mag_binary = None
        self._grad_dir_binary = None

        # Camera Calibration support
        self._calibration_mtx = None
        self._calibration_dist = None
        self._calibration_rvecs = None
        self._calibration_tvecs = None

        # Calibration images output
        self.callibration_chess_boards = []

        # Image processing
        self.gray = None
        self.source_channel = None

        self.warped = None
        self.binary_warped = None
        self.lane_find_visualization = None
        self.histogram = None

        # Detected Lanes
        self.right_lane_line = Line((720, 1280))
        self.left_lane_line = Line((720,1280))

        # Initialize 
        self._init_perspective_transform_matrices()
        self._init_calibrate_camera(calibration_image_path)
        self.load_thresholds()
