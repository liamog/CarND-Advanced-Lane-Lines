"""Lane Line Module. c Liam O'Gorman."""
import os.path
import pickle
import sys

import numpy as np
import scipy
import scipy.misc
from scipy.ndimage.interpolation import zoom

import cv2
from binary_image import BinaryImage, SourceType
from camera import Camera
from config import Config
from line import Line
from perspective import Perspective


class LaneLines():
    """The Lane Line Class."""

    def __init__(self, calibration_image_path):
        """Initializer."""
        # Set to the image currently being processed
        self._img = None
        self._images = []
        self.img_width = 0
        self.img_height = 0
        self.img_size = (0, 0)
        # Final image with lane lines superimposed.
        self.processed_image = None

        # range along y that we search for lane pixels.
        # Use to ignore noise in the distance and the car bonnet
        self._warped_y_range = (0, 700)

        # Image processing
        self._sobelx = None
        self._sobely = None

        self.current_binary_warped = None
        self.smooth_binary_warped = None
        self.lane_find_visualization = None
        self.histogram = None
        self.diagnostics_image = None

        # Detected Lanes
        self.right_lane_line = Line((720, 1280))
        self.left_lane_line = Line((720, 1280))

        # Counter for consecutive rejected lanes.
        self._rejected = 0

        self.camera = Camera(calibration_image_path)
        self.perspective = Perspective()
        self.diagnostics = Diagnostics()
        # Initialize
        self.binary_image_y_channel = BinaryImage(
            SourceType.Y_CHANNEL)
        self.binary_image_v_channel = BinaryImage(
            SourceType.V_CHANNEL)

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

    def _find_line_full_search(self):
        # mask top of warped image to remove noise
        self.diagnostics.sliding_window = True
        self.histogram = np.sum(
            self.smooth_binary_warped[
                self.smooth_binary_warped.shape[0] // 2:, :
            ], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = Config.SEARCH_WINDOWS
        # Set height of windows
        window_height = np.int(
            (self._warped_y_range[1] - self._warped_y_range[0]) / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.smooth_binary_warped.nonzero()
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
        fit_left_success = self.left_lane_line.fit_line(leftx, lefty)

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        fit_right_success = self.right_lane_line.fit_line(rightx, righty)

        if fit_left_success and fit_right_success:
            # Sanity check the current fit, if looks reasonable then
            # add to the smooth fit.
            probable_lane, mean, sigma = \
                self.left_lane_line.probable_lane_detected(
                    self.right_lane_line)
            self.diagnostics.average_lane_width = mean
            self.diagnostics.lane_width_stddev = sigma
            self.diagnostics.rejected = not probable_lane
            if probable_lane:
                self._rejected = 0
                self.diagnostics_image
                self.right_lane_line.add_current_fit_to_smooth()
                self.left_lane_line.add_current_fit_to_smooth()

        return self._visualize_lanes()

    def _find_lines_from_smooth(self):
        self.diagnostics.fast_fit = True
        nonzero = self.smooth_binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        self.left_lane_line.fit_line_from_smooth(
            nonzerox, nonzeroy, Config.MARGIN, self.lane_find_visualization)
        self.right_lane_line.fit_line_from_smooth(
            nonzerox, nonzeroy, Config.MARGIN, self.lane_find_visualization)
        probable_lane, mean, sigma = \
            self.left_lane_line.probable_lane_detected(self.right_lane_line)
        self.diagnostics.average_lane_width = mean
        self.diagnostics.lane_width_stddev = sigma
        self.diagnostics.rejected = not probable_lane

        if probable_lane:
            self._rejected = 0
            self.right_lane_line.add_current_fit_to_smooth()
            self.left_lane_line.add_current_fit_to_smooth()
        else:
            self._rejected += 1
            if self._rejected > Config.MAX_REJECTED:
                self._rejected = 0
                return self._find_line_full_search()

        return self._visualize_lanes()

    def _visualize_lanes(self):
        """Visualize the lane as an overlay image fill between the lines."""
        self.lane_find_visualization[self.left_lane_line.all_y,
                                     self.left_lane_line.all_x] = [255, 0, 0]
        self.lane_find_visualization[self.right_lane_line.all_y,
                                     self.right_lane_line.all_x] = [0, 0, 255]

        if self.right_lane_line.valid:
            right_curvature = self.right_lane_line.smooth_radius_of_curvature
            right_line_pos = self.right_lane_line.smooth_line_pos
        else:
            right_curvature = self.right_lane_line.current_radius_of_curvature
            right_line_pos = self.right_lane_line.current_line_pos

        if self.left_lane_line.valid:
            left_curvature = self.left_lane_line.smooth_radius_of_curvature
            left_line_pos = self.left_lane_line.smooth_line_pos
        else:
            left_curvature = self.left_lane_line.current_radius_of_curvature
            left_line_pos = self.left_lane_line.current_line_pos
        
        self.diagnostics.left_curvature = left_curvature
        self.diagnostics.right_curvature = right_curvature

        # visualize the current fit curve.left_curvature
        self.left_lane_line.visualize_lane_current_fit(
            self.lane_find_visualization, color=[255, 255, 0])
        self.right_lane_line.visualize_lane_current_fit(
            self.lane_find_visualization, color=[255, 255, 0])
        self.left_lane_line.visualize_lane_smooth_fit(
            self.lane_find_visualization, color=[255, 255, 255])
        self.right_lane_line.visualize_lane_smooth_fit(
            self.lane_find_visualization, color=[255, 255, 255])

        # visualize the lane as a poly fill.
        color_warp = self.left_lane_line.poly_fill_with_other_lane(
            self.right_lane_line)
        # Warp the blank back to original image space
        # using inverse perspective matrix
        lane_poly = self.perspective.process_image_inverse(color_warp)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_white = (255, 255, 255)
        line_type = 3
        curvature = (right_curvature + left_curvature) / 2
        curve_text_pos = (20, 160)
        curve_text = 'Curve Radius={:3.4f}'.format(curvature)

        cv2.putText(lane_poly,
                    curve_text,
                    curve_text_pos,
                    font,
                    font_scale,
                    font_white,
                    line_type)

        center_of_lane = ((right_line_pos - left_line_pos) / 2) + left_line_pos

        signed_dist_from_center = self.left_lane_line.camera_pos -\
            center_of_lane
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
                    font_white,
                    line_type)
        cv2.line(lane_poly,
                 (int(self.img_width / 2), 0),
                 (int(self.img_width / 2), int(self.img_height)),
                 (255, 0, 0), 3)
        self.diagnostics.write_to_image(self.diagnostics_image)


        # Combine the result with the original image
        return cv2.addWeighted(self.source_img, 1, lane_poly, 0.3, 0)

  
    def clear_images(self):
        """Clear the images."""
        self.smooth_binary_warped = None
        self.lane_find_visualization = None
        self.diagnostics_image = None
        self.histogram = None

    def process_image_with_diagnostics(self, img):
        """Process the image and append diagnostics to image."""
        self.process_image(img)
        size = np.shape(img)
        size = (int(size[0] / 2), int(size[1] / 2))

        rc = scipy.misc.imresize(self.binary_image_v_channel.source_channel, size)
        rc = np.dstack((rc, rc, rc))
        rcb = scipy.misc.imresize(
            self.binary_image_v_channel.processed_image, size)
        rcb = np.dstack((rcb, rcb, rcb))

        sc = scipy.misc.imresize(self.binary_image_y_channel.source_channel, size)
        sc = np.dstack((sc, sc, sc))
        scb = scipy.misc.imresize(
            self.binary_image_y_channel.processed_image, size)
        scb = np.dstack((scb, scb, scb))

        lfv = scipy.misc.imresize(self.lane_find_visualization, size)

        sbw = scipy.misc.imresize(self.smooth_binary_warped * 255, size)
        sbw = np.dstack((sbw, sbw, sbw))
        cbw = scipy.misc.imresize(self.current_binary_warped * 255, size)
        cbw = np.dstack((cbw, cbw, cbw))

        di = scipy.misc.imresize(self.diagnostics_image, size)

        diags_1_r1 = np.hstack((rc, sc))
        diags_1_r2 = np.hstack((rcb, scb))
        diags_1 = np.vstack((diags_1_r1, diags_1_r2))

        diags_2_r1 = np.hstack((cbw, lfv))
        diags_2_r2 = np.hstack((sbw, di))
        diags_2 = np.vstack((diags_2_r1, diags_2_r2))

        final_plus_diags = np.hstack((self.processed_image, diags_1, diags_2))
        return final_plus_diags


    def process_image(self, img):
        """Process image and return image with Lane Line drawn."""
        self.diagnostics.frame_number += 1
        self.clear_images()
        self.source_img = img
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        self.img_size = (self.img_width, self.img_height)

        # Pipeline
        # 1. Undistort
        undistored = self.camera.process_image(img)
        # 2. Warp
        warped = self.perspective.process_image(undistored)
        # 3. Red channel binary image
        r_binary = self.binary_image_v_channel.process_image(warped)
        # 4. Saturation channel binary image
        s_binary = self.binary_image_y_channel.process_image(warped)

        # Merge r_binary and s_binary
        self.current_binary_warped = np.zeros_like(r_binary, np.int8)
        self.current_binary_warped[(r_binary == 1) | (s_binary == 1)] = 1

        # Smooth the lines by merging the last n frames
        self._images.append(self.current_binary_warped)
        num_images = len(self._images)

        if num_images > Config.SMOOTH_OVER_N_FRAMES:
            # remove the oldest image
            del self._images[0]
        num_images = len(self._images)

        # merge binary image with previous images.
        self.smooth_binary_warped = np.zeros_like(self.current_binary_warped,
                                                  np.int8)
        for binary_img in self._images:
            self.smooth_binary_warped[(self.smooth_binary_warped == 1) |
                                      (binary_img == 1)] = 1

        single_channel = self.smooth_binary_warped * 255
        self.lane_find_visualization = np.dstack(
            (single_channel,
             single_channel,
             single_channel)).astype(np.uint8)
        self.diagnostics_image = np.zeros_like(
            self.lane_find_visualization)

        # mask top of warped image to remove noise
        self.smooth_binary_warped[0:self._warped_y_range[0]:1, ::] = 0
        # mask bottom of warped image to remove noise
        self.smooth_binary_warped[self._warped_y_range[1]:
                                  self.smooth_binary_warped.shape[0] - 1:
                                  1, ::] = 0
        if (not self.right_lane_line.valid or
                not self.left_lane_line.valid):
            self.processed_image = self._find_line_full_search()
            return self.processed_image
        else:
            self.processed_image = self._find_lines_from_smooth()
            return self.processed_image


class Diagnostics():
    """Diagnostic helper class."""

    def __init__(self):
        """Initializer."""
        self.frame_number = 0
        self.average_lane_width = None
        self.lane_width_stddev = None
        self.rejected = None
        self.sliding_window = None
        self.fast_fit = None
        self.left_curvature = None
        self.right_curvature = None

 
    def write_to_image(self, img):
        """Write diagnostic info to img."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_white = (255, 255, 255)
        font_red = (255, 0, 0)
        line_type = 3

        line_height = 50

        text_pos = (20, 100)
        text = 'FrameNumber={}'.format(
            self.frame_number)
        cv2.putText(img,
                    text,
                    text_pos,
                    font,
                    font_scale,
                    font_white,
                    line_type)

        text_pos = (text_pos[0], text_pos[1] + line_height)
        text = 'Average Lane Width={:3.4f}'.format(
            self.average_lane_width)
        cv2.putText(img,
                    text,
                    text_pos,
                    font,
                    font_scale,
                    font_white,
                    line_type)

        text_pos = (text_pos[0], text_pos[1] + line_height)
        text = 'Lane Width Std Dev={:3.4f}'.format(
            self.lane_width_stddev)
        cv2.putText(img,
                    text,
                    text_pos,
                    font,
                    font_scale,
                    font_white,
                    line_type)

        text_pos = (text_pos[0], text_pos[1] + line_height)
        text = 'Rejected={}'.format(
            self.rejected)
        if self.rejected:
            color = font_red
        else:
            color = font_white
        cv2.putText(img,
                    text,
                    text_pos,
                    font,
                    font_scale,
                    color,
                    line_type)

        text_pos = (text_pos[0], text_pos[1] + line_height)
        text = 'Sliding Window={}'.format(
            self.sliding_window)
        cv2.putText(img,
                    text,
                    text_pos,
                    font,
                    font_scale,
                    font_white,
                    line_type)
        text_pos = (text_pos[0], text_pos[1] + line_height)
        text = 'Fast Fit={}'.format(
            self.fast_fit)
        cv2.putText(img,
                    text,
                    text_pos,
                    font,
                    font_scale,
                    font_white,
                    line_type)
        text_pos = (text_pos[0], text_pos[1] + line_height)
        text = 'Left Curvature={:3.4f}, right curvature{:3.4f}'.format(
            self.left_curvature, self.right_curvature)
        cv2.putText(img,
                    text,
                    text_pos,
                    font,
                    font_scale,
                    font_white,
                    line_type)
