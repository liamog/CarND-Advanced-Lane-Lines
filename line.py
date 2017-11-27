"""line module."""
import sys

import numpy as np
import cv2

from config import Config

class Line():
    """Line class captures line as a polynomial."""


    def __init__(self, shape):
        """Line class Initializer."""

        # polynomial coefficients smoothed over the last n iterations
        self.smooth_fit = None
        # radius of curvature from the smooth fit.
        self.smooth_radius_of_curvature = 0.0

        # distance in meters of vehicle center from the smooth line
        self.smooth_line_pos = -1.0

        # samples used to calculate the smooth fit.
        self.smooth_fit_samples = np.empty((0,3), dtype='float')

        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # current radius of curvature of the line in some units
        self.current_radius_of_curvature = 0.0

        # distance in meters of vehicle center from the current line
        self.current_line_pos = -1.0

        # True if the smooth line is currently considered valid.
        self.valid = False

        # x values for detected line pixels
        self.all_x = None
        # y values for detected line pixels
        self.all_y = None

        # shape to use to calculate curvature and lane center. Should
        # be the same as the image size being used to detect lanes.
        self.shape = shape
        self.camera_pos = (self.shape[1] / 2) * Config.XM_PER_PIX

    def reset(self):
        self.valid = False
        self.smooth_fit_samples = np.empty((0, 3), dtype='float')
        self.smooth_fit = None
        self.smooth_line_pos = -1.0
        self.smooth_radius_of_curvature = 0.0

        self.current_fit = None
        self.current_radius_of_curvature = 0.0
        self.current_line_pos = -1.0

        self.all_x = None
        self.all_y = None

    def fit_line(self, nonzerox, nonzeroy):
        """Fit a line to the pixels defined in nonzerox, nonzeroy."""
        if len(nonzerox) == 0:
            return False
        self.current_fit = np.polyfit(nonzeroy, nonzerox, 2)
        self._calculate_current_curvature()
        self.all_x = nonzerox
        self.all_y = nonzeroy
        return True

    def _calc_center_offset(self, fitx):
        return fitx[self.shape[0] - 1] * Config.XM_PER_PIX

    def _calc_curvature(self, fitx, fity):
        y_eval = self.shape[0] - 10
        fit_cr = np.polyfit(fity * Config.YM_PER_PIX,
                            fitx.astype(np.float) * Config.XM_PER_PIX, 2)

        return ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1])**2)**1.5) / \
                 np.absolute(2 * fit_cr[0])

    def _calculate_current_curvature(self):
        fitx, fity = self.get_current_fitted_for_shape (self.shape)

        # Workaround for python2
        if sys.version_info[0] < 3:
            return
        # fit for max y
        self.current_line_pos = self._calc_center_offset(fitx)
        self.current_radius_of_curvature = self._calc_curvature(fitx, fity)
    
    def add_current_fit_to_smooth(self): 
        self.smooth_fit_samples = np.insert(
            self.smooth_fit_samples, 0, self.current_fit, axis=0)
        num_samples = np.shape(self.smooth_fit_samples)[0]
        if (num_samples >= Config.MIN_SMOOTH_SAMPLES):
            self.valid = True
            if (num_samples > Config.MAX_SMOOTH_SAMPLES):
                self.smooth_fit_samples = np.delete(
                    self.smooth_fit_samples, Config.MAX_SMOOTH_SAMPLES, axis=0)
                num_samples = Config.MAX_SMOOTH_SAMPLES

        sample_weights = np.arange(num_samples, 0, -1)
        self.smooth_fit = np.average(
            self.smooth_fit_samples, weights=sample_weights, axis=0)
        self._calculate_smooth_curvature()

    def _calculate_smooth_curvature(self):
        fitx, fity = self.get_smooth_fitted_for_shape(self.shape)
        self.smooth_line_pos = self._calc_center_offset(fitx)
        if sys.version_info[0] < 3:
            return
        self.smooth_radius_of_curvature = self._calc_curvature(fitx, fity)

    def fit_line_from_smooth(self, nonzerox, nonzeroy, margin, window_img):
        """
        Fit a line to the pixels defined in nonzerox, nonzeroy.

        First filtered by the current bestfit line + margin
        """

        # Draw the search space onto the image
        ploty = np.linspace(
            0, window_img.shape[0] - 1, window_img.shape[0])
        fitx = self.smooth_fit[0] * ploty**2 + \
            self.smooth_fit[1] * ploty + self.smooth_fit[2]
        line_window1 = np.array(
            [np.transpose(np.vstack([fitx - margin, ploty]))])
        line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([fitx + margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 100, 0))

        inds = ((nonzerox > (self.smooth_fit[0] * (nonzeroy**2) +
                             self.smooth_fit[1] * nonzeroy +
                             self.smooth_fit[2] - margin)) &
                (nonzerox < (self.smooth_fit[0] * (nonzeroy**2) +
                             self.smooth_fit[1] * nonzeroy +
                             self.smooth_fit[2] + margin)))

        return self.fit_line(nonzerox[inds],nonzeroy[inds])

    def visualize_lane_current_fit(self, img, color=[255, 255, 0]):
        """Draw the center of the latest fitted lane line on img."""
        if (self.current_fit is None):
            return
        ploty = np.linspace(0, img.shape[0] - 1,
                            img.shape[0])
        x_fitted = self.current_fit[0] * ploty**2 + \
            self.current_fit[1] * ploty + self.current_fit[2]
        np.clip(x_fitted, 0,
                img.shape[1] - 1, out=x_fitted)
        x_fitted = x_fitted.astype(int)
        ploty = ploty.astype(int)

        img[ploty, x_fitted] = color

    def visualize_lane_smooth_fit(self, img, color=[255, 255, 0]):
        """Draw the center of the average fitted lane line on img."""
        if (self.smooth_fit is None) : 
            return
        ploty = np.linspace(
            0, img.shape[0] - 1, img.shape[0])
        fitx = self.smooth_fit[0] * ploty**2 + \
            self.smooth_fit[1] * ploty + self.smooth_fit[2]
        np.clip(fitx, 0, img.shape[1] - 1, out=fitx)
        img[ploty.astype(int), fitx.astype(int)] = color

    def get_current_fitted_for_shape(self, shape):
        """Return fitted line from the current fit, x first the y."""
        y = np.linspace(0, shape[0] - 1, shape[0])
        x = self.current_fit[0] * y**2 + \
            self.current_fit[1] * y + self.current_fit[2]
        np.clip(x, 0, shape[1] - 1, out=x)
        return x.astype(int), y

    def get_smooth_fitted_for_shape(self, shape):
        """Return fitted line from the smooth fit, x first the y."""
        y = np.linspace(0, shape[0] - 1, shape[0])
        x = self.smooth_fit[0] * y**2 + \
            self.smooth_fit[1] * y + self.smooth_fit[2]
        np.clip(x, 0, shape[1] - 1, out=x)
        return x.astype(int), y

    def probable_lane_detected(self, other):
        """
        Returns True if the current_fit for both lanes are mostly parallel
        and appropriately spaced.
            :param self: 
            :param other: other lane to compare with
        """
        
        this_x, this_y = self.get_current_fitted_for_shape(self.shape)
        other_x, other_y = other.get_current_fitted_for_shape(other.shape)
        # We would expect each width to be within a certain range.
        diff_along_y = other_x - this_x
        # Take the average over bottom 75% of the image 
        diff_along_y = diff_along_y[-int(self.shape[0] * .75):]
        mean = np.mean(diff_along_y)
        sigma = np.std(diff_along_y)

        probable_lane = sigma < Config.PROBABLE_LANE_WIDTH_STDDEV and \
            mean > Config.PROBABLE_LANE_AVERAGE_WIDTH_RANGE[0] and \
            mean < Config.PROBABLE_LANE_AVERAGE_WIDTH_RANGE[1]
        return (probable_lane, mean, sigma)

    def poly_fill_with_other_lane(self, other):
        """Fill the area between lanes and return as a new image."""
        channel = np.zeros(self.shape).astype(np.uint8)
        img = np.dstack((channel, channel, channel))
        if self.smooth_fit is None :
            this_fitx, this_ploty = self.get_current_fitted_for_shape(self.shape)
        else:
            this_fitx, this_ploty = self.get_smooth_fitted_for_shape(self.shape)

        if other.smooth_fit is None:
            other_fitx, other_ploty = other.get_current_fitted_for_shape(
                self.shape)
        else:
            other_fitx, other_ploty = other.get_smooth_fitted_for_shape(self.shape)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_this = np.array([
            np.transpose(np.vstack([this_fitx, this_ploty]))
        ])
        pts_other = np.array([
            np.flipud(np.transpose(np.vstack([other_fitx,
                                              other_ploty])))
        ])
        pts = np.hstack((pts_this, pts_other))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(img, np.int_([pts]), (0, 255, 0))
        return img
