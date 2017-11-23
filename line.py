"""line module."""
import sys

import numpy as np

import cv2


class Line():
    """Line class captures line as a polynomial."""
    YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
    XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self, shape):
        """Line class Initializer."""
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients smoothed over the last n iterations
        self.smooth_fit = [np.array([False])]
        # radius of curvature from the smooth fit.
        self.smooth_radius_of_curvature = 0.0

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # current radius of curvature of the line in some units
        self.current_radius_of_curvature = 0.0

        # distance in meters of vehicle center from the smooth line
        self.line_base_pos = -1.0

        # samples used to calculate the smooth fit.
        self.smooth_fit_samples = np.empty((0,3), dtype='float')

        # True if the smooth line is currently considered valid.
        self.valid = False

        # x values for detected line pixels
        self.all_x = None
        # y values for detected line pixels
        self.all_y = None
        
        # shape to use to calculate curvature and lane center. Should
        # be the same as the image size being used to detect lanes.
        self.shape = shape
        self.car_center = (self.shape[1] / 2) * Line.XM_PER_PIX


    def fit_line(self, nonzerox, nonzeroy):
        """Fit a line to the pixels defined in nonzerox, nonzeroy."""
        if len(nonzerox) == 0:
            return False
        self.current_fit = np.polyfit(nonzeroy, nonzerox, 2)
        self._calculate_current_curvature()
        return True

    def _calc_center_offset(self, fitx):
        x_eval = fitx[self.shape[0] - 1] * Line.XM_PER_PIX
        return self.car_center - x_eval

    def _calc_curvature(self, fitx, fity):
        y_eval = self.shape[0] - 1
        fit_cr = np.polyfit(fity * Line.YM_PER_PIX,
                            fitx.astype(np.float) * Line.XM_PER_PIX, 2)

        return ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1])**2)**1.5) / \
                 np.absolute(2 * fit_cr[0])

    def _calculate_current_curvature(self):
        fitx, fity = self.get_current_fitted_for_shape (self.shape)

        # Workaround for python2
        if sys.version_info[0] < 3:
            return
        # fit for max y
        self.current_radius_of_curvature = self._calc_curvature(fitx, fity)
    
    def add_current_fit_to_smooth(self): 
        max_smooth_samples = 2
        self.smooth_fit_samples = np.insert(
            self.smooth_fit_samples, 0, self.current_fit, axis=0)
        if (np.shape(self.smooth_fit_samples)[0] > max_smooth_samples):
            self.valid = True
            self.smooth_fit_samples = np.delete(
                self.smooth_fit_samples, max_smooth_samples, axis=0)
        self.smooth_fit = np.average(self.smooth_fit_samples, axis=0)

    def _calculate_smooth_curvature(self):
        fitx, fity = self.get_smooth_fitted_for_shape(self.shape)
        if sys.version_info[0] < 3:
            return
        self.smooth_radius_of_curvature = self._calc_curvature(fitx, fity)

    def fit_line_from_smooth(self, nonzerox, nonzeroy, margin):
        """
        Fit a line to the pixels defined in nonzerox, nonzeroy.

        First filtered by the current bestfit line + margin
        """
        inds = ((nonzerox > (self.smooth_fit[0] * (nonzeroy**2) +
                             self.smooth_fit[1] * nonzeroy +
                             self.smooth_fit[2] - margin)) &
                (nonzerox < (self.smooth_fit[0] * (nonzeroy**2) +
                             self.smooth_fit[1] * nonzeroy +
                             self.smooth_fit[2] + margin)))
        self.all_x = nonzerox[inds]
        self.all_y = nonzeroy[inds]
        self.fit_line(self.all_x, self.all_y)

    def visualize_lane_current_fit(self, img, color=[255, 255, 0]):
        """Draw the center of the latest fitted lane line on img."""
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

    def poly_fill_with_other_lane(self, other, shape):
        """Fill the area between lanes and return as a new image."""
        channel = np.zeros(shape).astype(np.uint8)
        img = np.dstack((channel, channel, channel))

        this_fitx, this_ploty = self.get_smooth_fitted_for_shape(shape)
        other_fitx, other_ploty = other.get_smooth_fitted_for_shape(shape)

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
    
