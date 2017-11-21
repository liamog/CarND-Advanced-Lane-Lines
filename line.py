"""line module."""
import numpy as np
import cv2
import sys


class Line():
    """Line class captures line as a polynomial."""

    def __init__(self, shape):
        """Line class Initializer."""
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = 0.0
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # shape to use to calculate curvature and lane center. Should
        # be the same as the image size being used to detect lanes.
        self.shape = shape

    def fit_line(self, nonzerox, nonzeroy):
        """Fit a line to the pixels defined in nonzerox, nonzeroy."""
        if (len(nonzerox) == 0):
            self.detected = False
            return
        else:
            self.detected = True
        self.allx = nonzerox
        self.ally = nonzeroy
        self.current_fit = np.polyfit(nonzeroy, nonzerox, 2)
        # TODO Use a a weighted average for the best fit
        self.best_fit = self.current_fit
        self._calculate_curvature()

    def _calculate_curvature(self):
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        car_center = (self.shape[1] / 2) * xm_per_pix

        fitx, fity = self.get_best_fitted_for_shape(self.shape)
        x_eval = fitx[649] * xm_per_pix
        self.line_base_pos = car_center - x_eval
        if sys.version_info[0] < 3:
            return
        fit_cr = np.polyfit(fity * ym_per_pix,
                            fitx.astype(np.float) * xm_per_pix, 2)
        y_eval = np.max(fity)
        # fit for max y
        self.radius_of_curvature = (
            (1 + (2 * fit_cr[0] * y_eval + fit_cr[1])**2)**1.5) / \
            np.absolute(2 * fit_cr[0])

    def fit_line_from_current(self, nonzerox, nonzeroy, margin):
        """
        Fit a line to the pixels defined in nonzerox, nonzeroy.

        First filtered by the current bestfit line + margin
        """
        inds = ((nonzerox > (self.current_fit[0] * (nonzeroy**2) +
                             self.current_fit[1] * nonzeroy +
                             self.current_fit[2] - margin)) &
                (nonzerox < (self.current_fit[0] * (nonzeroy**2) +
                             self.current_fit[1] * nonzeroy +
                             self.current_fit[2] + margin)))
        x = nonzerox[inds]
        y = nonzeroy[inds]
        self.fit_line(x, y)

    def visualize_lane_current_fit(self, img, color=[255, 255, 0]):
        """Draw the center of the latest fitted lane line on img."""
        self.recent_ploty = np.linspace(0, img.shape[0] - 1,
                                        img.shape[0])
        self.recent_xfitted = self.current_fit[0] * self.recent_ploty**2 + \
            self.current_fit[1] * self.recent_ploty + self.current_fit[2]
        np.clip(self.recent_xfitted, 0,
                img.shape[1] - 1, out=self.recent_xfitted)
        self.recent_xfitted = self.recent_xfitted.astype(int)
        self.recent_ploty = self.recent_ploty.astype(int)

        img[self.recent_ploty, self.recent_xfitted] = color

    def visualize_lane_best_fit(self, img, color=[255, 255, 0]):
        """Draw the center of the average fitted lane line on img."""
        self.recent_ploty = np.linspace(
            0, self.img.shape[0] - 1, self.img.shape[0])
        fitx = self.best_fit[0] * self.recent_ploty**2 + \
            self.best_fit[1] * self.recent_ploty + self.best_fit[2]
        np.clip(fitx, 0, self.img.shape[1] - 1, out=fitx)
        img[self.recent_ploty.astype(int), fitx.astype(int)] = color

    def get_best_fitted_for_shape(self, shape):
        """Return fitted line, x first the y."""
        y = np.linspace(0, shape[0] - 1, shape[0])
        x = self.best_fit[0] * y**2 + \
            self.best_fit[1] * y + self.best_fit[2]
        np.clip(x, 0, shape[1] - 1, out=x)
        return x.astype(int), y

    def poly_fill_with_other_lane(self, other, shape):
        """Fill the area between lanes and return as a new image."""
        channel = np.zeros(shape).astype(np.uint8)
        img = np.dstack((channel, channel, channel))

        this_fitx, this_ploty = self.get_best_fitted_for_shape(shape)
        other_fitx, other_ploty = other.get_best_fitted_for_shape(shape)

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
