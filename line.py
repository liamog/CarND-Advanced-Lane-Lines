import numpy as np


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # y values of the last n fits of the line
        self.recent_ploty = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def visualize_lane_current_fit(self, img, color=[255, 255, 0]):
        self.recent_ploty = np.linspace(0, self.img.shape[0] - 1,
                                        self.img.shape[0])
        self.recent_xfitted = self.current_fit[0] * self.recent_ploty**2 + \
            self.current_fit[1] * self.recent_ploty + self.current_fit[2]
        np.clip(self.recent_xfitted, 0, self.img.shape[1] - 1, out=fitx)
        img[self.recent_ploty.astype(int), fitx.astype(int)] = color

    def visualize_lane_best_fit(self, img, color=[255, 255, 0]):
        self.recent_ploty = np.linspace(
            0, self.img.shape[0] - 1, self.img.shape[0])
        fitx = self.best_fit[0] * self.recent_ploty**2 + \
            self.best_fit[1] * self.recent_ploty + self.best_fit[2]
        np.clip(fitx, 0, self.img.shape[1] - 1, out=fitx)
        img[self.recent_ploty.astype(int), fitx.astype(int)] = color

    def poly_fill_with_other_lane(self, other_lane, shape):
        channel = np.zeros_like(shape).astype(np.uint8)
        img = np.dstack((channel, channel, channel))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_this = np.array([
            np.transpose(np.vstack([self.recent_xfitted, self.recent_ploty]))
        ])
        pts_other = np.array([
            np.flipud(np.transpose(np.vstack([other.recent_xfitted,
                                              other.recent_ploty])))
        ])
        pts = np.hstack((pts_this, pts_other))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(img, np.int_([pts]), (0, 255, 0))
        return img
