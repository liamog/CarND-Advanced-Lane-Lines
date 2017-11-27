from moviepy.editor import VideoFileClip
from IPython.display import HTML
from lane_lines import LaneLines
import numpy as np
import scipy
from scipy.ndimage.interpolation import zoom
import scipy.misc

import cv2


def write_image(name, img):
    """
    Write an image with the correct color space.
    """
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        if (channel_count == 3):
            cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(name, img)
    return name
lanes = LaneLines('camera_cal')
count = 0

diagnostics_enabled = True
regular_enabled = False
# input_base = "harder_challenge_video"
# input_base = "challenge_video"
input_base = "project_video"

input_filename = input_base + ".mp4"
output_filename = input_base + "_with_lanes.mp4"
output_diag_filename = input_base + "_with_diagnostics.mp4"


if regular_enabled:
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(lanes.process_image)
    clip.write_videofile(output_filename, audio=False)

if diagnostics_enabled:
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(lanes.process_image_with_diagnostics)
    clip.write_videofile(output_diag_filename, audio=False)




# clip = clip1.fl_image(process_debug_image).subclip(0, 2)
