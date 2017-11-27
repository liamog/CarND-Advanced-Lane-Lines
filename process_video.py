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


def process_debug_image(img):
    global lanes
    final = lanes.process_image(img)
    size = np.shape(lanes.lane_find_visualization)
    size = (int(size[0] / 2), int(size[1] / 2))
    lfv = scipy.misc.imresize(lanes.lane_find_visualization, size)
    bw = zoom(lanes.binary_warped * 255, 0.5)
    bw = np.dstack((bw, bw, bw))
    sc = zoom(lanes.source_channel, 0.5)
    sc = np.dstack((sc, sc, sc))
    ldqi = scipy.misc.imresize(lanes.diagnostics_image, size)

    diags_r1 = np.hstack((lfv, bw))
    diags_r2 = np.hstack((ldqi, sc))
    diags = np.vstack((diags_r1, diags_r2))
    final_plus_diags = np.hstack((final, diags))
    return final_plus_diags

count = 0

diagnostics_enabled = True
input_base = "project_video"

input_filename = input_base + ".mp4"
output_filename = input_base + "_with_lanes.mp4"
output_diag_filename = input_base + "_with_diagnostics.mp4"

if diagnostics_enabled:
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(process_debug_image).subclip(20,50)
    clip.write_videofile(output_diag_filename, audio=False)
else:
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(lanes.process_image)
    clip.write_videofile(output_filename, audio=False)



# clip = clip1.fl_image(process_debug_image).subclip(0, 2)
