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
    size = np.shape(img)
    size = (int(size[0] / 2), int(size[1] / 2))

    rc = scipy.misc.imresize(lanes.binary_image_r_channel.source_channel, size)
    rc = np.dstack((rc, rc, rc))
    rcb = scipy.misc.imresize(lanes.binary_image_r_channel.binary_warped, size)
    rcb = np.dstack((rcb, rcb, rcb))

    sc = scipy.misc.imresize(lanes.binary_image_s_channel.source_channel, size)
    sc = np.dstack((sc, sc, sc))
    scb = scipy.misc.imresize(lanes.binary_image_s_channel.binary_warped, size)
    scb = np.dstack((scb, scb, scb))

    lfv = scipy.misc.imresize(lanes.lane_find_visualization, size)

    sbw = scipy.misc.imresize(lanes.smooth_binary_warped * 255, size)
    sbw = np.dstack((sbw, sbw, sbw))
    cbw = scipy.misc.imresize(lanes.current_binary_warped * 255, size)
    cbw = np.dstack((cbw, cbw, cbw))

    di = scipy.misc.imresize(lanes.diagnostics_image, size)

    diags_1_r1 = np.hstack((rc, sc))
    diags_1_r2 = np.hstack((rcb, scb))
    diags_1 = np.vstack((diags_1_r1, diags_1_r2))

    diags_2_r1 = np.hstack((cbw, lfv))
    diags_2_r2 = np.hstack((sbw, di))
    diags_2 = np.vstack((diags_2_r1, diags_2_r2))

    final_plus_diags = np.hstack((final, diags_1, diags_2))
    return final_plus_diags

count = 0

diagnostics_enabled = True
input_base = "harder_challenge_video"
# input_base = "challenge_video"
# input_base = "project_video"

input_filename = input_base + ".mp4"
output_filename = input_base + "_with_lanes.mp4"
output_diag_filename = input_base + "_with_diagnostics.mp4"

if diagnostics_enabled:
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(process_debug_image)
    clip.write_videofile(output_diag_filename, audio=False)
else:
    clip1 = VideoFileClip(input_filename)
    clip = clip1.fl_image(lanes.process_image)
    clip.write_videofile(output_filename, audio=False)



# clip = clip1.fl_image(process_debug_image).subclip(0, 2)
