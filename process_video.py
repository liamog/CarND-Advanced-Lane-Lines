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
white_output = 'test_videos_output/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
# white_clip = clip1.fl_image(process_debug_image)
# white_clip = clip1.fl_image(process_debug_image).subclip(0, 2)
white_clip = clip1.fl_image(lanes.process_image)
white_clip.write_videofile(white_output, audio=False)
