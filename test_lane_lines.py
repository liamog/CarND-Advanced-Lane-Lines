"""Test lane lines."""

import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from lane_lines import LaneLines

files = glob.glob('project_video_imgs/*784.jpg')
files.sort()
lanes = LaneLines('camera_cal')
count = 0
for file in files:
    print(file)
    img = mpimg.imread(file)
    final = lanes.process_image(img)
    count += 1
    if (count % 1 == 0):
        print(count)

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(50, 50))

        ax1.imshow(img)
        ax1.set_title(file + ' Original Image', fontsize=10)

        ax2.imshow(lanes.diagnostics_image)
        ax2.set_title(file + ' s_channel Image - S', fontsize=10)

        ax3.imshow(lanes.binary_warped, cmap='gray')
        ax3.set_title(file + ' Thresholded Grad. Dir.', fontsize=10)

        ax4.imshow(lanes.lane_find_visualization)
        ax4.set_title(file + ' Lane Find.', fontsize=10)

        ax5.imshow(final)
        ax5.set_title(file + ' Final.', fontsize=10)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.waitforbuttonpress()
    

