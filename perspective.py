import cv2
import numpy as np

class Perspective():
    def __init__(self):
        # perspective transform matrix
        self._perspective_transform = None

        # perspective transform inverse matrix
        self._perspective_inverse = None

        # processed image
        self.processed_image = None

        self._init_perspective_transform_matrices()

    def _init_perspective_transform_matrices(self):
        src = np.float32([[230.0, 700.0], [531.0, 495.0],
                          [762.5, 495.0], [1080.0, 700.0]])
        dst = np.float32([[230.0, 700.0], [230.0, 495.0], [
            1080.0, 495.0], [1080.0, 700.0]])
        self._perspective_transform = cv2.getPerspectiveTransform(src, dst)
        self._perspective_inverse = cv2.getPerspectiveTransform(dst, src)

    def process_image(self, img):
        self.processed_image = None

        img_size = (img.shape[1], img.shape[0])
        self.processed_image = cv2.warpPerspective(img,
                                                   self._perspective_transform,
                                                   img_size,
                                                   flags=cv2.INTER_LINEAR)
        return self.processed_image

    def process_image_inverse(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img,
                                   self._perspective_inverse,
                                   img_size, 
                                   flags=cv2.INTER_LINEAR)
