import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    Resize image while keeping aspect ratio and add padding.
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute scaling ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute new unpadded dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)

    dw /= 2  # divide padding into two sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)

def compute_homography(frame_width, frame_height, map_width=600, map_height=800):
    """
    Compute the homography transformation from the camera view to a top-down map.
    """
    src_points = np.array([
        [50, frame_height],                         # bottom left of the area on the frame
        [frame_width - 50, frame_height],             # bottom right
        [frame_width - 50, int(frame_height * 0.1)],    # top right
        [50, int(frame_height * 0.1)]                 # top left
    ], dtype=np.float32)

    dst_points = np.array([
        [0, map_height],          # bottom left on the map
        [map_width, map_height],  # bottom right
        [map_width, 0],           # top right
        [0, 0]                    # top left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H, src_points, dst_points