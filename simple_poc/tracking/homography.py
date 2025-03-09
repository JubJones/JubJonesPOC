import cv2
import numpy as np


def compute_homography(frame_width, frame_height, map_width, map_height):
    """Compute homography matrix for perspective transformation"""
    src_points = np.float32([
        [frame_width * 0.1, frame_height * 0.1],
        [frame_width * 0.9, frame_height * 0.1],
        [frame_width * 0.9, frame_height * 0.95],
        [frame_width * 0.1, frame_height * 0.95]
    ])

    dst_points = np.float32([
        [0, 0],
        [map_width, 0],
        [map_width, map_height],
        [0, map_height]
    ])

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H, src_points, dst_points
