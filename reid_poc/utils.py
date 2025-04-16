# -*- coding: utf-8 -*-
"""General utility functions used across the pipeline."""

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Set
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
import cv2 # Import OpenCV

# Use relative import if utils.py is in the same package
from .alias_types import FeatureVector, CameraID

logger = logging.getLogger(__name__) # Use logger from the current module

def sorted_alphanumeric(data: List[str]) -> List[str]:
    """Sorts a list of strings alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def calculate_cosine_similarity(feat1: Optional[FeatureVector], feat2: Optional[FeatureVector]) -> float:
    """Calculates cosine similarity between two feature vectors, handling potential issues."""
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()

    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if not np.any(feat1) or not np.any(feat2): return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0

    try:
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0)
        return float(np.clip(similarity, 0.0, 1.0))
    except ValueError as ve:
        # logger.debug(f"Cosine distance ValueError: {ve}. Norms: {np.linalg.norm(feat1):.2f}, {np.linalg.norm(feat2):.2f}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error during cosine distance: {e}", exc_info=False)
        return 0.0

def normalize_embedding(embedding: FeatureVector) -> FeatureVector:
    """Normalizes a feature vector (L2 normalization)."""
    if embedding is None: return embedding
    norm = np.linalg.norm(embedding)
    if norm < 1e-6:
        # logger.debug("Embedding norm near zero, returning original.")
        return embedding
    return embedding / norm

def normalize_overlap_set(overlap_set: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Ensures camera pairs in the set are stored consistently (sorted tuple)."""
    normalized = set()
    for c1, c2 in overlap_set:
        normalized.add(tuple(sorted((c1, c2))))
    return normalized


# --- Homography Functions ---

def load_homography_matrix(
    camera_id: CameraID,
    scene_id: str,
    points_dir: Path = Path(".") # Assume .npz files are in the script's dir
) -> Optional[np.ndarray]:
    """
    Loads homography points from an .npz file and computes the homography matrix.
    """
    filename = points_dir / f"homography_points_{camera_id}_scene_{scene_id}.npz"
    logger.debug(f"Attempting to load homography from: {filename}") # DEBUG Log attempt
    if not filename.is_file():
        logger.warning(f"Homography file not found for Cam {camera_id}, Scene {scene_id}: {filename}")
        return None

    try:
        data = np.load(str(filename))
        image_points = data['image_points']
        map_points = data['map_points']

        logger.debug(f"[{camera_id}] Loaded points. Image points shape: {image_points.shape}, Map points shape: {map_points.shape}") # DEBUG Log shapes

        if len(image_points) < 4 or len(map_points) < 4:
            logger.warning(f"Insufficient points (<4) found in {filename} for Cam {camera_id}.")
            return None
        if len(image_points) != len(map_points):
            logger.error(f"Mismatch in point counts in {filename} for Cam {camera_id}.")
            return None

        # Use findHomography which is robust (RANSAC) and handles > 4 points
        homography_matrix, mask = cv2.findHomography(image_points, map_points, cv2.RANSAC, 5.0)

        if homography_matrix is None:
             logger.error(f"Homography calculation failed (cv2.findHomography returned None) for Cam {camera_id} using {filename}.")
             return None

        logger.info(f"Loaded homography matrix for Cam {camera_id} from {filename.name}")
        # DEBUG Log the matrix (or its shape/type) - Careful, can be large
        # logger.debug(f"[{camera_id}] Calculated Homography Matrix:\n{homography_matrix}")
        logger.debug(f"[{camera_id}] Calculated Homography Matrix (shape: {homography_matrix.shape}, type: {homography_matrix.dtype})")
        return homography_matrix

    except Exception as e:
        logger.error(f"Error loading or computing homography for Cam {camera_id} from {filename}: {e}", exc_info=True)
        return None


def project_point_to_map(
    image_point_xy: Tuple[float, float],
    homography_matrix: np.ndarray
) -> Optional[Tuple[float, float]]:
    """
    Projects a single image point (x, y) to map coordinates (X, Y) using the homography matrix.
    """
    if homography_matrix is None:
        logger.debug(f"Projection skipped: Homography matrix is None for point {image_point_xy}") # DEBUG Log skip
        return None

    logger.debug(f"Projecting image point {image_point_xy} using H matrix (shape {homography_matrix.shape})") # DEBUG Log input

    try:
        # Input point needs to be in shape (1, 1, 2) for perspectiveTransform
        img_pt = np.array([[image_point_xy]], dtype=np.float32)

        # Apply perspective transformation
        map_pt = cv2.perspectiveTransform(img_pt, homography_matrix)

        # Result is in shape (1, 1, 2), extract the coordinates
        if map_pt is not None and map_pt.shape == (1, 1, 2):
            map_x = float(map_pt[0, 0, 0])
            map_y = float(map_pt[0, 0, 1])
            logger.debug(f"  -> Projection successful: Map point ({map_x:.2f}, {map_y:.2f})") # DEBUG Log result
            return (map_x, map_y)
        else:
            logger.warning(f"Perspective transform returned unexpected shape or None for point {image_point_xy}. Result: {map_pt}")
            return None

    except Exception as e:
        # Reduce noise by logging as debug unless it's frequent
        logger.debug(f"Error projecting point {image_point_xy}: {e}", exc_info=False)
        return None