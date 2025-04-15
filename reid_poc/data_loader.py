# -*- coding: utf-8 -*-
"""Functions for loading dataset information and individual frames."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from reid_poc.config import PipelineConfig, _find_scene_path, _find_image_dir # Use relative import
from reid_poc.alias_types import CameraID, FrameData # Use relative import
from reid_poc.utils import sorted_alphanumeric # Use relative import

logger = logging.getLogger(__name__)

def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    """
    Loads dataset structure, finds image directories for selected cameras,
    and lists image filenames based on the first valid camera.
    Relies on config object having been populated by setup_paths_and_config.
    """
    logger.info("--- Loading Dataset Information (Paths and Filenames) ---")
    camera_dirs: Dict[CameraID, Path] = {}
    image_filenames: List[str] = []
    found_sequence = False

    # Scene path should already be determined during config setup
    base_scene_path = _find_scene_path(config.dataset_base_path, config.selected_scene)
    if not base_scene_path:
        # This case should ideally be caught in config setup, but double-check
        raise FileNotFoundError(f"Scene path for '{config.selected_scene}' could not be determined.")

    # Iterate through the validated cameras from the config
    for cam_id in config.selected_cameras:
        img_dir = _find_image_dir(base_scene_path, cam_id)
        if img_dir:
            camera_dirs[cam_id] = img_dir
            # Try to get filenames from this camera if not already found
            if not found_sequence:
                try:
                    current_filenames = sorted_alphanumeric([
                        f.name for f in img_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                    ])
                    if current_filenames:
                        image_filenames = current_filenames
                        found_sequence = True
                        logger.info(f"Using camera '{cam_id}' ({len(image_filenames)} frames) for frame sequence.")
                    # else: # Logged during config setup
                        # logger.warning(f"No image files found in {img_dir} for camera {cam_id}")
                except Exception as e:
                    logger.warning(f"Error listing files for {cam_id} in {img_dir}: {e}")
        # else: # Logged during config setup
            # logger.warning(f"Skipping camera {cam_id}: No valid image directory found.")

    if not found_sequence:
        raise RuntimeError(f"Failed to find any image files in the directories for the selected cameras: {config.selected_cameras}")
    if not camera_dirs:
         raise RuntimeError(f"No valid camera data sources found for any selected camera in scene {config.selected_scene}.")

    # Verify that the final list of cameras with directories matches the config list
    if set(camera_dirs.keys()) != set(config.selected_cameras):
        logger.warning(f"Mismatch between configured cameras ({config.selected_cameras}) and found directories ({list(camera_dirs.keys())}). Using cameras with found directories.")
        # Update config in case validation somehow failed previously
        config.selected_cameras = sorted(list(camera_dirs.keys()))

    return camera_dirs, image_filenames


def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the image file corresponding to 'filename' for each camera directory provided."""
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename
        img: FrameData = None
        if image_path.is_file():
            try:
                # Use imdecode to handle potential path issues with special characters
                img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    logger.warning(f"[{cam_id}] Failed to load image (imdecode returned None or empty): {image_path}")
                    img = None # Ensure it's None if loading failed
            except Exception as e:
                logger.error(f"[{cam_id}] Error reading image file {image_path}: {e}")
                img = None
        else:
            # Log only if the file is expected but missing (might happen if datasets are inconsistent)
            # logger.debug(f"[{cam_id}] Image file not found: {image_path}") # Make this debug level
            pass
        current_frames[cam_id] = img
    return current_frames