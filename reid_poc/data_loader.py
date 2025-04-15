# -*- coding: utf-8 -*-
"""Functions for loading dataset information and individual frames."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

from reid_poc.config import PipelineConfig # Use relative import
from reid_poc.alias_types import CameraID, FrameData # Use relative import
from reid_poc.utils import sorted_alphanumeric # Use relative import

logger = logging.getLogger(__name__)

def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    """Loads dataset structure, finds image directories for selected cameras, and lists image filenames."""
    logger.info("--- Loading Dataset Information ---")
    camera_dirs: Dict[CameraID, Path] = {}
    valid_cameras: List[CameraID] = []

    # Try common dataset structures
    potential_scene_paths = [
        config.dataset_base_path / "train" / "train" / config.selected_scene,
        config.dataset_base_path / "train" / config.selected_scene,
        config.dataset_base_path / config.selected_scene
    ]
    base_scene_path = None
    for p in potential_scene_paths:
        if p.is_dir():
            base_scene_path = p
            logger.info(f"Using scene path: {base_scene_path}")
            break

    if not base_scene_path:
         checked_paths_str = "\n - ".join(map(str, potential_scene_paths))
         raise FileNotFoundError(f"Scene directory '{config.selected_scene}' not found in expected locations:\n - {checked_paths_str}")

    # Find image directories for each selected camera
    for cam_id in config.selected_cameras:
        # Look for common image folder names within the camera directory
        potential_img_dirs = [
            base_scene_path / cam_id / "img1", # Common MOTChallenge format
            base_scene_path / cam_id / "rgb",
            base_scene_path / cam_id # Images directly in camera folder
        ]
        found_dir = None
        for img_dir in potential_img_dirs:
            if img_dir.is_dir():
                # Check if directory contains common image files
                image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                if image_files:
                    found_dir = img_dir
                    logger.info(f"Found valid image directory for {cam_id}: {found_dir} ({len(image_files)} images detected).")
                    break # Found a valid directory for this camera

        if found_dir:
            camera_dirs[cam_id] = found_dir
            valid_cameras.append(cam_id)
        else:
            logger.warning(f"No valid image directory (containing .jpg/.png) found for camera {cam_id} under {base_scene_path}. Checked common subdirs (img1, rgb) and the camera root. Skipping this camera.")

    if not valid_cameras:
        raise RuntimeError(f"No valid camera data sources found for any selected camera in scene {config.selected_scene}.")

    # Update config in-place if some cameras were skipped
    if len(valid_cameras) != len(config.selected_cameras):
         logger.warning(f"Processing only the valid cameras found: {valid_cameras}")
         config.selected_cameras = valid_cameras # Modify the config object directly

    # Get the list of frame filenames from the first valid camera
    image_filenames: List[str] = []
    first_cam_dir = camera_dirs[valid_cameras[0]]
    try:
        image_filenames = sorted_alphanumeric([f.name for f in first_cam_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
        if not image_filenames:
            raise ValueError(f"No image files (.jpg, .jpeg, .png, .bmp) found in the directory for the first valid camera: {first_cam_dir}")
        logger.info(f"Found {len(image_filenames)} frames based on camera {valid_cameras[0]}. Assuming consistent filenames across cameras.")
    except Exception as e:
        logger.critical(f"Failed to list image files from {first_cam_dir}: {e}", exc_info=True)
        raise RuntimeError(f"Failed list image files: {e}") from e

    return camera_dirs, image_filenames


def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the image file corresponding to 'filename' for each camera directory provided."""
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename
        img: FrameData = None
        if image_path.is_file():
            try:
                img = cv2.imread(str(image_path))
                if img is None or img.size == 0:
                    logger.warning(f"[{cam_id}] Failed to load image (cv2.imread returned None or empty): {image_path}")
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