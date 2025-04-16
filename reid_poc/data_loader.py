"""Functions for loading dataset information and individual frames."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

# Use relative imports
from reid_poc.config import PipelineConfig # Keep config import
from reid_poc.alias_types import CameraID, FrameData
from reid_poc.utils import sorted_alphanumeric

logger = logging.getLogger(__name__)

def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    """
    Retrieves validated image directories from the config and lists image filenames
    based on the first valid camera found during config setup.
    """
    logger.info("--- Loading Dataset Information (Paths and Filenames) ---")
    camera_dirs: Dict[CameraID, Path] = {}
    image_filenames: List[str] = []
    found_sequence = False

    # Iterate through the validated cameras from the config
    # relies on config.selected_cameras being sorted
    for cam_id in config.selected_cameras:
        cam_cfg = config.cameras_config.get(cam_id)
        if cam_cfg and cam_cfg.image_dir and cam_cfg.image_dir.is_dir():
            img_dir = cam_cfg.image_dir
            camera_dirs[cam_id] = img_dir # Store the validated path
            logger.debug(f"Using validated image directory for {cam_id}: {img_dir}")

            # Try to get filenames from this camera if not already found
            # (Assumes config setup already determined a valid sequence source)
            if not found_sequence:
                try:
                    current_filenames = sorted_alphanumeric([
                        f.name for f in img_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                    ])
                    if current_filenames:
                        image_filenames = current_filenames
                        found_sequence = True
                        logger.info(f"Confirmed frame sequence using camera '{cam_id}' ({len(image_filenames)} frames).")
                    # else: # Already logged during config setup if first attempt failed
                except Exception as e:
                    logger.warning(f"Error listing files for {cam_id} in {img_dir} during data loading: {e}")
        else:
            logger.warning(f"Camera {cam_id} listed in selected_cameras but missing valid configuration or image_dir in config.cameras_config. Skipping.")


    if not found_sequence:
        raise RuntimeError(f"Failed to find any image files in the validated directories for the selected cameras: {config.selected_cameras}")
    if not camera_dirs:
         raise RuntimeError(f"No valid camera data sources found for any selected camera in scene {config.selected_scene}.")

    # Double-check consistency (should always match if config setup is correct)
    if set(camera_dirs.keys()) != set(config.selected_cameras):
        logger.error(f"CRITICAL INCONSISTENCY: Configured cameras ({config.selected_cameras}) don't match found directories ({list(camera_dirs.keys())}) after validation!")
        # Update config in case validation somehow failed previously? Or raise error?
        # Let's raise for safety, this indicates a bug in config setup.
        raise RuntimeError("Mismatch between selected cameras and validated directories found.")


    return camera_dirs, image_filenames


def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the image file corresponding to 'filename' for each camera directory provided."""
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename
        img: FrameData = None
        if image_path.is_file():
            try:
                img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    logger.warning(f"[{cam_id}] Failed to decode image (empty): {image_path}")
                    img = None
            except Exception as e:
                logger.error(f"[{cam_id}] Error reading image file {image_path}: {e}")
                img = None
        else:
             # Log less frequently if file consistently missing across frames
             # logger.debug(f"[{cam_id}] Image file not found: {image_path}")
             pass
        current_frames[cam_id] = img
    return current_frames
