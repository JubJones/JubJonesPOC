"""Function for initializing trackers for each camera."""

import logging
from pathlib import Path
from typing import List, Dict

import torch

try:
    from boxmot import create_tracker
    from boxmot.trackers.basetracker import BaseTracker
except ImportError as e:
    logging.critical(f"Failed to import boxmot components needed for Tracking. Is boxmot installed? Error: {e}")
    # Allow script to potentially continue but tracker init will fail.
    BaseTracker = type(None) # Define a dummy type
    create_tracker = None

from reid_poc.alias_types import CameraID # Use relative import
from reid_poc.models import get_reid_device_specifier_string # Use relative import

logger = logging.getLogger(__name__)

def initialize_trackers(
    camera_ids: List[CameraID],
    tracker_type: str,
    tracker_config_path: Path,
    device: torch.device
) -> Dict[CameraID, BaseTracker]:
    """Initializes a tracker instance for each specified camera ID."""
    if create_tracker is None:
        logger.critical("FATAL ERROR: BoxMOT create_tracker not available. Cannot initialize trackers.")
        raise ImportError("BoxMOT components required for tracker initialization are missing.")

    logger.info(f"Initializing {tracker_type} trackers...")
    trackers: Dict[CameraID, BaseTracker] = {}

    if not tracker_config_path or not tracker_config_path.is_file():
        logger.critical(f"FATAL ERROR: Tracker config not found or not a file: {tracker_config_path}")
        raise FileNotFoundError(f"Tracker config not found: {tracker_config_path}")

    try:
        # Use the same device specifier logic as ReID models for consistency
        tracker_device_str = get_reid_device_specifier_string(device)

        for cam_id in camera_ids:
            # Create a new tracker instance for each camera
            # Reid weights are set to None as ReID is handled separately in the pipeline
            # Per_class is False based on original code assumption
            tracker_instance = create_tracker(
                tracker_type=tracker_type,
                tracker_config=str(tracker_config_path),
                reid_weights=None, # ReID is handled by the main pipeline, not the tracker itself
                device=tracker_device_str,
                half=False, # Assuming FP32 based on original code
                per_class=False
            )
            # Reset tracker state if possible
            if hasattr(tracker_instance, 'reset'):
                tracker_instance.reset()

            trackers[cam_id] = tracker_instance
            logger.info(f"Initialized {tracker_type} for camera {cam_id} on device '{tracker_device_str}' using config {tracker_config_path.name}")

        logger.info(f"Initialized {len(trackers)} tracker instances.")
        return trackers
    except Exception as e:
        logger.critical(f"FATAL ERROR initializing trackers: {e}", exc_info=True)
        raise RuntimeError("Failed to initialize trackers") from e