# -*- coding: utf-8 -*-
"""Handles configuration loading, definition, and compute device determination."""

import logging
import os
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch

# Attempt to locate boxmot path for default config lookup
try:
    import boxmot as boxmot_root_module
    BOXMOT_PATH = Path(boxmot_root_module.__file__).parent
except ImportError:
    BOXMOT_PATH = None # Handle case where boxmot isn't installed when config is defined

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration settings for the multi-camera pipeline."""
    # Paths
    dataset_base_path: Path = Path(
        os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"))
    reid_model_weights: Path = Path("osnet_x0_25_msmt17.pt") # Default name, path resolved later
    tracker_config_path: Optional[Path] = None # Resolved later

    # Dataset
    selected_scene: str = "s10"
    selected_cameras: List[str] = field(default_factory=lambda: ["c09", "c12", "c13", "c16"])

    # Model Params
    person_class_id: int = 1
    detection_confidence_threshold: float = 0.5
    reid_similarity_threshold: float = 0.65
    gallery_ema_alpha: float = 0.9 # Exponential Moving Average for gallery updates
    reid_refresh_interval_frames: int = 10 # Processed frames between ReID updates for a track

    # Performance
    detection_input_width: Optional[int] = 640 # Resize detection input width, None to disable
    use_amp: bool = True # Use Automatic Mixed Precision for detection if CUDA available

    # Tracker
    tracker_type: str = 'bytetrack'

    # Frame Skipping
    frame_skip_rate: int = 1 # Process 1 out of every N frames (1 = process all)

    # Execution
    device: torch.device = field(init=False) # Set by get_compute_device later

    # Visualization
    draw_bounding_boxes: bool = True
    show_track_id: bool = True
    show_global_id: bool = True
    window_name: str = "Multi-Camera Tracking & Re-ID"
    display_wait_ms: int = 1 # OpenCV waitKey delay
    max_display_width: int = 1920 # Max width for the combined display window

    # <<< DEBUG FLAGS - Set to False for standard operation >>>
    enable_debug_logging: bool = False # If True, would set logger level to DEBUG
    log_raw_detections: bool = False # If True, pipeline would log raw detections (needs code adjustment)


def get_compute_device() -> torch.device:
    """Determines and confirms the primary compute device (CUDA or CPU)."""
    logger.info("--- Determining Compute Device ---")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"Attempting CUDA: {torch.cuda.get_device_name(device)}")
            # Simple test tensor operation
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA confirmed.")
        except Exception as e:
            logger.warning(f"CUDA failed ({e}). Falling back to CPU...")
            device = torch.device("cpu") # Explicitly set back to CPU
    logger.info(f"Using device: {device.type}")
    return device

def setup_paths_and_config() -> PipelineConfig:
    """Initializes configuration, resolves paths, and determines the compute device."""
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve() # Assumes config.py is in the project root or similar
    config = PipelineConfig()
    config.device = get_compute_device()

    # Validate frame skip rate
    if config.frame_skip_rate < 1:
        logger.warning(f"Invalid frame_skip_rate ({config.frame_skip_rate}). Setting to 1.")
        config.frame_skip_rate = 1

    # --- Resolve Tracker Config Path ---
    tracker_filename = f"{config.tracker_type}.yaml"
    potential_paths = []
    if BOXMOT_PATH:
        potential_paths.append(BOXMOT_PATH / "configs" / tracker_filename)
    potential_paths.extend([
        script_dir / "configs" / tracker_filename,
        script_dir / tracker_filename,
        Path.cwd() / "configs" / tracker_filename,
        Path.cwd() / tracker_filename
    ])

    found_path = next((p for p in potential_paths if p.is_file()), None)

    if not found_path:
        logger.warning(f"Tracker config '{tracker_filename}' not found in standard locations. Searching common config directories...")
        potential_config_dirs = { p.parent for p in potential_paths }
        found_yaml = next( (yaml_file for dir_path in potential_config_dirs if dir_path.exists() for yaml_file in dir_path.glob('*.yaml') if yaml_file.is_file()), None)
        if found_yaml:
            logger.warning(f"Using fallback tracker config found: {found_yaml}.")
            found_path = found_yaml
        else:
            checked_dirs_str = "\n - ".join(map(str, sorted(list(potential_config_dirs))))
            raise FileNotFoundError(f"No tracker config (.yaml) found for '{config.tracker_type}' or any fallback in checked directories:\n - {checked_dirs_str}")

    config.tracker_config_path = found_path.resolve()
    logger.info(f"Using tracker config: {config.tracker_config_path}")

    # --- Resolve ReID Weights Path ---
    if not config.reid_model_weights.is_file():
        logger.info(f"ReID weights '{config.reid_model_weights.name}' not found relative to current dir. Searching...")
        potential_reid_paths = [
            script_dir / "weights" / config.reid_model_weights.name,
            Path.cwd() / "weights" / config.reid_model_weights.name,
            script_dir / config.reid_model_weights.name,
            Path.cwd() / config.reid_model_weights.name,
            config.reid_model_weights # Check original relative path again
        ]
        found_reid_path = next((p for p in potential_reid_paths if p.is_file()), None)
        if not found_reid_path:
             # Check BoxMOT path as last resort if path exists
            if BOXMOT_PATH and (BOXMOT_PATH / config.reid_model_weights.name).is_file():
                found_reid_path = BOXMOT_PATH / config.reid_model_weights.name
                logger.info(f"Found ReID weights in boxmot directory: {found_reid_path}")
            else:
                checked_paths_str = "\n - ".join(map(str, potential_reid_paths))
                raise FileNotFoundError(f"ReID weights '{config.reid_model_weights.name}' not found in checked paths:\n - {checked_paths_str}")
        config.reid_model_weights = found_reid_path.resolve()

    logger.info(f"Using ReID weights: {config.reid_model_weights}")

    # --- Validate Dataset Path ---
    if not config.dataset_base_path.is_dir():
        raise FileNotFoundError(f"Dataset base path not found or not a directory: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")

    logger.info("Configuration setup complete.")
    # logger.info(f"Final Config: {config}") # Optional: Log full config if needed
    return config