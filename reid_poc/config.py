# FILE: reid_poc/config.py
# -*- coding: utf-8 -*-
"""Handles configuration loading, definition, and compute device determination."""

import logging
import os
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
import torch
import cv2 # Needed for frame shape detection and homography calculation
import numpy as np # Needed for homography calculation

# Attempt to locate boxmot path for default config lookup
try:
    import boxmot as boxmot_root_module
    BOXMOT_PATH = Path(boxmot_root_module.__file__).parent
except ImportError:
    BOXMOT_PATH = None # Handle case where boxmot isn't installed when config is defined

# Import handoff structures and new types from alias_types
from reid_poc.alias_types import (
    CameraID, CameraHandoffConfig, ExitRule, HomographyMap, HomographyMatrix
)
from reid_poc.utils import sorted_alphanumeric, normalize_overlap_set # Use relative import for utils

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration settings for the multi-camera pipeline."""
    # --- Paths ---
    dataset_base_path: Path = Path(
        os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"))
    reid_model_weights: Path = Path("osnet_x0_25_msmt17.pt") # Default name, path resolved later
    tracker_config_path: Optional[Path] = None # Resolved later
    # --- NEW: Path to calibration point files ---
    calibration_points_dir: Path = Path(".") # Default to current dir, resolved later

    # --- Dataset ---
    selected_scene: str = "s10"
    # selected_cameras: List[str] = field(default_factory=lambda: ["c09", "c12", "c13", "c16"]) # Now derived

    # --- Model Params ---
    person_class_id: int = 1
    detection_confidence_threshold: float = 0.5
    reid_similarity_threshold: float = 0.65
    gallery_ema_alpha: float = 0.9 # Exponential Moving Average for gallery updates
    reid_refresh_interval_frames: int = 10 # Processed frames between ReID updates for a track
    lost_track_buffer_frames: int = 200

    # --- Performance ---
    detection_input_width: Optional[int] = 640 # Resize detection input width, None to disable
    use_amp: bool = True # Use Automatic Mixed Precision for detection if CUDA available

    # --- Tracker ---
    tracker_type: str = 'bytetrack'

    # --- Frame Skipping ---
    frame_skip_rate: int = 1 # Process 1 out of every N frames (1 = process all)

    # --- Handoff Logic ---
    cameras_handoff_config: Dict[CameraID, CameraHandoffConfig] = field(default_factory=dict) # Populated in setup
    possible_overlaps: Set[Tuple[str, str]] = field(default_factory=lambda: {("c09", "c16"), ("c09", "c13"), ("c12", "c13")})
    no_overlaps: Set[Tuple[str, str]] = field(default_factory=lambda: {("c12", "c09"), ("c12", "c16"), ("c13", "c16")})
    min_bbox_overlap_ratio_in_quadrant: float = 0.40 # Threshold for handoff trigger

    # --- BEV / Homography ---
    # --- NEW: Store calculated homographies ---
    homography_matrices: HomographyMap = field(default_factory=dict, init=False) # Populated in setup

    # --- Execution ---
    device: torch.device = field(init=False) # Set by get_compute_device later
    selected_cameras: List[CameraID] = field(init=False) # Derived from handoff config keys after validation

    # --- Visualization ---
    draw_bounding_boxes: bool = True
    show_track_id: bool = True
    show_global_id: bool = True
    draw_quadrant_lines: bool = True # Added for handoff viz
    highlight_handoff_triggers: bool = True # Added for handoff viz
    show_map_coordinates_on_cam: bool = False # NEW: Option to show (X,Y) on camera view
    window_name: str = "Multi-Camera Tracking, Re-ID & Handoff POC"
    display_wait_ms: int = 20 # OpenCV waitKey delay
    max_display_width: int = 1920 # Max width for the combined display window
    # --- NEW: BEV Map visualization config (Optional) ---
    # bev_map_base_image_path: Optional[Path] = None # Path to the background map image
    # generate_bev_visualization: bool = False # If True, pipeline creates BEV image

    # <<< DEBUG FLAGS - Set to False for standard operation >>>
    enable_debug_logging: bool = False # If True, would set logger level to DEBUG
    log_raw_detections: bool = False # If True, pipeline would log raw detections


def get_compute_device() -> torch.device:
    """Determines and confirms the primary compute device (CUDA > MPS > CPU)."""
    # --- Function remains the same ---
    logger.info("--- Determining Compute Device ---")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"Attempting CUDA: {torch.cuda.get_device_name(device)}")
            # Simple test tensor operation
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA confirmed.")
            return device
        except Exception as e:
            logger.warning(f"CUDA failed ({e}). Checking other options...")

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            try:
                device = torch.device("mps")
                logger.info("Attempting MPS (Apple Silicon GPU)...")
                _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
                logger.info("MPS confirmed usable.")
                return device
            except Exception as e:
                logger.warning(f"MPS detected but failed usability test ({e}). Falling back to CPU.")
        else:
            logger.info("MPS backend not built for this PyTorch version.")

    device = torch.device("cpu") # Default to CPU
    logger.info(f"Using device: {device.type}")
    return device

def _find_scene_path(base_path: Path, scene_name: str) -> Optional[Path]:
    """Tries common directory structures to find the scene path."""
    # --- Function remains the same ---
    potential_scene_paths = [
        base_path / "train" / "train" / scene_name,
        base_path / "train" / scene_name,
        base_path / scene_name
    ]
    for p in potential_scene_paths:
        if p.is_dir():
            logger.info(f"Found scene path: {p}")
            return p
    return None

def _find_image_dir(base_scene_path: Path, cam_id: str) -> Optional[Path]:
    """Tries common image subdirectory names within a camera directory."""
    # --- Function remains the same ---
    potential_img_dirs = [
        base_scene_path / cam_id / "img1", # Common MOTChallenge format
        base_scene_path / cam_id / "rgb",
        base_scene_path / cam_id # Images directly in camera folder
    ]
    for img_dir in potential_img_dirs:
        if img_dir.is_dir():
            image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            if image_files:
                logger.debug(f"Found valid image directory for {cam_id}: {img_dir}")
                return img_dir
    return None

# --- NEW Function: Load Homography Points and Calculate Matrix ---
def load_and_calculate_homography(
    calibration_points_dir: Path,
    cam_id: CameraID,
    scene_id: str
) -> Optional[HomographyMatrix]:
    """Loads calibration points from .npz file and calculates homography matrix."""
    expected_filename = calibration_points_dir / f"homography_points_{cam_id}_scene_{scene_id}.npz"
    if not expected_filename.is_file():
        logger.warning(f"Homography calibration file not found for Cam {cam_id}, Scene {scene_id}: {expected_filename}")
        return None

    try:
        data = np.load(str(expected_filename))
        image_points = data.get('image_points')
        map_points = data.get('map_points')

        if image_points is None or map_points is None:
            logger.error(f"File {expected_filename} is missing 'image_points' or 'map_points' data.")
            return None
        if len(image_points) < 4 or len(map_points) < 4:
            logger.error(f"File {expected_filename} contains less than 4 point pairs ({len(image_points)} found). Cannot compute homography.")
            return None
        if len(image_points) != len(map_points):
            logger.error(f"Mismatch in point counts in {expected_filename} ({len(image_points)} vs {len(map_points)}).")
            return None

        # Calculate Homography using findHomography (more robust for >4 points)
        # RANSAC is a good method to handle potential outliers in manual clicks
        h_matrix, status = cv2.findHomography(image_points, map_points, cv2.RANSAC, 5.0)

        if h_matrix is None or status is None or not np.any(status):
             logger.error(f"cv2.findHomography failed for {cam_id}, Scene {scene_id}. Check point quality/collinearity.")
             return None

        # Check how many points were considered inliers by RANSAC
        inlier_count = np.sum(status)
        logger.info(f"Homography calculated for Cam {cam_id}, Scene {scene_id} using {inlier_count}/{len(image_points)} inliers.")
        if inlier_count < 4:
             logger.warning(f"Low inlier count ({inlier_count}) for homography calculation for {cam_id}. Result might be unstable.")
             # Decide if you want to return None or the potentially unstable matrix
             # return None # Stricter approach
             return h_matrix # Lenient approach

        return h_matrix

    except FileNotFoundError:
        logger.warning(f"Homography calibration file not found (double check): {expected_filename}")
        return None
    except Exception as e:
        logger.error(f"Error loading or calculating homography for {cam_id}, Scene {scene_id} from {expected_filename}: {e}", exc_info=True)
        return None


def setup_paths_and_config() -> PipelineConfig:
    """Initializes configuration, resolves paths, defines handoff rules, determines compute device, and loads homographies."""
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve()
    config = PipelineConfig()
    config.device = get_compute_device()

    # --- Resolve Calibration Points Directory ---
    if not config.calibration_points_dir.is_absolute():
         # Try relative to script dir first, then current working directory
         script_rel_path = (script_dir / config.calibration_points_dir).resolve()
         cwd_rel_path = (Path.cwd() / config.calibration_points_dir).resolve()
         if script_rel_path.is_dir():
              config.calibration_points_dir = script_rel_path
              logger.info(f"Resolved calibration points dir relative to script: {config.calibration_points_dir}")
         elif cwd_rel_path.is_dir():
              config.calibration_points_dir = cwd_rel_path
              logger.info(f"Resolved calibration points dir relative to cwd: {config.calibration_points_dir}")
         else:
              # Keep the original path and log a warning, load_and_calculate_homography will handle file not found
              logger.warning(f"Calibration points directory not found at relative paths: {config.calibration_points_dir}")

    # Validate frame skip rate
    if config.frame_skip_rate < 1:
        logger.warning(f"Invalid frame_skip_rate ({config.frame_skip_rate}). Setting to 1.")
        config.frame_skip_rate = 1

    # Validate Dataset Path
    if not config.dataset_base_path.is_dir():
        raise FileNotFoundError(f"Dataset base path not found or not a directory: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")

    # Find Scene Path
    base_scene_path = _find_scene_path(config.dataset_base_path, config.selected_scene)
    if not base_scene_path:
        checked_paths_str = "\n - ".join(map(str, [
             config.dataset_base_path / "train" / "train" / config.selected_scene,
             config.dataset_base_path / "train" / config.selected_scene,
             config.dataset_base_path / config.selected_scene]))
        raise FileNotFoundError(f"Scene directory '{config.selected_scene}' not found in expected locations:\n - {checked_paths_str}")

    # Define Handoff Rules and Validate Camera Paths/Shapes
    # <<< --- DEFINE HANDOFF RULES HERE --- >>>
    # Using s10 example rules provided
    defined_handoff_configs: Dict[CameraID, CameraHandoffConfig] = {
        "c09": CameraHandoffConfig(
            id="c09",
            exit_rules=[
                ExitRule(direction='down', target_cam_id='c13', target_entry_area='upper right', notes='wait; overlap c13/c16 possible'),
            ]
        ),
        "c12": CameraHandoffConfig(
            id="c12",
            exit_rules=[
                ExitRule(direction='left', target_cam_id='c13', target_entry_area='upper left', notes='overlap c13 possible'),
            ]
        ),
        "c13": CameraHandoffConfig(
            id="c13",
            exit_rules=[
                ExitRule(direction='right', target_cam_id='c09', target_entry_area='down', notes='wait; overlap c09 possible'),
                ExitRule(direction='left', target_cam_id='c12', target_entry_area='upper left', notes='overlap c12 possible'),
            ]
        ),
        "c16": CameraHandoffConfig( # Camera included but no exit rules defined from it
            id="c16",
            exit_rules=[]
        ),
    }
    # <<< --- END OF HANDOFF RULES --- >>>

    logger.info("Validating camera paths and detecting initial frame shapes...")
    valid_cameras_handoff: Dict[CameraID, CameraHandoffConfig] = {}
    first_image_filename: Optional[str] = None

    # Determine image filenames (same logic as before)
    for cam_id in sorted(defined_handoff_configs.keys()):
        img_dir = _find_image_dir(base_scene_path, cam_id)
        if img_dir:
            try:
                image_filenames_current = sorted_alphanumeric([
                    f.name for f in img_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                ])
                if image_filenames_current:
                    first_image_filename = image_filenames_current[0]
                    logger.info(f"Using camera '{cam_id}' to determine frame sequence (found {len(image_filenames_current)} frames).")
                    break
                else:
                    logger.warning(f"No image files found in detected directory for {cam_id}: {img_dir}")
            except Exception as e:
                logger.warning(f"Error reading image files for {cam_id} from {img_dir}: {e}")
        else:
            logger.warning(f"No valid image directory (img1, rgb, or root) found for camera {cam_id} under {base_scene_path}.")

    if not first_image_filename:
        raise RuntimeError(f"Could not find any image files in any configured camera directory for scene {config.selected_scene} to establish frame sequence.")

    # Validate cameras and get frame shapes (same logic as before)
    for cam_id, cam_handoff_cfg in defined_handoff_configs.items():
        img_dir = _find_image_dir(base_scene_path, cam_id)
        if not img_dir:
            logger.error(f"Camera '{cam_id}' defined in handoff rules, but no valid image directory found. Excluding.")
            continue

        first_frame_path = img_dir / first_image_filename
        frame_shape: Optional[Tuple[int, int]] = None
        if first_frame_path.is_file():
            try:
                img = cv2.imread(str(first_frame_path))
                if img is not None and img.size > 0:
                    frame_shape = img.shape[:2]
                    logger.info(f"  Camera '{cam_id}': Path '{img_dir.name}' OK, Shape {frame_shape}")
                else:
                    logger.warning(f"  Camera '{cam_id}': Path '{img_dir.name}' OK, but failed to load first frame '{first_image_filename}'. Shape unknown.")
            except Exception as e:
                logger.error(f"  Camera '{cam_id}': Error reading first frame '{first_frame_path}': {e}. Shape unknown.")
        else:
             logger.warning(f"  Camera '{cam_id}': Path '{img_dir.name}' OK, but first frame '{first_image_filename}' not found. Shape unknown.")

        cam_handoff_cfg.frame_shape = frame_shape
        valid_cameras_handoff[cam_id] = cam_handoff_cfg

    if not valid_cameras_handoff:
        raise RuntimeError(f"No valid camera configurations remaining after checking paths and first frame for scene {config.selected_scene}.")

    config.cameras_handoff_config = valid_cameras_handoff
    config.selected_cameras = sorted(list(valid_cameras_handoff.keys()))
    logger.info(f"Final list of cameras to process: {config.selected_cameras}")

    # Normalize overlap sets
    config.possible_overlaps = normalize_overlap_set(config.possible_overlaps)
    config.no_overlaps = normalize_overlap_set(config.no_overlaps)
    logger.info(f"Normalized possible overlaps: {config.possible_overlaps}")

    # --- NEW: Load Homography Matrices ---
    logger.info("--- Loading Homography Matrices ---")
    calculated_homographies: HomographyMap = {}
    for cam_id in config.selected_cameras:
        h_matrix = load_and_calculate_homography(
            config.calibration_points_dir,
            cam_id,
            config.selected_scene
        )
        calculated_homographies[cam_id] = h_matrix # Store matrix or None if failed
        if h_matrix is None:
             logger.warning(f"Proceeding without homography for camera {cam_id}. BEV mapping will be skipped for this camera.")
        else:
             logger.info(f"Successfully loaded and calculated homography for {cam_id}.")

    config.homography_matrices = calculated_homographies # Assign to config object

    # --- Resolve Tracker Config Path (same logic as before) ---
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
        # Fallback search... (omitted for brevity, same as before)
        raise FileNotFoundError(f"No tracker config (.yaml) found for '{config.tracker_type}'")
    config.tracker_config_path = found_path.resolve()
    logger.info(f"Using tracker config: {config.tracker_config_path}")

    # --- Resolve ReID Weights Path (same logic as before) ---
    if not config.reid_model_weights.is_file():
        # Search logic... (omitted for brevity, same as before)
        potential_reid_paths = [
            script_dir / "weights" / config.reid_model_weights.name,
            Path.cwd() / "weights" / config.reid_model_weights.name,
            script_dir / config.reid_model_weights.name,
            Path.cwd() / config.reid_model_weights.name,
        ]
        if BOXMOT_PATH and (BOXMOT_PATH / config.reid_model_weights.name).is_file():
             potential_reid_paths.append(BOXMOT_PATH / config.reid_model_weights.name)
        potential_reid_paths.append(config.reid_model_weights)
        found_reid_path = next((p for p in potential_reid_paths if p.is_file()), None)
        if not found_reid_path:
             raise FileNotFoundError(f"ReID weights '{config.reid_model_weights.name}' not found")
        config.reid_model_weights = found_reid_path.resolve()
    logger.info(f"Using ReID weights: {config.reid_model_weights}")

    logger.info("Configuration setup complete.")
    return config