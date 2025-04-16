"""Handles configuration loading, definition, and compute device determination."""

import logging
import os
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set, Any
import torch
import cv2 # Needed for frame shape detection during setup
import numpy as np # Needed for homography matrix type hint
from datetime import datetime, timezone # For timestamping
import yaml # Added for YAML config loading

# Attempt to locate boxmot path for default config lookup
try:
    import boxmot as boxmot_root_module
    BOXMOT_PATH = Path(boxmot_root_module.__file__).parent
except ImportError:
    BOXMOT_PATH = None # Handle case where boxmot isn't installed when config is defined

# Import handoff structures from alias_types
from reid_poc.alias_types import CameraID, CameraHandoffConfig, ExitRule
# Import utils relatively
from reid_poc.utils import sorted_alphanumeric, normalize_overlap_set, load_homography_matrix

# Ensure root logger is configured (useful if this module is imported early)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__) # Use logger for this module


# --- MODIFIED: Moved some config structure definitions here ---
@dataclass
class CameraConfig:
    """Configuration specific to a camera, loaded from YAML and augmented."""
    id: CameraID
    image_dir: Optional[Path] = None # Validated image directory path (ADDED)
    frame_shape: Optional[Tuple[int, int]] = None # (height, width), auto-detected
    exit_rules: List[ExitRule] = field(default_factory=list)
    homography_matrix: Optional[np.ndarray] = None # Loaded homography matrix (ADDED)

@dataclass
class PipelineConfig:
    """Configuration settings for the multi-camera pipeline."""
    # --- Paths ---
    dataset_base_path: Path = Path(
        os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"))
    reid_model_weights: Path = Path("osnet_x0_25_msmt17.pt") # Default name, path resolved later
    tracker_config_path: Optional[Path] = None # Resolved later
    # REMOVED: bev_map_background_path (moved to YAML)

    # --- Dataset ---
    selected_scene: str = "s10"
    # selected_cameras: List[str] # Now derived from scene config keys

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

    # --- Handoff Logic (Loaded from YAML) ---
    cameras_config: Dict[CameraID, CameraConfig] = field(default_factory=dict) # Populated in setup (REPLACED cameras_handoff_config)
    possible_overlaps: Set[Tuple[str, str]] = field(default_factory=set) # Populated from YAML
    no_overlaps: Set[Tuple[str, str]] = field(default_factory=set) # Populated from YAML
    min_bbox_overlap_ratio_in_quadrant: float = 0.40 # Loaded from YAML

    # --- Execution ---
    device: torch.device = field(init=False) # Set by get_compute_device later
    selected_cameras: List[CameraID] = field(init=False) # Derived from handoff config keys after validation
    # REMOVED: homography_matrices (moved into CameraConfig)

    # --- Visualization ---
    draw_bounding_boxes: bool = True
    show_track_id: bool = True
    show_global_id: bool = True
    draw_quadrant_lines: bool = True
    highlight_handoff_triggers: bool = True
    window_name: str = "Multi-Camera Tracking, Re-ID & Handoff POC"
    display_wait_ms: int = 20
    max_display_width: int = 1920

    # --- BEV Map Visualization Config (Loaded from YAML) ---
    enable_bev_map: bool = True
    bev_map_display_size: Tuple[int, int] = (700, 1000) # (Height, Width)
    bev_map_world_scale: float = 0.5
    bev_map_world_origin_offset_px: Tuple[int, int] = (25, 50) # (X, Y)
    bev_map_background_path: Optional[Path] = None # Loaded from YAML, resolved

    # --- Homography Settings (Loaded from YAML) ---
    require_homography_for_bev: bool = False # Loaded from YAML

    # --- Output Configuration ---
    save_predictions: bool = False
    output_predictions_dir: Path = Path("predictions")
    save_bev_maps: bool = False
    output_bev_maps_dir: Path = Path("bev_maps")

    # >>> SET THIS TO TRUE TO ENABLE DEBUG LOGS <<<
    enable_debug_logging: bool = True
    log_raw_detections: bool = False


def get_compute_device() -> torch.device:
    """Determines and confirms the primary compute device (CUDA > MPS > CPU)."""
    logger.info("--- Determining Compute Device ---")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"Attempting CUDA: {torch.cuda.get_device_name(device)}")
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

    device = torch.device("cpu")
    logger.info(f"Using device: {device.type}")
    return device

def _find_scene_path(base_path: Path, scene_name: str) -> Optional[Path]:
    """Tries common directory structures to find the scene path."""
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

def _parse_exit_rules(rules_data: List[Dict[str, Any]]) -> List[ExitRule]:
    """Parses exit rule dictionaries into ExitRule objects."""
    parsed_rules = []
    for i, rule_dict in enumerate(rules_data):
        try:
            # Basic validation
            if not all(k in rule_dict for k in ['direction', 'target_cam_id', 'target_entry_area']):
                logger.warning(f"Skipping invalid exit rule (missing keys) at index {i}: {rule_dict}")
                continue
            parsed_rules.append(ExitRule(
                direction=rule_dict['direction'],
                target_cam_id=rule_dict['target_cam_id'],
                target_entry_area=rule_dict['target_entry_area'],
                notes=rule_dict.get('notes') # Optional
            ))
        except Exception as e:
            logger.warning(f"Error parsing exit rule at index {i}: {rule_dict}. Error: {e}")
    return parsed_rules


def setup_paths_and_config() -> PipelineConfig:
    """
    Initializes configuration, loads scene YAML, resolves paths, validates cameras,
    loads homography, creates output dirs, and determines compute device.
    """
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent # Assuming script is in reid_poc/
    config = PipelineConfig()

    # --- Set logging level based on config EARLY ---
    log_level = logging.DEBUG if config.enable_debug_logging else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        force=True)
    logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")

    config.device = get_compute_device()

    # --- Load Scene Specific YAML Configuration ---
    scene_config_filename = script_dir / "configs" / f"{config.selected_scene}_config.yaml"
    if not scene_config_filename.is_file():
         scene_config_filename = project_root / "configs" / f"{config.selected_scene}_config.yaml" # Try one level up

    if not scene_config_filename.is_file():
        raise FileNotFoundError(f"Scene configuration file not found for scene '{config.selected_scene}'. Expected at: {script_dir / 'configs'} or {project_root / 'configs'}")

    logger.info(f"Loading scene configuration from: {scene_config_filename}")
    try:
        with open(scene_config_filename, 'r') as f:
            scene_yaml_data = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading or parsing scene YAML file {scene_config_filename}: {e}") from e

    # --- Populate Config from YAML ---
    config.enable_bev_map = scene_yaml_data.get('enable_bev_map', config.enable_bev_map)
    config.bev_map_display_size = tuple(scene_yaml_data.get('bev_map_display_size', config.bev_map_display_size))
    config.bev_map_world_scale = scene_yaml_data.get('bev_map_world_scale', config.bev_map_world_scale)
    config.bev_map_world_origin_offset_px = tuple(scene_yaml_data.get('bev_map_world_origin_offset_px', config.bev_map_world_origin_offset_px))
    bev_path_str = scene_yaml_data.get('bev_map_background_path')
    if bev_path_str:
        # Resolve relative to project root first, then script dir, then treat as absolute
        potential_bev_path = project_root / bev_path_str
        if not potential_bev_path.is_file():
            potential_bev_path = script_dir / bev_path_str
        if not potential_bev_path.is_file():
             potential_bev_path = Path(bev_path_str) # Try absolute / direct path

        if potential_bev_path.is_file():
            config.bev_map_background_path = potential_bev_path.resolve()
            logger.info(f"Using BEV map background image: {config.bev_map_background_path}")
        else:
             logger.warning(f"BEV background path specified ('{bev_path_str}') but file not found relative to project root, script dir, or as absolute path. Using black background.")
             config.bev_map_background_path = None
    else:
         config.bev_map_background_path = None # Explicitly None if not specified

    config.require_homography_for_bev = scene_yaml_data.get('require_homography_for_bev', config.require_homography_for_bev)
    config.min_bbox_overlap_ratio_in_quadrant = scene_yaml_data.get('min_bbox_overlap_ratio_in_quadrant', config.min_bbox_overlap_ratio_in_quadrant)

    raw_possible_overlaps = scene_yaml_data.get('possible_overlaps', [])
    config.possible_overlaps = normalize_overlap_set({tuple(pair) for pair in raw_possible_overlaps})
    raw_no_overlaps = scene_yaml_data.get('no_overlaps', [])
    config.no_overlaps = normalize_overlap_set({tuple(pair) for pair in raw_no_overlaps})
    logger.info(f"Loaded possible overlaps: {config.possible_overlaps}")

    raw_handoff_configs = scene_yaml_data.get('cameras_handoff_config', {})
    if not raw_handoff_configs:
         raise ValueError(f"No 'cameras_handoff_config' section found in {scene_config_filename}")

    # --- Validate Frame Skip Rate ---
    if config.frame_skip_rate < 1:
        logger.warning(f"Invalid frame_skip_rate ({config.frame_skip_rate}). Setting to 1.")
        config.frame_skip_rate = 1

    # --- Validate Dataset Path ---
    if not config.dataset_base_path.is_dir():
        raise FileNotFoundError(f"Dataset base path not found or not a directory: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")

    # --- Find Scene Path ---
    base_scene_path = _find_scene_path(config.dataset_base_path, config.selected_scene)
    if not base_scene_path:
         # ... (error message as before)
         raise FileNotFoundError(f"Scene directory '{config.selected_scene}' not found...")

    # --- Validate Cameras, Paths, Shapes, and Load Homography ---
    logger.info("Validating camera paths, detecting initial frame shapes, and loading homography...")
    valid_cameras_config: Dict[CameraID, CameraConfig] = {}
    first_image_filename: Optional[str] = None
    first_valid_cam_for_seq = None

    # Find image sequence from first camera listed in YAML config
    for cam_id in sorted(raw_handoff_configs.keys()):
         img_dir_check = _find_image_dir(base_scene_path, cam_id)
         if img_dir_check:
             try:
                 image_filenames_current = sorted_alphanumeric([
                     f.name for f in img_dir_check.iterdir()
                     if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                 ])
                 if image_filenames_current:
                     first_image_filename = image_filenames_current[0]
                     first_valid_cam_for_seq = cam_id
                     logger.info(f"Using camera '{cam_id}' to determine frame sequence (found {len(image_filenames_current)} frames). First frame: '{first_image_filename}'")
                     break # Found sequence
                 else:
                     logger.warning(f"No image files found in detected directory for {cam_id}: {img_dir_check}")
             except Exception as e:
                 logger.warning(f"Error reading image files for {cam_id} from {img_dir_check}: {e}")
         else:
            logger.warning(f"Could not find valid image directory for initial sequence check for camera {cam_id} under {base_scene_path}.")

    if not first_image_filename or not first_valid_cam_for_seq:
         raise RuntimeError(f"Could not find any image files in any configured camera directory for scene {config.selected_scene} to establish frame sequence.")

    # Now validate all cameras defined in the YAML
    homography_points_dir = script_dir / "homography_points/"
    missing_homography_for_bev = []

    for cam_id, cam_data in raw_handoff_configs.items():
        img_dir = _find_image_dir(base_scene_path, cam_id)
        if not img_dir:
            logger.error(f"Camera '{cam_id}' defined in config, but no valid image directory found. Excluding.")
            continue

        # Load first frame to get shape
        first_frame_path = img_dir / first_image_filename
        frame_shape: Optional[Tuple[int, int]] = None
        if first_frame_path.is_file():
            try:
                img = cv2.imread(str(first_frame_path))
                if img is not None and img.size > 0:
                    frame_shape = img.shape[:2] # (height, width)
                    logger.info(f"  Camera '{cam_id}': Path '{img_dir.name}' OK, Shape {frame_shape}")
                else:
                    logger.warning(f"  Camera '{cam_id}': Path OK, but failed to load first frame '{first_image_filename}'. Shape unknown.")
            except Exception as e:
                logger.error(f"  Camera '{cam_id}': Error reading first frame '{first_frame_path}': {e}. Shape unknown.")
        else:
             logger.warning(f"  Camera '{cam_id}': Path OK, but first frame '{first_image_filename}' not found. Shape unknown.")

        # Load Homography
        h_matrix = load_homography_matrix(cam_id, config.selected_scene, homography_points_dir)
        if h_matrix is None:
             logger.warning(f"Could not load homography matrix for camera {cam_id}.")
             if config.enable_bev_map and config.require_homography_for_bev:
                  missing_homography_for_bev.append(cam_id)

        # Parse rules
        exit_rules = _parse_exit_rules(cam_data.get('exit_rules', []))

        # Store validated config
        valid_cameras_config[cam_id] = CameraConfig(
             id=cam_id,
             image_dir=img_dir.resolve(), # Store resolved path
             frame_shape=frame_shape,
             exit_rules=exit_rules,
             homography_matrix=h_matrix
         )

    if not valid_cameras_config:
        raise RuntimeError(f"No valid camera configurations remaining for scene {config.selected_scene}.")

    # --- Enforce Homography Requirement ---
    if missing_homography_for_bev:
         raise RuntimeError(f"BEV map enabled and require_homography_for_bev=True, but homography matrix is missing for camera(s): {', '.join(missing_homography_for_bev)}. Please generate homography files or disable the requirement.")

    config.cameras_config = valid_cameras_config
    config.selected_cameras = sorted(list(valid_cameras_config.keys()))
    logger.info(f"Final list of cameras to process: {config.selected_cameras}")
    logger.info(f"Loaded {sum(1 for cfg in config.cameras_config.values() if cfg.homography_matrix is not None)} homography matrices.")

    # --- Resolve Tracker Config Path ---
    tracker_filename = f"{config.tracker_type}.yaml"
    potential_paths = []
    if BOXMOT_PATH: potential_paths.append(BOXMOT_PATH / "configs" / tracker_filename)
    potential_paths.extend([
        script_dir / "configs" / tracker_filename, project_root / "configs" / tracker_filename,
        script_dir / tracker_filename, project_root / tracker_filename,
        Path.cwd() / "configs" / tracker_filename, Path.cwd() / tracker_filename
    ])
    # ... (rest of tracker path finding logic as before) ...
    found_path = next((p for p in potential_paths if p.is_file()), None)
    if not found_path:
        # ... (fallback logic as before) ...
        raise FileNotFoundError(f"No tracker config (.yaml) found for '{config.tracker_type}'...") # Shortened message
    config.tracker_config_path = found_path.resolve()
    logger.info(f"Using tracker config: {config.tracker_config_path}")


    # --- Resolve ReID Weights Path ---
    # ... (ReID weight path finding logic as before, potentially checking project_root / 'weights') ...
    potential_reid_paths = [
        script_dir / "weights" / config.reid_model_weights.name,
        project_root / "weights" / config.reid_model_weights.name,
        script_dir / config.reid_model_weights.name,
        project_root / config.reid_model_weights.name,
        Path.cwd() / "weights" / config.reid_model_weights.name,
        Path.cwd() / config.reid_model_weights.name,
    ]
    if BOXMOT_PATH and (BOXMOT_PATH / config.reid_model_weights.name).is_file():
        potential_reid_paths.append(BOXMOT_PATH / config.reid_model_weights.name)
    potential_reid_paths.append(config.reid_model_weights) # Check path provided directly
    found_reid_path = next((p for p in potential_reid_paths if p.is_file()), None)
    if not found_reid_path:
         # ... (error message as before) ...
         raise FileNotFoundError(f"ReID weights '{config.reid_model_weights.name}' not found...") # Shortened message
    config.reid_model_weights = found_reid_path.resolve()
    logger.info(f"Using ReID weights: {config.reid_model_weights}")


    # --- Create Output Directories ---
    cwd = Path.cwd() # Assume relative to where main.py is run
    if config.save_predictions:
        try:
            output_dir = cwd / config.output_predictions_dir; output_dir.mkdir(parents=True, exist_ok=True)
            config.output_predictions_dir = output_dir.resolve()
            logger.info(f"Saving predictions enabled. Output directory: {config.output_predictions_dir}")
        except Exception as e: logger.error(f"Failed create predictions dir: {e}. Disabling."); config.save_predictions = False
    else: logger.info("Saving predictions JSON disabled.")

    if config.save_bev_maps:
        if not config.enable_bev_map: logger.warning("Save BEV maps enabled, but BEV generation disabled."); config.save_bev_maps = False
        else:
            try:
                output_dir_bev = cwd / config.output_bev_maps_dir; output_dir_bev.mkdir(parents=True, exist_ok=True)
                config.output_bev_maps_dir = output_dir_bev.resolve()
                logger.info(f"Saving BEV maps enabled. Output directory: {config.output_bev_maps_dir}")
            except Exception as e: logger.error(f"Failed create BEV maps dir: {e}. Disabling."); config.save_bev_maps = False
    else: logger.info("Saving BEV map images disabled.")

    logger.info("Configuration setup complete.")
    return config