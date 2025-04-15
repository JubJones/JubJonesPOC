# -*- coding: utf-8 -*-
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN

logger = logging.getLogger(__name__)
# Ensure torchvision.transforms.v2 is available or fallback if needed
try:
    from torchvision.transforms.v2 import Compose
    from torchvision.transforms.v2 import functional as F_v2
    logger.info("Using torchvision.transforms.v2")
except ImportError:
    logger.warning("torchvision.transforms.v2 not found, falling back to v1.")
    from torchvision.transforms import Compose # Fallback for older torchvision
    from torchvision.transforms import functional as F_v1 # Fallback
from PIL import Image

# --- Basic Logging ---
# <<< Set Logging Level to INFO >>>
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("HandoffTriggerPOC")

# --- Type Aliases ---
CameraID = str
FrameData = Optional[np.ndarray]
BoundingBox = np.ndarray # xyxy format [x1, y1, x2, y2]
Detection = Dict[str, Any] # {'bbox_xyxy': BoundingBox, 'conf': float, 'class_id': int}
ExitDirection = str # 'up', 'down', 'left', 'right'

# --- Configuration ---

@dataclass
class ExitRule:
    direction: ExitDirection
    target_cam_id: str
    target_entry_area: str # Descriptive: 'upper left', 'down', etc.
    notes: Optional[str] = None # e.g., 'wait required', 'overlap possible'

@dataclass
class CameraHandoffConfig:
    id: CameraID
    source_path: Path # Path to the 'rgb' folder for this camera
    frame_shape: Optional[Tuple[int, int]] = None # (height, width), auto-detected
    exit_rules: List[ExitRule] = field(default_factory=list)

@dataclass
class PipelineHandoffConfig:
    cameras: Dict[CameraID, CameraHandoffConfig]

    # Relationship Info (derived from user description)
    possible_overlaps: Set[Tuple[str, str]] = field(default_factory=lambda: {("c09", "c16"), ("c09", "c13"), ("c12", "c13")})
    no_overlaps: Set[Tuple[str, str]] = field(default_factory=lambda: {("c12", "c09"), ("c12", "c16"), ("c13", "c16")})

    # Detector Config
    detection_confidence_threshold: float = 0.5
    person_class_id: int = 1 # Faster R-CNN typically uses 1 for person
    detection_input_width: Optional[int] = 640
    use_amp: bool = True

    # Exit Detection Params
    edge_threshold_ratio: float = 0.15 # Increased edge region size
    min_bbox_area_ratio_in_edge: float = 0.30

    # Execution
    device: torch.device = field(default_factory=lambda: torch.device("cpu")) # Default, will be updated

    # Display
    max_display_width: int = 1920
    display_wait_ms: int = 100 # Slow down to see triggers


# --- Helper Functions ---
def sorted_alphanumeric(data: List[str]) -> List[str]:
    """Sorts strings containing numbers naturally."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_compute_device() -> torch.device:
    """Determines the best available compute device (CUDA > MPS > CPU)."""
    logger.info("--- Determining Compute Device ---")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"CUDA available: {torch.cuda.get_device_name(device)}")
            # Simple test
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA confirmed usable.")
            return device
        except Exception as e:
            logger.warning(f"CUDA detected but failed usability test ({e}). Checking other options.")
    else:
         logger.info("CUDA / MPS not available.")

    logger.info("Using CPU device.")
    return torch.device("cpu")

def normalize_overlap_set(overlap_set: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Ensures pairs are stored consistently (e.g., ('a', 'b') not ('b', 'a'))."""
    normalized = set()
    for c1, c2 in overlap_set:
        normalized.add(tuple(sorted((c1, c2))))
    return normalized

# --- Detection Logic (Adapted from original script) ---

def load_detector(device: torch.device) -> Tuple[FasterRCNN, Any]:
    """Loads the Faster R-CNN model and corresponding transforms."""
    logger.info("Loading Faster R-CNN detector (ResNet50 FPN)...")
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        model.to(device)
        model.eval()
        # Get the transforms associated with the weights
        # This should work for both v1 and v2 transforms API
        transforms = weights.transforms()
        logger.info("Faster R-CNN Detector loaded successfully.")
        return model, transforms
    except Exception as e:
        logger.critical(f"FATAL ERROR loading Faster R-CNN: {e}", exc_info=True)
        raise RuntimeError("Failed to load object detector") from e

def preprocess_frame_for_detection(
    frame_bgr: np.ndarray,
    transforms: Any,
    target_width: Optional[int],
    device: torch.device
) -> Tuple[Optional[torch.Tensor], float, float, Tuple[int, int]]:
    """Prepares a single frame for the detector, including optional resizing."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None, 1.0, 1.0, (0, 0)

    original_h, original_w = frame_bgr.shape[:2]
    img_for_det = frame_bgr
    scale_x, scale_y = 1.0, 1.0

    # Optional resizing
    if target_width and original_w > target_width:
        scale = target_width / original_w
        target_h = int(original_h * scale)
        try:
            img_for_det = cv2.resize(frame_bgr, (target_width, target_h), interpolation=cv2.INTER_LINEAR)
            scale_x = original_w / target_width
            scale_y = original_h / target_h
        except Exception as resize_err:
            logger.warning(f"Resize failed: {resize_err}. Using original.")
            img_for_det = frame_bgr
            scale_x, scale_y = 1.0, 1.0

    try:
        # Convert BGR (OpenCV) to RGB
        img_rgb = cv2.cvtColor(img_for_det, cv2.COLOR_BGR2RGB)

        # Apply transforms. The transforms object from weights should handle
        # conversion (e.g., to tensor) internally. v2 might prefer tensors.
        # Let's assume PIL input is safer for compatibility for now.
        img_pil = Image.fromarray(img_rgb)
        input_tensor = transforms(img_pil)

        return input_tensor.to(device), scale_x, scale_y, (original_h, original_w)

    except Exception as transform_err:
        logger.error(f"Preprocessing failed: {transform_err}")
        return None, 1.0, 1.0, (original_h, original_w)


def detect_persons(
    detector: FasterRCNN,
    batch_input_tensors: List[torch.Tensor],
    batch_scale_factors: List[Tuple[float, float]],
    batch_original_shapes: List[Tuple[int, int]],
    batch_cam_ids: List[CameraID],
    config: PipelineHandoffConfig
) -> Dict[CameraID, List[Detection]]:
    """Performs batched detection and post-processes results."""
    detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)
    if not batch_input_tensors:
        return detections_per_camera

    all_predictions: List[Dict[str, torch.Tensor]] = []
    try:
        with torch.no_grad():
            use_amp_runtime = config.use_amp and config.device.type == 'cuda'
            with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                # Ensure input is a list of tensors
                if isinstance(batch_input_tensors, torch.Tensor):
                    # Handle cases where only one image is processed?
                    # Detector expects a list even for a single image.
                    batch_input_tensors = [batch_input_tensors]
                all_predictions = detector(batch_input_tensors)
    except Exception as e:
        logger.error(f"Detection inference failed: {e}", exc_info=False)
        return detections_per_camera # Return empty if inference fails

    if len(all_predictions) != len(batch_cam_ids):
        logger.error(f"Detection output mismatch: {len(all_predictions)} predictions vs {len(batch_cam_ids)} inputs.")
        return detections_per_camera

    # Postprocess
    for i, prediction_dict in enumerate(all_predictions):
        cam_id = batch_cam_ids[i]
        scale_x, scale_y = batch_scale_factors[i]
        original_h, original_w = batch_original_shapes[i]

        try:
            # Move predictions to CPU for numpy conversion
            pred_boxes = prediction_dict['boxes'].cpu().numpy()
            pred_labels = prediction_dict['labels'].cpu().numpy()
            pred_scores = prediction_dict['scores'].cpu().numpy()

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if label == config.person_class_id and score >= config.detection_confidence_threshold:
                    # Scale box back to original frame coordinates
                    x1, y1, x2, y2 = box
                    orig_x1 = np.clip(x1 * scale_x, 0, original_w - 1)
                    orig_y1 = np.clip(y1 * scale_y, 0, original_h - 1)
                    orig_x2 = np.clip(x2 * scale_x, 0, original_w - 1)
                    orig_y2 = np.clip(y2 * scale_y, 0, original_h - 1)

                    # Ensure box has valid dimensions
                    if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1:
                        bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32)
                        detections_per_camera[cam_id].append({
                            'bbox_xyxy': bbox_orig,
                            'conf': float(score),
                            'class_id': int(label)
                        })
        except Exception as postproc_err:
            logger.error(f"[{cam_id}] Postprocessing detection error: {postproc_err}")

    return detections_per_camera

# --- Exit Condition Logic ---

def check_exit_conditions(
    detections: List[Detection],
    frame_shape: Tuple[int, int],
    edge_threshold_ratio: float,
    min_bbox_area_ratio_in_edge: float
) -> List[Tuple[BoundingBox, ExitDirection]]:
    """
    Checks if detections are near edges, assuming exit direction based on location.
    Returns a list of (bbox, assumed_exit_direction) for potential exits.
    """
    potential_exits = []
    if not detections or not frame_shape:
        return potential_exits

    H, W = frame_shape
    if H == 0 or W == 0: return potential_exits # Avoid division by zero if shape is invalid

    edge_h = int(H * edge_threshold_ratio)
    edge_w = int(W * edge_threshold_ratio)

    # Define edge regions (x1, y1, x2, y2)
    edge_regions: Dict[ExitDirection, Tuple[int, int, int, int]] = {
        'up': (0, 0, W - 1, edge_h),
        'down': (0, H - edge_h, W - 1, H - 1), # Adjusted calculation for bottom edge
        'left': (0, 0, edge_w, H - 1),
        'right': (W - edge_w, 0, W - 1, H - 1), # Adjusted calculation for right edge
    }

    for det in detections:
        bbox = det['bbox_xyxy']
        x1, y1, x2, y2 = map(int, bbox)
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        if bbox_w <= 0 or bbox_h <= 0: continue # Skip invalid boxes
        bbox_area = bbox_w * bbox_h

        for direction, (ex1, ey1, ex2, ey2) in edge_regions.items():
            # Calculate intersection area between bbox and edge region
            inter_x1 = max(x1, ex1)
            inter_y1 = max(y1, ey1)
            inter_x2 = min(x2, ex2)
            inter_y2 = min(y2, ey2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            # Check if intersection area is significant relative to bbox area
            if inter_area / bbox_area >= min_bbox_area_ratio_in_edge:
                potential_exits.append((bbox, direction))
                # Assume only one exit direction per detection for simplicity
                break

    return potential_exits

# --- Handoff Processor Class ---

class HandoffTrigger:
    """Handles detection and triggering handoff messages based on edge proximity."""
    def __init__(self, config: PipelineHandoffConfig):
        self.config = config
        # Normalize overlap sets once during init
        self.config.possible_overlaps = normalize_overlap_set(config.possible_overlaps)
        self.config.no_overlaps = normalize_overlap_set(config.no_overlaps)

        self.detector, self.detector_transforms = load_detector(config.device)
        self.exit_triggers_this_frame: List[Tuple[CameraID, BoundingBox, ExitRule]] = []

    def process_frame_batch(self, frames: Dict[CameraID, FrameData]) -> Dict[CameraID, List[Detection]]:
        """Processes a batch of frames: detects persons and checks for exit triggers."""
        self.exit_triggers_this_frame.clear()
        batch_input_tensors = []
        batch_scale_factors = []
        batch_original_shapes = []
        batch_cam_ids = []

        # --- Stage 1: Preprocess ---
        valid_cam_ids_this_batch = []
        for cam_id, frame in frames.items():
            if frame is not None:
                 input_tensor, scale_x, scale_y, orig_shape = preprocess_frame_for_detection(
                     frame, self.detector_transforms, self.config.detection_input_width, self.config.device
                 )
                 if input_tensor is not None:
                     batch_input_tensors.append(input_tensor)
                     batch_scale_factors.append((scale_x, scale_y))
                     batch_original_shapes.append(orig_shape)
                     batch_cam_ids.append(cam_id)
                     valid_cam_ids_this_batch.append(cam_id) # Keep track of cams we expect results for

        # --- Stage 2: Detect ---
        detections_per_camera = defaultdict(list) # Ensure it's defined even if detection fails
        if batch_input_tensors:
            detections_per_camera = detect_persons(
                self.detector, batch_input_tensors, batch_scale_factors,
                batch_original_shapes, batch_cam_ids, self.config
            )
        else:
             logger.debug("No valid frames preprocessed for detection.")


        # --- Stage 3: Check Exit Conditions & Trigger ---
        for cam_id in valid_cam_ids_this_batch: # Iterate over cams we actually processed
            detections = detections_per_camera.get(cam_id, []) # Get detections or empty list
            cam_config = self.config.cameras.get(cam_id)

            if not cam_config or cam_config.frame_shape is None:
                logger.warning(f"Skipping exit check for {cam_id}: No config or frame shape.")
                continue

            potential_exits = check_exit_conditions(
                detections,
                cam_config.frame_shape,
                self.config.edge_threshold_ratio,
                self.config.min_bbox_area_ratio_in_edge
            )

            for bbox, exit_direction in potential_exits:
                # Find the corresponding exit rule in config
                matched_rule = None
                for rule in cam_config.exit_rules:
                    if rule.direction == exit_direction:
                        matched_rule = rule
                        break

                if matched_rule:
                    self.exit_triggers_this_frame.append((cam_id, bbox, matched_rule))

                    # Determine relevant cameras based on rule and overlap info
                    target_cam = matched_rule.target_cam_id

                    # Simplified Relevant Cams: Just the target + anything overlapping the target
                    # Uses the pre-normalized overlap set
                    final_relevant_cams = {target_cam}
                    for c1, c2 in self.config.possible_overlaps:
                         pair = tuple(sorted((c1, c2)))
                         if c1 == target_cam: final_relevant_cams.add(c2)
                         if c2 == target_cam: final_relevant_cams.add(c1)

                    # Format the log message
                    log_msg = (
                        f"EXIT TRIGGER: Cam [{cam_id}] Person at [{bbox.astype(int)}] "
                        f"assumed exit [{matched_rule.direction}]. "
                        f"Potential entry: Cam [{matched_rule.target_cam_id}] "
                        f"Area [{matched_rule.target_entry_area}]. "
                        f"Cameras to check: {sorted(list(final_relevant_cams))}"
                    )
                    if matched_rule.notes:
                        log_msg += f" (Notes: {matched_rule.notes})"
                    logger.info(log_msg)

        return detections_per_camera # Return all detections for drawing

    def draw_results(self, frames: Dict[CameraID, FrameData], detections_per_camera: Dict[CameraID, List[Detection]]) -> Dict[CameraID, FrameData]:
        """Draws detections, edge regions, and highlights triggered exits."""
        annotated_frames = {}
        default_h, default_w = 1080, 1920 # Fallback size
        first_valid_frame = next((f for f in frames.values() if f is not None), None)
        if first_valid_frame is not None:
             default_h, default_w = first_valid_frame.shape[:2]

        edge_color = (255, 0, 255) # Magenta for edge regions
        detection_color = (0, 255, 0) # Green for detected persons
        trigger_color = (0, 0, 255) # Red for triggered exit highlight

        for cam_id, frame in frames.items():
            cam_config = self.config.cameras.get(cam_id)
            # Use detected shape if available, else default
            H, W = (cam_config.frame_shape if cam_config and cam_config.frame_shape else (default_h, default_w))

            if frame is None:
                # Create placeholder if frame is missing
                annotated_frame = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(annotated_frame, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, trigger_color, 2)
                annotated_frames[cam_id] = annotated_frame
                continue

            annotated_frame = frame.copy()

            # Draw Edge Regions for visualization
            if H > 0 and W > 0: # Only draw if shape is valid
                edge_h = int(H * self.config.edge_threshold_ratio)
                edge_w = int(W * self.config.edge_threshold_ratio)
                cv2.rectangle(annotated_frame, (0, 0), (W - 1, edge_h), edge_color, 1) # Up
                cv2.rectangle(annotated_frame, (0, H - edge_h), (W - 1, H - 1), edge_color, 1) # Down
                cv2.rectangle(annotated_frame, (0, 0), (edge_w, H - 1), edge_color, 1) # Left
                cv2.rectangle(annotated_frame, (W - edge_w, 0), (W - 1, H - 1), edge_color, 1) # Right

            # Draw Detections
            for det in detections_per_camera.get(cam_id, []):
                x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
                conf = det['conf']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), detection_color, 2)
                label = f"Person {conf:.2f}"
                # Adjust text position if near top edge
                text_y = y1 - 10 if y1 > 20 else y2 + 15
                cv2.putText(annotated_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 1, cv2.LINE_AA)

            # Highlight triggered exits
            for trig_cam_id, trig_bbox, trig_rule in self.exit_triggers_this_frame:
                 if trig_cam_id == cam_id:
                     x1, y1, x2, y2 = map(int, trig_bbox)
                     # Draw thicker red box around triggered detection
                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), trigger_color, 3)
                     # Add text below the box
                     cv2.putText(annotated_frame, f"EXIT? ({trig_rule.direction} -> {trig_rule.target_cam_id})",
                                 (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trigger_color, 2)

            annotated_frames[cam_id] = annotated_frame

        return annotated_frames

# --- Main Execution Functions (Adapted) ---

def setup_paths_and_config() -> PipelineHandoffConfig:
    """Sets up the main configuration object based on dataset structure."""
    logger.info("--- Setting up Configuration and Paths ---")

    # <<< --- DEFINE YOUR CONFIGURATION HERE --- >>>
    try:
        # Try getting path from environment variable first
        dataset_base_str = os.getenv("MTMMC_PATH", None)
        if dataset_base_str:
             dataset_base = Path(dataset_base_str)
             logger.info(f"Using MTMMC_PATH from environment: {dataset_base}")
        else:
             # Fallback path if environment variable not set
             default_path = "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"
             dataset_base = Path(default_path)
             logger.info(f"MTMMC_PATH not set, using default: {dataset_base}")

        if not dataset_base.is_dir():
             logger.warning(f"Dataset base path not found: {dataset_base}. Please set MTMMC_PATH or adjust the default.")
             # Attempting execution might still fail later if path is wrong

    except Exception as e:
         logger.error(f"Error determining dataset base path: {e}. Please check paths.")
         raise # Re-raise after logging

    scene = "s10" # CHANGE AS NEEDED
    # Construct path according to MTMMC structure: MTMMC/train/train/sXX
    base_scene_path = dataset_base / "train" / "train" / scene

    # Verify base scene path exists
    if not base_scene_path.is_dir():
        # Fallback: Check if structure is MTMMC/train/sXX (without extra 'train')
        alt_scene_path_1 = dataset_base / "train" / scene
        # Fallback: Check if structure is MTMMC/sXX (root level)
        alt_scene_path_2 = dataset_base / scene

        if alt_scene_path_1.is_dir():
            base_scene_path = alt_scene_path_1
            logger.warning(f"Using alternative scene path structure: {base_scene_path}")
        elif alt_scene_path_2.is_dir():
             base_scene_path = alt_scene_path_2
             logger.warning(f"Using alternative scene path structure: {base_scene_path}")
        else:
            raise FileNotFoundError(f"Scene directory '{scene}' not found under expected paths within {dataset_base}")

    config = PipelineHandoffConfig(
        cameras={
            # Define cameras and their RGB paths based on the identified base_scene_path
            "c09": CameraHandoffConfig(
                id="c09",
                source_path=base_scene_path / "c09" / "rgb", # Path to images
                exit_rules=[
                    ExitRule(direction='down', target_cam_id='c13', target_entry_area='upper right', notes='wait; overlap c13/c16 possible'),
                ]
            ),
            "c12": CameraHandoffConfig(
                id="c12",
                source_path=base_scene_path / "c12" / "rgb",
                exit_rules=[
                    ExitRule(direction='left', target_cam_id='c13', target_entry_area='upper left', notes='overlap c13 possible'),
                    # Assuming 'upper left' direction primarily involves moving left
                ]
            ),
            "c13": CameraHandoffConfig(
                id="c13",
                source_path=base_scene_path / "c13" / "rgb",
                exit_rules=[
                    # Add rules if needed, e.g., maybe exiting right goes somewhere?
                ]
            ),
             "c16": CameraHandoffConfig(
                 id="c16",
                 source_path=base_scene_path / "c16" / "rgb",
                 exit_rules=[
                    # Add rules if needed
                 ]
             ),
        },
        # Overlap/No-overlap rules based on user text
        possible_overlaps = {("c09", "c16"), ("c09", "c13"), ("c12", "c13")},
        no_overlaps = {("c12", "c09"), ("c12", "c16"), ("c13", "c16")},

        # Other params
        detection_confidence_threshold=0.5,
        person_class_id=1,
        detection_input_width=640,
        use_amp=True,
        edge_threshold_ratio=0.15,
        min_bbox_area_ratio_in_edge=0.30,
        max_display_width=1920,
        display_wait_ms=50, # Adjusted wait time
    )
    # <<< --- END OF CONFIGURATION --- >>>

    config.device = get_compute_device() # Determine device after config init

    # Verify source paths *after* defining config
    for cam_id, cam_cfg in config.cameras.items():
        if not cam_cfg.source_path.is_dir():
             # Log error but allow execution to proceed to maybe find other errors
             logger.error(f"Source path for camera '{cam_id}' NOT FOUND: {cam_cfg.source_path}")
             # raise FileNotFoundError(f"Source path for camera '{cam_id}' not found: {cam_cfg.source_path}")
        else:
             logger.info(f"Camera {cam_id} source verified: {cam_cfg.source_path}")

    logger.info("Configuration setup complete.")
    return config

def load_dataset_info(config: PipelineHandoffConfig) -> Tuple[List[str], Dict[CameraID, Path], PipelineHandoffConfig]:
    """Loads image filenames from the first camera and detects frame shapes for all."""
    logger.info("--- Loading Dataset Information ---")
    image_filenames = []
    camera_dirs: Dict[CameraID, Path] = {}
    valid_camera_configs = {}

    # Filter out cameras with non-existent source paths first
    for cam_id, cam_cfg in config.cameras.items():
         if cam_cfg.source_path.is_dir():
             camera_dirs[cam_id] = cam_cfg.source_path
             valid_camera_configs[cam_id] = cam_cfg
         else:
             logger.error(f"Excluding camera {cam_id} due to missing source path: {cam_cfg.source_path}")

    if not camera_dirs:
         raise RuntimeError("No valid camera source directories found. Check configuration and dataset paths.")

    # Use the first *valid* camera to get the list of filenames
    first_valid_cam_id = next(iter(camera_dirs))
    first_valid_cam_path = camera_dirs[first_valid_cam_id]
    logger.info(f"Using camera '{first_valid_cam_id}' at {first_valid_cam_path} to list frames.")

    try:
        # List only image files
        image_files = [f for f in first_valid_cam_path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        image_filenames = sorted_alphanumeric([f.name for f in image_files])
        if not image_filenames:
            raise ValueError(f"No valid image files found in {first_valid_cam_path}")
        logger.info(f"Found {len(image_filenames)} frames based on {first_valid_cam_id}.")
    except Exception as e:
        logger.critical(f"Failed list images from {first_valid_cam_path}: {e}", exc_info=True)
        raise

    # Auto-detect frame shapes using the first frame
    logger.info("Detecting frame shapes from first frame...")
    first_frame_batch = load_frames_for_batch(camera_dirs, image_filenames[0])
    for cam_id, frame in first_frame_batch.items():
        if cam_id in valid_camera_configs: # Only update valid configs
            if frame is not None:
                 config.cameras[cam_id].frame_shape = frame.shape[:2]
                 logger.info(f"  {cam_id}: Detected shape {frame.shape[:2]}")
            else:
                 logger.warning(f"Could not load first frame for {cam_id} to detect shape.")
                 # Keep frame_shape as None in config if detection fails
                 config.cameras[cam_id].frame_shape = None

    # Update config to only include cameras that were found and processed
    config.cameras = valid_camera_configs
    if not config.cameras:
         raise RuntimeError("No camera configurations remained after path validation.")
    logger.info(f"Processing with valid cameras: {list(config.cameras.keys())}")

    return image_filenames, camera_dirs, config

def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the specified image file for each camera directory."""
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename
        img: Optional[np.ndarray] = None # Explicitly type hint
        if image_path.is_file():
            try:
                # Use imdecode for better path handling, especially non-ASCII
                img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    # Don't log every frame, maybe just once per camera?
                    # logger.debug(f"[{cam_id}] Load failed (imdecode returned None): {image_path}")
                    img = None
            except Exception as e:
                logger.error(f"[{cam_id}] Read error {image_path}: {e}")
                img = None
        # else: # File not found, expected if sequences have different lengths
        #     pass
        current_frames[cam_id] = img
    return current_frames

def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines annotated frames into a grid display."""
    valid_annotated = {cid: f for cid, f in annotated_frames.items() if f is not None and f.size > 0}

    if not valid_annotated:
        # Handle case where no frames are available at all
        combined_display = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        # Determine grid size (prefer wider layout)
        num_cams = len(valid_annotated)
        if num_cams <= 1: cols = 1
        elif num_cams == 2: cols = 2
        else: cols = 2 # Default to 2xN for 3 or 4 cameras
        rows = int(np.ceil(num_cams / cols))

        # Get target size from the first available frame
        first_cam_id = next(iter(valid_annotated))
        target_h, target_w = valid_annotated[first_cam_id].shape[:2]

        # Prepare frames for tiling (resize if needed)
        frames_to_tile = []
        cam_order = sorted(valid_annotated.keys()) # Display in consistent order

        for cam_id in cam_order:
            frame = valid_annotated[cam_id]
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                try:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                except Exception as resize_err:
                    logger.warning(f"[{cam_id}] Grid resize error: {resize_err}. Using blank.")
                    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{cam_id} Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            frames_to_tile.append(frame)

        # Create the combined canvas
        combined_h = rows * target_h
        combined_w = cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        # Place frames onto the canvas
        frame_idx_display = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx_display < len(frames_to_tile):
                    try:
                        y_start, y_end = r * target_h, (r + 1) * target_h
                        x_start, x_end = c * target_w, (c + 1) * target_w
                        combined_display[y_start:y_end, x_start:x_end] = frames_to_tile[frame_idx_display]
                    except ValueError as slice_err:
                        logger.error(f"Grid placement error for frame {frame_idx_display}: {slice_err}.")
                    frame_idx_display += 1

        # --- Final Scaling ---
        disp_h, disp_w = combined_display.shape[:2]
        scale = 1.0
        if disp_w > max_width:
             scale = max_width / disp_w

        if scale != 1.0:
            try:
                disp_h_new, disp_w_new = int(disp_h * scale), int(disp_w * scale)
                combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err:
                logger.error(f"Final display resize failed: {final_resize_err}")

    # --- Show Image ---
    try:
         # Ensure window exists before showing
         # cv2.namedWindow ensures it's created once and properties are set.
         # Check visibility to see if user closed it.
         if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
              cv2.imshow(window_name, combined_display)
         else:
              # If window property check fails or returns < 1, window is likely closed.
              # Let the main loop handle breaking.
              pass
    except cv2.error:
         # This might happen if the window was never created or destroyed unexpectedly.
         # Let the main loop handle breaking.
         logger.debug(f"cv2.error likely means window '{window_name}' is not available.")


# --- Main Execution ---
def main():
    """Main execution function."""
    trigger_processor: Optional[HandoffTrigger] = None
    detections_last_batch: Dict[CameraID, List[Detection]] = defaultdict(list)
    window_name = "Handoff Trigger POC" # Define once

    try:
        config = setup_paths_and_config()
        image_filenames, camera_dirs, config = load_dataset_info(config) # Load info & detect shapes

        # Initialize the main processor class AFTER shapes are known
        trigger_processor = HandoffTrigger(config)

        # Create the display window initially
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        logger.info(f"Display window '{window_name}' created.")


        logger.info("--- Starting Frame Processing Loop ---")
        total_frames_loaded = 0
        loop_start_time = time.perf_counter()

        for frame_idx, current_filename in enumerate(image_filenames):
            iter_start_time = time.perf_counter()

            # --- Load Frames ---
            current_frames = load_frames_for_batch(camera_dirs, current_filename)
            # Check if *at least one* frame was loaded successfully
            if not any(f is not None for f in current_frames.values()):
                # Only log if it's an early frame, otherwise might be end of sequence
                if frame_idx < 10:
                     logger.warning(f"Frame {frame_idx}: No valid images loaded for any camera ('{current_filename}').")
                # Check if we should stop if *all* cameras finish
                # break # Or continue? Continue allows partial processing if some cameras have longer sequences
                continue # Skip processing if no frames loaded this iteration

            total_frames_loaded += 1

            # --- Process Batch (Detect & Check Exits) ---
            if trigger_processor:
                detections_last_batch = trigger_processor.process_frame_batch(current_frames)

                # --- Annotate and Display ---
                annotated_frames = trigger_processor.draw_results(current_frames, detections_last_batch)
                display_combined_frames(window_name, annotated_frames, config.max_display_width)
            else:
                 # Fallback display raw frames if processor failed init (shouldn't happen)
                 logger.error("Trigger processor not initialized!")
                 display_combined_frames(window_name, current_frames, config.max_display_width)

            # --- Logging and Timing ---
            iter_end_time = time.perf_counter()
            frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000
            current_loop_duration = iter_end_time - loop_start_time
            avg_display_fps = total_frames_loaded / current_loop_duration if current_loop_duration > 0 else 0

            if frame_idx < 5 or frame_idx % 10 == 0: # Log periodically
                 det_count = sum(len(dets) for dets in detections_last_batch.values())
                 logger.info(f"Frame {frame_idx:<5} Time: {frame_proc_time_ms:>7.1f}ms | AvgFPS: {avg_display_fps:5.2f} | Dets: {det_count:<3}")

            # --- User Input & Window Check ---
            key = cv2.waitKey(config.display_wait_ms) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed.")
                break
            elif key == ord('p'):
                logger.info("Paused. Press any key in the display window to resume.")
                cv2.waitKey(0)
                logger.info("Resuming.")

            # Check if window was closed after waitKey
            try:
                 if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                      logger.info("Display window closed by user.")
                      break
            except cv2.error:
                 logger.info("Display window seems to have been destroyed.")
                 break # Exit loop if window is gone

        # --- End of Loop ---
        loop_end_time = time.perf_counter()
        total_time = loop_end_time - loop_start_time
        logger.info("--- Processing Loop Finished ---")
        logger.info(f"Processed {total_frames_loaded} frame batches.")
        if total_frames_loaded > 0 and total_time > 0.01:
             final_avg_display_fps = total_frames_loaded / total_time
             logger.info(f"Total time: {total_time:.2f}s. Overall Avg Display FPS: {final_avg_display_fps:.2f}")
        else:
             logger.info("No frames processed or time too short for accurate FPS.")


    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e:
        logger.critical(f"Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        cv2.destroyAllWindows()
        # Add delay to help ensure windows close fully before script ends
        for _ in range(5): cv2.waitKey(1)

        if trigger_processor and trigger_processor.config.device.type != 'cpu':
             logger.info(f"Releasing GPU ({trigger_processor.config.device.type}) resources...")
             try:
                 # Explicitly delete models and clear cache
                 if hasattr(trigger_processor, 'detector'):
                     del trigger_processor.detector
                 del trigger_processor
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                     logger.info("CUDA cache cleared.")

             except Exception as clean_e:
                 logger.error(f"Error during GPU cleanup: {clean_e}")
        logger.info("Exiting script.")

if __name__ == "__main__":
    main()