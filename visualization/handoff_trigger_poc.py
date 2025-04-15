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

# Ensure torchvision.transforms.v2 is available or fallback if needed
try:
    from torchvision.transforms.v2 import Compose
    from torchvision.transforms.v2 import functional as F_v2
except ImportError:
    from torchvision.transforms import Compose
    from torchvision.transforms import functional as F_v1
from PIL import Image

# --- Basic Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("HandoffTriggerPOC")

# --- Type Aliases ---
CameraID = str
FrameData = Optional[np.ndarray]
BoundingBox = np.ndarray  # xyxy format [x1, y1, x2, y2]
Detection = Dict[str, Any]  # {'bbox_xyxy': BoundingBox, 'conf': float, 'class_id': int}
ExitDirection = str  # 'up', 'down', 'left', 'right' (represents rule trigger direction)
QuadrantName = str  # 'upper_left', 'upper_right', 'lower_left', 'lower_right'


# --- Configuration ---

@dataclass
class ExitRule:
    direction: ExitDirection  # Direction rule applies to (e.g., 'down', 'left')
    target_cam_id: str
    target_entry_area: str  # Descriptive: 'upper right', 'upper left' etc. (of target cam)
    notes: Optional[str] = None


@dataclass
class CameraHandoffConfig:
    id: CameraID
    source_path: Path  # Path to the 'rgb' folder for this camera
    frame_shape: Optional[Tuple[int, int]] = None  # (height, width), auto-detected
    exit_rules: List[ExitRule] = field(default_factory=list)


@dataclass
class PipelineHandoffConfig:
    cameras: Dict[CameraID, CameraHandoffConfig]
    possible_overlaps: Set[Tuple[str, str]] = field(
        default_factory=lambda: {("c09", "c16"), ("c09", "c13"), ("c12", "c13")})
    no_overlaps: Set[Tuple[str, str]] = field(default_factory=lambda: {("c12", "c09"), ("c12", "c16"), ("c13", "c16")})
    detection_confidence_threshold: float = 0.5
    person_class_id: int = 1
    detection_input_width: Optional[int] = 640
    use_amp: bool = True
    min_bbox_overlap_ratio_in_quadrant: float = 0.40  # Threshold for trigger
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    max_display_width: int = 1920
    display_wait_ms: int = 50


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
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA confirmed usable.")
            return device
        except Exception as e:
            logger.warning(f"CUDA detected but failed usability test ({e}). Checking other options.")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            try:
                device = torch.device("mps")
                logger.info("MPS (Apple Silicon GPU) available.")
                _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
                logger.info("MPS confirmed usable.")
                return device
            except Exception as e:
                logger.warning(f"MPS detected but failed usability test ({e}). Falling back to CPU.")
        else:
            logger.info("MPS backend not available for this PyTorch build.")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def normalize_overlap_set(overlap_set: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Ensures pairs are stored consistently."""
    normalized = set()
    for c1, c2 in overlap_set:
        normalized.add(tuple(sorted((c1, c2))))
    return normalized


# --- Detection Logic ---
def load_detector(device: torch.device) -> Tuple[FasterRCNN, Any]:
    """Loads the Faster R-CNN model and corresponding transforms."""
    logger.info("Loading Faster R-CNN detector (ResNet50 FPN)...")
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        model.to(device)
        model.eval()
        transforms = weights.transforms()
        logger.info("Faster R-CNN Detector loaded successfully.")
        return model, transforms
    except Exception as e:
        logger.critical(f"FATAL ERROR loading Faster R-CNN: {e}", exc_info=True)
        raise RuntimeError("Failed to load object detector") from e


def preprocess_frame_for_detection(
        frame_bgr: np.ndarray, transforms: Any, target_width: Optional[int], device: torch.device
) -> Tuple[Optional[torch.Tensor], float, float, Tuple[int, int]]:
    """Prepares a single frame for the detector, including optional resizing."""
    if frame_bgr is None or frame_bgr.size == 0: return None, 1.0, 1.0, (0, 0)
    original_h, original_w = frame_bgr.shape[:2]
    img_for_det = frame_bgr
    scale_x, scale_y = 1.0, 1.0
    if target_width and original_w > target_width:
        scale = target_width / original_w
        target_h = int(original_h * scale)
        try:
            img_for_det = cv2.resize(frame_bgr, (target_width, target_h), interpolation=cv2.INTER_LINEAR)
            scale_x = original_w / target_width
            scale_y = original_h / target_h
        except Exception as resize_err:
            logger.warning(f"Resize failed: {resize_err}. Using original.")
            img_for_det = frame_bgr;
            scale_x, scale_y = 1.0, 1.0
    try:
        img_rgb = cv2.cvtColor(img_for_det, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = transforms(img_pil)
        return input_tensor.to(device), scale_x, scale_y, (original_h, original_w)
    except Exception as transform_err:
        logger.error(f"Preprocessing failed: {transform_err}")
        return None, 1.0, 1.0, (original_h, original_w)


def detect_persons(
        detector: FasterRCNN, batch_input_tensors: List[torch.Tensor],
        batch_scale_factors: List[Tuple[float, float]], batch_original_shapes: List[Tuple[int, int]],
        batch_cam_ids: List[CameraID], config: PipelineHandoffConfig
) -> Dict[CameraID, List[Detection]]:
    """Performs batched detection and post-processes results."""
    detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)
    if not batch_input_tensors: return detections_per_camera
    all_predictions: List[Dict[str, torch.Tensor]] = []
    try:
        with torch.no_grad():
            use_amp_runtime = config.use_amp and config.device.type == 'cuda'
            with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                if isinstance(batch_input_tensors, torch.Tensor): batch_input_tensors = [batch_input_tensors]
                all_predictions = detector(batch_input_tensors)
    except Exception as e:
        logger.error(f"Detection inference failed: {e}", exc_info=False);
        return detections_per_camera
    if len(all_predictions) != len(batch_cam_ids):
        logger.error(f"Detection output mismatch: {len(all_predictions)} vs {len(batch_cam_ids)}")
        return detections_per_camera
    for i, prediction_dict in enumerate(all_predictions):
        cam_id, (scale_x, scale_y), (original_h, original_w) = batch_cam_ids[i], batch_scale_factors[i], \
        batch_original_shapes[i]
        try:
            pred_boxes, pred_labels, pred_scores = prediction_dict['boxes'].cpu().numpy(), prediction_dict[
                'labels'].cpu().numpy(), prediction_dict['scores'].cpu().numpy()
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if label == config.person_class_id and score >= config.detection_confidence_threshold:
                    x1, y1, x2, y2 = box
                    orig_x1, orig_y1 = np.clip(x1 * scale_x, 0, original_w - 1), np.clip(y1 * scale_y, 0,
                                                                                         original_h - 1)
                    orig_x2, orig_y2 = np.clip(x2 * scale_x, 0, original_w - 1), np.clip(y2 * scale_y, 0,
                                                                                         original_h - 1)
                    if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1:
                        bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32)
                        detections_per_camera[cam_id].append(
                            {'bbox_xyxy': bbox_orig, 'conf': float(score), 'class_id': int(label)})
        except Exception as postproc_err:
            logger.error(f"[{cam_id}] Postprocessing detection error: {postproc_err}")
    return detections_per_camera


# --- Exit Condition Logic ---
def check_exit_conditions_by_quadrant(
        detections: List[Detection], frame_shape: Tuple[int, int],
        exit_rules: List[ExitRule], min_bbox_overlap_ratio: float
) -> List[Tuple[BoundingBox, ExitRule]]:
    """ Checks if detections overlap significantly with quadrants associated with exit rules."""
    triggered_exits = []
    if not detections or not frame_shape or not exit_rules or frame_shape[0] <= 0 or frame_shape[1] <= 0:
        return triggered_exits
    H, W = frame_shape
    mid_x, mid_y = W // 2, H // 2

    quadrant_regions: Dict[QuadrantName, Tuple[int, int, int, int]] = {
        'upper_left': (0, 0, mid_x, mid_y), 'upper_right': (mid_x, 0, W, mid_y),
        'lower_left': (0, mid_y, mid_x, H), 'lower_right': (mid_x, mid_y, W, H),
    }

    # Map exit directions to relevant quadrants based on common sense/rules
    # This mapping is crucial and might need tuning based on camera angles
    direction_to_quadrants: Dict[ExitDirection, List[QuadrantName]] = {
        'up': ['upper_left', 'upper_right'],
        'down': ['lower_left', 'lower_right'],
        'left': ['upper_left', 'lower_left'],
        'right': ['upper_right', 'lower_right'],
        # Allow specific rules if needed, e.g., 'upper_left_diag': ['upper_left']
        # For now, stick to cardinal directions mentioned in rules.
    }

    processed_detections = set()
    for rule in exit_rules:
        relevant_quadrant_names = direction_to_quadrants.get(rule.direction, [])
        if not relevant_quadrant_names: continue

        for det_idx, det in enumerate(detections):
            if det_idx in processed_detections: continue
            bbox = det['bbox_xyxy']
            x1, y1, x2, y2 = map(int, bbox)
            bbox_w, bbox_h = x2 - x1, y2 - y1
            if bbox_w <= 0 or bbox_h <= 0: continue
            bbox_area = bbox_w * bbox_h

            total_intersection_area = 0
            for quad_name in relevant_quadrant_names:
                qx1, qy1, qx2, qy2 = quadrant_regions[quad_name]
                inter_x1, inter_y1 = max(x1, qx1), max(y1, qy1)
                inter_x2, inter_y2 = min(x2, qx2), min(y2, qy2)
                inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
                total_intersection_area += inter_w * inter_h

            if total_intersection_area / bbox_area >= min_bbox_overlap_ratio:
                triggered_exits.append((bbox, rule))
                processed_detections.add(det_idx)
    return triggered_exits


# --- Handoff Processor Class ---
class HandoffTrigger:
    """Handles detection and triggering handoff messages based on quadrant overlap."""

    def __init__(self, config: PipelineHandoffConfig):
        self.config = config
        self.config.possible_overlaps = normalize_overlap_set(config.possible_overlaps)
        self.config.no_overlaps = normalize_overlap_set(config.no_overlaps)
        self.detector, self.detector_transforms = load_detector(config.device)
        # Store only needed trigger info: List[Tuple[CameraID, BoundingBox, ExitRule]]
        self.exit_triggers_this_frame: List[Tuple[CameraID, BoundingBox, ExitRule]] = []

    def process_frame_batch(self, frames: Dict[CameraID, FrameData]) -> Dict[CameraID, List[Detection]]:
        """Processes a batch of frames: detects persons and checks for exit triggers."""
        self.exit_triggers_this_frame.clear()
        batch_input_tensors, batch_scale_factors, batch_original_shapes, batch_cam_ids, valid_cam_ids = [], [], [], [], []

        for cam_id, frame in frames.items():
            if cam_id not in self.config.cameras: continue
            if frame is not None:
                input_tensor, scale_x, scale_y, orig_shape = preprocess_frame_for_detection(
                    frame, self.detector_transforms, self.config.detection_input_width, self.config.device
                )
                if input_tensor is not None:
                    batch_input_tensors.append(input_tensor)
                    batch_scale_factors.append((scale_x, scale_y));
                    batch_original_shapes.append(orig_shape)
                    batch_cam_ids.append(cam_id);
                    valid_cam_ids.append(cam_id)

        detections_per_camera = defaultdict(list)
        if batch_input_tensors:
            detections_per_camera = detect_persons(
                self.detector, batch_input_tensors, batch_scale_factors,
                batch_original_shapes, batch_cam_ids, self.config
            )

        for cam_id in valid_cam_ids:
            detections = detections_per_camera.get(cam_id, [])
            cam_config = self.config.cameras.get(cam_id)
            if not cam_config or not cam_config.frame_shape or not cam_config.exit_rules: continue

            triggered_exits = check_exit_conditions_by_quadrant(
                detections, cam_config.frame_shape, cam_config.exit_rules,
                self.config.min_bbox_overlap_ratio_in_quadrant
            )

            for bbox, matched_rule in triggered_exits:
                self.exit_triggers_this_frame.append((cam_id, bbox, matched_rule))  # Store trigger

                # Log detailed info to console
                target_cam = matched_rule.target_cam_id
                relevant_cams_set = {target_cam}
                for c1, c2 in self.config.possible_overlaps:
                    if c1 == target_cam: relevant_cams_set.add(c2)
                    if c2 == target_cam: relevant_cams_set.add(c1)
                log_msg = (
                    f"EXIT TRIGGER: Cam [{cam_id}] Person at [{bbox.astype(int)}] matches rule "
                    f"[{matched_rule.direction}]. Potential entry: Cam [{target_cam}] "
                    f"Area [{matched_rule.target_entry_area}]. "
                    f"Cameras to check: {sorted(list(relevant_cams_set))}"
                )
                if matched_rule.notes: log_msg += f" (Notes: {matched_rule.notes})"
                logger.info(log_msg)

        return detections_per_camera

    def draw_results(self, frames: Dict[CameraID, FrameData], detections_per_camera: Dict[CameraID, List[Detection]]) -> \
    Dict[CameraID, FrameData]:
        """Draws detections, quadrant lines, and highlights triggered exits with source->target text."""
        annotated_frames = {}
        default_h, default_w = 1080, 1920
        first_valid_frame = next((f for f in frames.values() if f is not None), None)
        if first_valid_frame is not None: default_h, default_w = first_valid_frame.shape[:2]

        quadrant_line_color = (128, 128, 128);
        detection_color = (0, 255, 0)
        trigger_box_color = (0, 0, 255);
        trigger_text_color = (255, 255, 0)  # Cyan text

        # Create a map for quick lookup of triggers by bbox per camera
        trigger_map: Dict[CameraID, Dict[Tuple, ExitRule]] = defaultdict(dict)
        for cam_id, bbox, rule in self.exit_triggers_this_frame:
            bbox_key = tuple(bbox.astype(int))  # Use tuple of int coords as key
            trigger_map[cam_id][bbox_key] = rule  # Store the rule that triggered this box

        for cam_id, frame in frames.items():
            cam_config = self.config.cameras.get(cam_id)
            H, W = (cam_config.frame_shape if cam_config and cam_config.frame_shape else (default_h, default_w))
            if H <= 0 or W <= 0: H, W = default_h, default_w  # Safety check

            if frame is None:
                annotated_frame = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(annotated_frame, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            trigger_box_color, 2)
                annotated_frames[cam_id] = annotated_frame
                continue

            annotated_frame = frame.copy()
            mid_x, mid_y = W // 2, H // 2
            cv2.line(annotated_frame, (mid_x, 0), (mid_x, H), quadrant_line_color, 1)
            cv2.line(annotated_frame, (0, mid_y), (W, mid_y), quadrant_line_color, 1)

            # Draw Detections and Trigger Highlights
            for det in detections_per_camera.get(cam_id, []):
                x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
                conf = det['conf']
                bbox_key = (x1, y1, x2, y2)

                # Check if this detection triggered an exit rule
                triggered_rule = trigger_map.get(cam_id, {}).get(bbox_key)

                box_color = trigger_box_color if triggered_rule else detection_color
                thickness = 3 if triggered_rule else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)

                # Add standard detection text
                label = f"P {conf:.2f}"
                text_y_det = y1 - 10 if y1 > 20 else y2 + 15
                cv2.putText(annotated_frame, label, (x1, text_y_det), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1,
                            cv2.LINE_AA)

                # Add trigger text if applicable
                if triggered_rule:
                    target_text = f"{cam_id} -> {triggered_rule.target_cam_id}"
                    # Position trigger text above the standard text
                    text_y_trig = text_y_det - 15 if text_y_det == y1 - 10 else text_y_det + 15
                    cv2.putText(annotated_frame, target_text, (x1, text_y_trig),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, trigger_text_color, 2)

            annotated_frames[cam_id] = annotated_frame
        return annotated_frames


# --- Main Execution Functions ---
def setup_paths_and_config() -> PipelineHandoffConfig:
    """Sets up the main configuration object based on dataset structure."""
    logger.info("--- Setting up Configuration and Paths ---")
    try:
        dataset_base_str = os.getenv("MTMMC_PATH", None)
        if dataset_base_str:
            dataset_base = Path(dataset_base_str); logger.info(f"Using MTMMC_PATH: {dataset_base}")
        else:
            default_path = "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"; dataset_base = Path(
                default_path); logger.info(f"Using default path: {dataset_base}")
        if not dataset_base.is_dir(): logger.error(f"Dataset base path NOT FOUND: {dataset_base}")
    except Exception as e:
        logger.error(f"Error determining dataset base path: {e}"); raise

    scene = "s10"  # CHANGE SCENE AS NEEDED
    base_scene_path = dataset_base / "train" / "train" / scene
    if not base_scene_path.is_dir():
        alt_paths = [dataset_base / "train" / scene, dataset_base / scene]
        found_alt = False
        for alt_path in alt_paths:
            if alt_path.is_dir(): base_scene_path = alt_path; logger.warning(
                f"Using alternative scene path: {base_scene_path}"); found_alt = True; break
        if not found_alt: raise FileNotFoundError(
            f"Scene '{scene}' not found in expected locations within {dataset_base}")

    # <<< --- DEFINE BIDIRECTIONAL RULES HERE --- >>>
    config = PipelineHandoffConfig(
        cameras={
            "c09": CameraHandoffConfig(
                id="c09", source_path=base_scene_path / "c09" / "rgb",
                exit_rules=[
                    # c09 Down -> c13 Upper Right
                    ExitRule(direction='down', target_cam_id='c13', target_entry_area='upper right',
                             notes='wait; overlap c13/c16 possible'),
                ]
            ),
            "c12": CameraHandoffConfig(
                id="c12", source_path=base_scene_path / "c12" / "rgb",
                exit_rules=[
                    # c12 Upper Left -> c13 Upper Left (Assume 'Upper Left' exit maps to 'left' direction check)
                    ExitRule(direction='left', target_cam_id='c13', target_entry_area='upper left',
                             notes='overlap c13 possible'),
                ]
            ),
            "c13": CameraHandoffConfig(
                id="c13", source_path=base_scene_path / "c13" / "rgb",
                exit_rules=[
                    # c13 Upper Right -> c09 Down (Assume 'Upper Right' exit maps to 'right' direction check)
                    ExitRule(direction='right', target_cam_id='c09', target_entry_area='down',
                             notes='wait; overlap c09 possible'),
                    # c13 Upper Left -> c12 Upper Left (Assume 'Upper Left' exit maps to 'left' direction check)
                    ExitRule(direction='left', target_cam_id='c12', target_entry_area='upper left',
                             notes='overlap c12 possible'),
                ]
            ),
            "c16": CameraHandoffConfig(
                id="c16", source_path=base_scene_path / "c16" / "rgb", exit_rules=[]
            ),
        },
        possible_overlaps={("c09", "c16"), ("c09", "c13"), ("c12", "c13")},
        no_overlaps={("c12", "c09"), ("c12", "c16"), ("c13", "c16")},
        detection_confidence_threshold=0.5, person_class_id=1,
        detection_input_width=640, use_amp=True,
        min_bbox_overlap_ratio_in_quadrant=0.40,
        max_display_width=1920, display_wait_ms=50,
    )
    # <<< --- END OF CONFIGURATION --- >>>

    config.device = get_compute_device()
    valid_cameras = {}
    for cam_id, cam_cfg in config.cameras.items():
        if cam_cfg.source_path.is_dir():
            valid_cameras[cam_id] = cam_cfg; logger.info(f"Cam {cam_id}: {cam_cfg.source_path} [OK]")
        else:
            logger.error(f"Cam {cam_id}: {cam_cfg.source_path} [NOT FOUND] - Excluding.")
    if not valid_cameras: raise RuntimeError("No valid camera source directories found. Check config.")
    config.cameras = valid_cameras  # Update config to only include valid cameras
    logger.info("Configuration setup complete.")
    return config


def load_dataset_info(config: PipelineHandoffConfig) -> Tuple[List[str], Dict[CameraID, Path], PipelineHandoffConfig]:
    """Loads image filenames from the first valid camera and detects frame shapes."""
    logger.info("--- Loading Dataset Information ---")
    camera_dirs: Dict[CameraID, Path] = {cam_id: cam.source_path for cam_id, cam in config.cameras.items()}
    first_valid_cam_id = next(iter(camera_dirs))
    first_valid_cam_path = camera_dirs[first_valid_cam_id]
    logger.info(f"Using camera '{first_valid_cam_id}' at {first_valid_cam_path} to list frames.")
    try:
        image_files = [f for f in first_valid_cam_path.iterdir() if
                       f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        image_filenames = sorted_alphanumeric([f.name for f in image_files])
        if not image_filenames: raise ValueError(f"No valid image files found in {first_valid_cam_path}")
        logger.info(f"Found {len(image_filenames)} frames based on {first_valid_cam_id}.")
    except Exception as e:
        logger.critical(f"Failed list images from {first_valid_cam_path}: {e}", exc_info=True); raise
    logger.info("Detecting frame shapes from first frame...")
    first_frame_batch = load_frames_for_batch(camera_dirs, image_filenames[0])
    for cam_id, frame in first_frame_batch.items():
        if cam_id in config.cameras:
            config.cameras[cam_id].frame_shape = frame.shape[:2] if frame is not None else None
            status = f"Detected shape {frame.shape[:2]}" if frame is not None else "Frame load failed, shape=None"
            logger.info(f"  {cam_id}: {status}")
    return image_filenames, camera_dirs, config


def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the specified image file for each camera directory."""
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename
        img: Optional[np.ndarray] = None
        if image_path.is_file():
            try:
                img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None or img.size == 0: img = None
            except Exception as e:
                logger.error(f"[{cam_id}] Read error {image_path}: {e}"); img = None
        current_frames[cam_id] = img
    return current_frames


def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines annotated frames into a grid display."""
    valid_annotated = {cid: f for cid, f in annotated_frames.items() if f is not None and f.size > 0}
    if not valid_annotated:
        try:  # Attempt to clear the window or show placeholder
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                placeholder = np.zeros((200, 300, 3), dtype=np.uint8);
                cv2.putText(placeholder, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow(window_name, placeholder)
        except cv2.error:
            pass
        return

    num_cams = len(valid_annotated)
    cols = 2 if num_cams > 1 else 1  # Default to 2 columns if more than 1 cam
    rows = int(np.ceil(num_cams / cols))
    first_cam_id = next(iter(valid_annotated))
    target_h, target_w = valid_annotated[first_cam_id].shape[:2]
    if target_h <= 0 or target_w <= 0: logger.error("Invalid target dimensions."); return

    frames_to_tile = []
    cam_order = sorted(valid_annotated.keys())
    for cam_id in cam_order:
        frame = valid_annotated[cam_id]
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            try:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            except Exception:
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8); cv2.putText(frame, f"{cam_id} Err", (10, 30),
                                                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                                                       (0, 0, 255), 1)
        frames_to_tile.append(frame)

    combined_h, combined_w = rows * target_h, cols * target_w
    if combined_h <= 0 or combined_w <= 0: logger.error(
        f"Invalid combined dimensions: {combined_w}x{combined_h}"); return
    combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(frames_to_tile):
                try:
                    combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = frames_to_tile[
                        idx]
                except ValueError as e:
                    logger.error(f"Grid placement error: {e}")
                idx += 1

    disp_h, disp_w = combined_display.shape[:2]
    scale = min(1.0, max_width / disp_w) if disp_w > 0 else 1.0
    if abs(scale - 1.0) > 1e-3:
        try:
            disp_h_new, disp_w_new = int(disp_h * scale), int(disp_w * scale)
            if disp_h_new > 0 and disp_w_new > 0: combined_display = cv2.resize(combined_display,
                                                                                (disp_w_new, disp_h_new),
                                                                                interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"Final display resize failed: {e}")

    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1: cv2.imshow(window_name, combined_display)
    except cv2.error:
        pass


# --- Main Execution ---
def main():
    """Main execution function."""
    trigger_processor: Optional[HandoffTrigger] = None
    detections_last_batch: Dict[CameraID, List[Detection]] = defaultdict(list)
    window_name = "Handoff Trigger POC (Quadrant, Bidirectional Rules)"
    is_paused = False

    try:
        config = setup_paths_and_config()
        image_filenames, camera_dirs, config = load_dataset_info(config)
        trigger_processor = HandoffTrigger(config)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        logger.info(f"Display window '{window_name}' created. Press 'p' to pause/resume, 'q' to quit.")
        logger.info("--- Starting Frame Processing Loop ---")
        total_frames_loaded, frame_idx = 0, 0
        loop_start_time = time.perf_counter()

        while frame_idx < len(image_filenames):
            current_filename = image_filenames[frame_idx]
            iter_start_time = time.perf_counter()

            if not is_paused:
                current_frames = load_frames_for_batch(camera_dirs, current_filename)
                if not any(f is not None for f in current_frames.values()):
                    if frame_idx < 10: logger.warning(f"Frame {frame_idx}: No images loaded for '{current_filename}'.")
                    frame_idx += 1;
                    continue
                total_frames_loaded += 1

                if trigger_processor:
                    detections_last_batch = trigger_processor.process_frame_batch(current_frames)
                    annotated_frames = trigger_processor.draw_results(current_frames, detections_last_batch)
                    display_combined_frames(window_name, annotated_frames, config.max_display_width)
                else:
                    logger.error("Trigger processor missing!");
                    display_combined_frames(window_name, current_frames, config.max_display_width)

                iter_end_time = time.perf_counter()
                if frame_idx < 5 or frame_idx % 10 == 0:
                    frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000
                    current_loop_duration = iter_end_time - loop_start_time
                    avg_display_fps = total_frames_loaded / current_loop_duration if current_loop_duration > 0 else 0
                    det_count = sum(len(dets) for dets in detections_last_batch.values())
                    logger.info(
                        f"Frame {frame_idx:<5} Time: {frame_proc_time_ms:>7.1f}ms | AvgFPS: {avg_display_fps:5.2f} | Dets: {det_count:<3}")

                frame_idx += 1  # Advance frame only if not paused

            wait_duration = config.display_wait_ms if not is_paused else 1
            key = cv2.waitKey(wait_duration) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed."); break
            elif key == ord('p'):
                is_paused = not is_paused
                logger.info("<<<< PAUSED >>>> Press 'p' to resume." if is_paused else ">>>> RESUMED >>>>")

            try:  # Check window status
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: logger.info(
                    "Display window closed."); break
            except cv2.error:
                logger.info("Display window error."); break  # Break if window error occurs

        loop_end_time = time.perf_counter();
        total_time = loop_end_time - loop_start_time
        logger.info(
            f"--- Processing Loop Finished (Processed {total_frames_loaded} frames up to index {frame_idx - 1}) ---")
        if total_frames_loaded > 0 and total_time > 0.01: logger.info(
            f"Total time: {total_time:.2f}s. Overall Avg Display FPS: {total_frames_loaded / total_time:.2f}")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e:
        logger.critical(f"Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        cv2.destroyAllWindows();
        for _ in range(5): cv2.waitKey(1)  # Helps ensure window closes
        if trigger_processor and hasattr(trigger_processor, 'config') and trigger_processor.config.device.type != 'cpu':
            logger.info(f"Releasing GPU ({trigger_processor.config.device.type}) resources...")
            try:
                if hasattr(trigger_processor, 'detector'): del trigger_processor.detector
                del trigger_processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); logger.info("CUDA cache cleared.")
                elif hasattr(torch.backends, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache(); logger.info("MPS cache cleared.")  # If available
                else:
                    logger.info("GPU resources released (managed by system).")
            except Exception as clean_e:
                logger.error(f"Error during GPU cleanup: {clean_e}")
        logger.info("Exiting script.")


if __name__ == "__main__":
    main()
