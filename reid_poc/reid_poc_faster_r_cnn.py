# -*- coding: utf-8 -*-
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, NamedTuple

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN

# Ensure torchvision.transforms.v2 is available or fallback if needed
try:
    from torchvision.transforms.v2 import Compose
except ImportError:
    from torchvision.transforms import Compose  # Fallback for older torchvision
from PIL import Image
from scipy.spatial.distance import cosine as cosine_distance

# --- BoxMOT Imports ---
try:
    from boxmot import create_tracker
    import boxmot as boxmot_root_module

    BOXMOT_PATH = Path(boxmot_root_module.__file__).parent
except ImportError as e:
    logging.critical(f"Failed to import boxmot. Is it installed? Error: {e}")
    sys.exit(1)

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.trackers.basetracker import BaseTracker

# --- End BoxMOT Imports ---

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


# --- Configuration ---
@dataclass
class PipelineConfig:
    """Configuration settings for the multi-camera pipeline."""
    # Paths - Use Path objects for cross-platform compatibility
    dataset_base_path: Path = Path(
        os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"))
    reid_model_weights: Path = Path("osnet_x0_25_msmt17.pt")
    tracker_config_path: Optional[Path] = None  # Set dynamically

    # Dataset Selection
    selected_scene: str = "s10"
    selected_cameras: List[str] = field(default_factory=lambda: ["c09", "c12", "c13", "c16"])

    # Model Parameters
    person_class_id: int = 1
    detection_confidence_threshold: float = 0.5
    reid_similarity_threshold: float = 0.65
    gallery_ema_alpha: float = 0.9  # Weighting for historical embedding in EMA update
    reid_refresh_interval_frames: int = 10

    # --- Performance Optimizations ---
    detection_input_width: Optional[int] = 640  # 1920
    use_amp: bool = True  # Use Automatic Mixed Precision (FP16) for detection on CUDA GPUs

    # Tracker Type
    tracker_type: str = 'bytetrack'

    # Execution
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Visualization
    draw_bounding_boxes: bool = True
    show_track_id: bool = True
    show_global_id: bool = True
    window_name: str = "Multi-Camera Tracking & Re-ID"
    display_wait_ms: int = 1
    max_display_width: int = 1920


# --- Type Aliases and Structures ---
CameraID = str
TrackID = int  # Use Python's native int for type hinting keys
GlobalID = int
TrackKey = Tuple[CameraID, TrackID]
FeatureVector = np.ndarray
BoundingBox = np.ndarray  # xyxy format
Detection = Dict[
    str, Any]  # {'bbox_xyxy': BoundingBox, 'conf': float, 'class_id': int} # Coordinates are relative to ORIGINAL frame size
TrackData = Dict[str, Any]  # {'bbox_xyxy': BoundingBox, 'track_id': TrackID, 'global_id': Optional[GlobalID], ...}
FrameData = Optional[np.ndarray]
Timings = Dict[str, float]
ScaleFactors = Tuple[float, float]  # (scale_x, scale_y) -> multiply detected coords by these to get original coords


class ProcessedBatchResult(NamedTuple):
    """Results structure for a processed batch of frames."""
    results_per_camera: Dict[CameraID, List[TrackData]]
    timings: Timings


# --- Helper Functions --- (Identical to previous version, omitted for brevity)
def sorted_alphanumeric(data: List[str]) -> List[str]:
    """Sorts a list of strings alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def get_compute_device() -> torch.device:
    """Selects and returns the best available compute device."""
    logger.info("--- Determining Compute Device ---")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"Attempting to use CUDA device: {torch.cuda.get_device_name(device)}")
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA device confirmed.")
            return device
        except Exception as e:
            logger.warning(f"CUDA reported available, but test failed ({e}). Falling back...")
    # Add MPS check here if needed
    logger.info("Using CPU device.")
    return torch.device("cpu")


def calculate_cosine_similarity(feat1: Optional[FeatureVector], feat2: Optional[FeatureVector]) -> float:
    """Calculates cosine similarity between two feature vectors, handling None and invalid inputs."""
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten();
    feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if not np.any(feat1) or not np.any(feat2): return 0.0  # Check if either is all zeros
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0
    try:
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0)
        return float(np.clip(similarity, 0.0, 1.0))
    except ValueError as e:
        logger.error(f"Cosine distance error (likely shape mismatch {feat1.shape} vs {feat2.shape}): {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error calculating cosine distance: {e}")
        return 0.0


def _get_reid_device_specifier_string(device: torch.device) -> str:
    """Determines the device string specifier needed by ReidAutoBackend."""
    if device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        return str(idx)
    elif device.type == 'mps':
        return 'mps'
    return 'cpu'


def _normalize_embedding(embedding: FeatureVector) -> FeatureVector:
    """Normalizes a feature vector using L2 norm."""
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-6)  # Add epsilon for numerical stability


# --- Main Processing Class ---

class MultiCameraPipeline:
    """Handles multi-camera detection, tracking, and Re-Identification."""

    def __init__(self, config: PipelineConfig):
        """Initializes models, trackers, and state."""
        self.config = config
        self.device = config.device
        self.camera_ids = config.selected_cameras
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids} on device: {self.device}")
        if config.detection_input_width:
            logger.info(f"Detection resizing ENABLED to width: {config.detection_input_width}")
        else:
            logger.info("Detection resizing DISABLED.")
        if config.use_amp and self.device.type == 'cuda':
            logger.info("AMP (FP16) for detection ENABLED.")
        elif config.use_amp:
            logger.warning("AMP requested but device is not CUDA. AMP will be disabled.")
        else:
            logger.info("AMP (FP16) for detection DISABLED.")

        self.detector, self.detector_transforms = self._load_detector()
        self.reid_model = self._load_reid_model()
        self.trackers = self._initialize_trackers()

        # State Management
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.next_global_id: GlobalID = 1
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        self.track_last_reid_frame: Dict[TrackKey, int] = {}

    def _load_detector(self) -> Tuple[FasterRCNN, Compose]:
        """Loads the Faster R-CNN object detector model."""
        logger.info("Loading SHARED Faster R-CNN detector...")
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            model.to(self.device)
            model.eval()
            # Important: Get transforms AFTER model is on device? Usually not needed, but check if issues arise.
            transforms = weights.transforms()
            logger.info("Faster R-CNN Detector loaded successfully.")
            return model, transforms
        except Exception as e:
            logger.critical(f"FATAL ERROR loading Faster R-CNN: {e}", exc_info=True)
            raise RuntimeError("Failed to load object detector") from e

    def _load_reid_model(self) -> Optional[BaseModelBackend]:
        """Loads the OSNet Re-Identification model."""
        reid_weights = self.config.reid_model_weights
        logger.info(f"Loading SHARED OSNet ReID model from: {reid_weights}")
        if not reid_weights.is_file():
            logger.critical(f"FATAL ERROR: ReID weights file not found at {reid_weights}")
            raise FileNotFoundError(f"ReID weights not found: {reid_weights}")
        try:
            reid_device_specifier = _get_reid_device_specifier_string(self.device)
            # ReidAutoBackend might have its own half precision logic, set half=False here
            # as we control detection precision separately. Check BoxMOT docs if needed.
            reid_model_handler = ReidAutoBackend(
                weights=reid_weights, device=reid_device_specifier, half=False
            )
            model = reid_model_handler.model
            if hasattr(model, "warmup"): model.warmup()
            logger.info("OSNet ReID Model loaded successfully.")
            return model
        except Exception as e:
            logger.critical(f"FATAL ERROR loading ReID model: {e}", exc_info=True)
            raise RuntimeError("Failed to load ReID model") from e

    def _initialize_trackers(self) -> Dict[CameraID, BaseTracker]:
        """Initializes a tracker instance for each camera."""
        logger.info(f"Initializing {self.config.tracker_type} trackers...")
        trackers: Dict[CameraID, BaseTracker] = {}
        tracker_config_path = self.config.tracker_config_path
        if not tracker_config_path or not tracker_config_path.is_file():
            logger.critical(f"FATAL ERROR: Tracker config not found: {tracker_config_path}")
            raise FileNotFoundError(f"Tracker config not found: {tracker_config_path}")
        try:
            tracker_device_str = _get_reid_device_specifier_string(self.device)
            for cam_id in self.camera_ids:
                # Ensure tracker doesn't use its own ReID or half precision unless intended
                tracker_instance = create_tracker(
                    tracker_type=self.config.tracker_type,
                    tracker_config=str(tracker_config_path),
                    reid_weights=None,  # We handle ReID separately
                    device=tracker_device_str,
                    half=False,  # Trackers might use half precision internally, control if needed
                    per_class=False
                )
                if hasattr(tracker_instance, 'reset'): tracker_instance.reset()
                trackers[cam_id] = tracker_instance
                logger.info(
                    f"Initialized {self.config.tracker_type} for camera {cam_id} on device '{tracker_device_str}'")
            logger.info(f"Initialized {len(trackers)} tracker instances.")
            return trackers
        except Exception as e:
            logger.critical(f"FATAL ERROR initializing trackers: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize trackers") from e

    # REMOVED _detect_persons as it's now integrated into process_frame_batch

    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[
        TrackID, FeatureVector]:
        """Extracts Re-ID features for a given set of tracked detections."""
        features: Dict[TrackID, FeatureVector] = {}  # Hint uses TrackID=int
        if self.reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.shape[0] == 0:
            return features

        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids = tracked_dets_np[:, 4]  # Keep as numpy array initially

        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4:
            logger.warning(f"Invalid bbox shape for FE. Shape: {bboxes_xyxy.shape}. Skipping.")
            return features

        try:
            # BoxMOT expects BGR frame and XYXY boxes relative to that frame
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)
            if batch_features is not None and len(batch_features) == len(track_ids):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        # *** Explicitly cast track ID to Python int for dictionary key ***
                        current_track_id: TrackID = int(track_ids[i])
                        features[current_track_id] = det_feature  # Store raw feature
            # else: logger.warning(f"FE mismatch/empty. Expected {len(track_ids)}, got {len(batch_features) if batch_features is not None else 'None'}.")
        except Exception as e:
            logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features

    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[
        TrackKey, Optional[GlobalID]]:
        """Compares features of triggered tracks against the gallery to assign Global IDs."""
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {}
        if not features_per_track: return newly_assigned_global_ids

        valid_gallery_items = [(gid, emb) for gid, emb in self.reid_gallery.items() if
                               emb is not None and np.isfinite(emb).all() and emb.size > 0]
        if not valid_gallery_items:
            valid_gallery_ids, valid_gallery_embeddings = [], []
        else:
            valid_gallery_ids, valid_gallery_embeddings = zip(*valid_gallery_items)

        for track_key, new_embedding in features_per_track.items():
            newly_assigned_global_ids[track_key] = None
            if new_embedding is None or not np.isfinite(new_embedding).all() or new_embedding.size == 0:
                logger.warning(f"Skipping ReID for {track_key} due to invalid new embedding.")
                continue

            best_match_global_id: Optional[GlobalID] = None
            if valid_gallery_ids:
                try:
                    similarities = np.array(
                        [calculate_cosine_similarity(new_embedding, gal_emb) for gal_emb in valid_gallery_embeddings])
                    max_similarity_idx = np.argmax(similarities)
                    max_similarity = similarities[max_similarity_idx]
                    if max_similarity >= self.config.reid_similarity_threshold:
                        best_match_global_id = valid_gallery_ids[max_similarity_idx]
                except Exception as sim_err:
                    logger.error(f"Similarity calculation error for {track_key}: {sim_err}")

            assigned_global_id: Optional[GlobalID] = None
            normalized_new_embedding = _normalize_embedding(new_embedding)

            if best_match_global_id is not None:
                # Case A: Matched an existing Global ID
                assigned_global_id = best_match_global_id
                current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                if current_gallery_emb is not None:
                    updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb +
                                         (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                    self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                else:  # Defensive
                    logger.warning(f"Gallery embedding for matched GID {assigned_global_id} None. Overwriting.")
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding
            else:
                # Case B: No match found
                last_known_global_id = self.track_to_global_id.get(track_key)
                if last_known_global_id is not None and last_known_global_id in self.reid_gallery:
                    # Case B.1: Re-assign previous ID & update gallery
                    assigned_global_id = last_known_global_id
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb +
                                             (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                        self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                    else:  # Defensive
                        logger.warning(f"Gallery embedding for re-assigned GID {assigned_global_id} None. Overwriting.")
                        self.reid_gallery[assigned_global_id] = normalized_new_embedding
                else:
                    # Case B.2: Assign new Global ID
                    assigned_global_id = self.next_global_id
                    self.next_global_id += 1
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding

            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id  # Update main mapping

        return newly_assigned_global_ids

    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """Updates the 'last seen' state and removes stale entries from state dictionaries."""
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys: new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_reid: del self.track_last_reid_frame[key]

        keys_to_delete_global = set(self.track_to_global_id.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_global: del self.track_to_global_id[key]
        # Optional gallery cleanup can be added here

    def process_frame_batch(self, frames: Dict[CameraID, FrameData], frame_idx: int) -> ProcessedBatchResult:
        """Processes a batch of frames: [Resize] -> Preprocess -> Batch Detect -> Scale Boxes -> Track -> Conditionally Re-ID -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)

        # --- Stage 1a: Preprocess Frames for Batch Detection ---
        t_prep_start = time.time()
        batch_input_tensors: List[torch.Tensor] = []
        batch_cam_ids: List[CameraID] = []
        batch_original_shapes: List[Tuple[int, int]] = []
        batch_scale_factors: List[ScaleFactors] = []  # (scale_x, scale_y) to map back

        for cam_id, frame_bgr in frames.items():
            if frame_bgr is not None and frame_bgr.size > 0:
                original_h, original_w = frame_bgr.shape[:2]
                frame_for_det = frame_bgr  # Start with original
                scale_x, scale_y = 1.0, 1.0  # Default scale factors

                # --- Resizing (Optional) ---
                if self.config.detection_input_width and original_w > self.config.detection_input_width:
                    target_w = self.config.detection_input_width
                    scale = target_w / original_w
                    target_h = int(original_h * scale)
                    try:
                        frame_for_det = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        # Store factors to scale detections back up: orig_coord = detected_coord * scale_factor
                        scale_x = original_w / target_w
                        scale_y = original_h / target_h
                    except Exception as resize_err:
                        logger.warning(f"[{cam_id}] Resizing failed: {resize_err}. Using original frame.")
                        frame_for_det = frame_bgr  # Fallback to original
                        scale_x, scale_y = 1.0, 1.0

                # --- Preprocessing ---
                try:
                    img_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    # Apply detector's required transforms (might include normalization, tensor conversion)
                    input_tensor = self.detector_transforms(img_pil)

                    batch_input_tensors.append(input_tensor.to(self.device))
                    batch_cam_ids.append(cam_id)
                    batch_original_shapes.append((original_h, original_w))
                    batch_scale_factors.append((scale_x, scale_y))
                except Exception as transform_err:
                    logger.error(f"[{cam_id}] Preprocessing/Transform failed: {transform_err}")

            # else: Frame is None or empty, implicitly skipped

        timings['preprocess'] = time.time() - t_prep_start

        # --- Stage 1b: Batched Detection ---
        t_detect_start = time.time()
        all_predictions: List[Dict[str, torch.Tensor]] = []
        if batch_input_tensors:  # Only run inference if there's something to process
            try:
                with torch.no_grad():
                    # Enable AMP only if configured AND on CUDA
                    use_amp_runtime = self.config.use_amp and self.device.type == 'cuda'
                    with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                        all_predictions = self.detector(batch_input_tensors)
            except Exception as e:
                logger.error(f"Batched detection inference failed: {e}", exc_info=False)
        timings['detection_batched'] = time.time() - t_detect_start

        # --- Stage 1c: Postprocess Detections & Scale Boxes ---
        t_postproc_start = time.time()
        detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(
            list)  # Stores detections scaled to ORIGINAL frame size

        if len(all_predictions) == len(batch_cam_ids):  # Check if prediction count matches input count
            for i, prediction_dict in enumerate(all_predictions):
                cam_id = batch_cam_ids[i]
                original_h, original_w = batch_original_shapes[i]
                scale_x, scale_y = batch_scale_factors[i]

                try:
                    pred_boxes = prediction_dict['boxes'].cpu().numpy()
                    pred_labels = prediction_dict['labels'].cpu().numpy()
                    pred_scores = prediction_dict['scores'].cpu().numpy()

                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        if label == self.config.person_class_id and score >= self.config.detection_confidence_threshold:
                            # Box coordinates are relative to the RESIZED input tensor
                            x1, y1, x2, y2 = box

                            # --- Scale boxes back to ORIGINAL frame coordinates ---
                            orig_x1 = max(0.0, x1 * scale_x)
                            orig_y1 = max(0.0, y1 * scale_y)
                            orig_x2 = min(float(original_w - 1), x2 * scale_x)
                            orig_y2 = min(float(original_h - 1), y2 * scale_y)

                            # Basic validity check on scaled coordinates
                            if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1:
                                detections_per_camera[cam_id].append({
                                    'bbox_xyxy': np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32),
                                    'conf': float(score),
                                    'class_id': self.config.person_class_id
                                })
                except Exception as postproc_err:
                    logger.error(f"[{cam_id}] Error postprocessing/scaling detections: {postproc_err}")
        else:
            if batch_input_tensors:  # Log error only if we expected predictions
                logger.error(
                    f"Detection output mismatch: Expected {len(batch_cam_ids)} predictions, got {len(all_predictions)}")

        timings['postprocess_scale'] = time.time() - t_postproc_start

        # --- Stage 1d: Tracking per Camera ---
        t_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}  # Tracks with boxes relative to ORIGINAL frame
        current_frame_active_track_keys: Set[TrackKey] = set()
        tracks_to_extract_features_for: Dict[CameraID, List[np.ndarray]] = defaultdict(
            list)  # Track data for ReID trigger

        for cam_id in self.camera_ids:  # Iterate through all expected cameras
            tracker = self.trackers.get(cam_id)
            if not tracker:
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8))
                continue

            # Get the processed detections (already scaled to original coords) for this camera
            cam_detections = detections_per_camera.get(cam_id, [])
            np_dets = np.empty((0, 6))
            if cam_detections:
                try:
                    # Format for BoxMOT: [x1, y1, x2, y2, conf, class_id]
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                except Exception as format_err:
                    logger.error(f"[{cam_id}] Error formatting detections for tracker: {format_err}")
                    np_dets = np.empty((0, 6))

            # Get the ORIGINAL frame for the tracker update (some trackers use appearance)
            # If frame is missing, provide a dummy frame
            original_frame_bgr = frames.get(cam_id)
            dummy_frame_shape = batch_original_shapes[batch_cam_ids.index(cam_id)] if cam_id in batch_cam_ids else (
                100, 100)  # Use known shape if available
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*dummy_frame_shape, 3),
                                                                                             dtype=np.uint8)

            try:
                # Update tracker with scaled detections and the original (or dummy) frame
                tracked_dets_np = tracker.update(np_dets, dummy_frame)  # Expects BGR
                current_frame_tracker_outputs[cam_id] = np.array(
                    tracked_dets_np) if tracked_dets_np is not None and len(tracked_dets_np) > 0 else np.empty((0, 8))
            except Exception as e:
                logger.error(f"[{cam_id}] Tracker update failed: {e}")
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8))

            # --- Identify active tracks and triggers for Re-ID ---
            if current_frame_tracker_outputs[cam_id].shape[0] > 0:
                previous_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in current_frame_tracker_outputs[cam_id]:
                    # BoxMOT format: [x1, y1, x2, y2, track_id, conf, class_id, optional_feature]
                    if len(track_data) >= 5:  # Min xyxy, track_id
                        try:
                            track_id = int(track_data[4])
                        except (ValueError, IndexError):
                            continue  # Skip if track ID is invalid

                        current_track_key: TrackKey = (cam_id, track_id)
                        current_frame_active_track_keys.add(current_track_key)  # Log all active tracks

                        # Only check ReID trigger if we had a *valid* original frame for FE
                        if original_frame_bgr is not None and original_frame_bgr.size > 0:
                            is_newly_seen = track_id not in previous_cam_track_ids
                            last_reid_attempt = self.track_last_reid_frame.get(current_track_key,
                                                                               -self.config.reid_refresh_interval_frames)
                            is_due_for_refresh = (
                                                         frame_idx - last_reid_attempt) >= self.config.reid_refresh_interval_frames

                            if is_newly_seen or is_due_for_refresh:
                                # Need track_data containing bbox relative to original frame for FE
                                tracks_to_extract_features_for[cam_id].append(track_data)
                                self.track_last_reid_frame[current_track_key] = frame_idx  # Record attempt time

        timings['tracking'] = time.time() - t_track_start
        # Old combined time isn't accurate anymore, remove or recalculate if needed
        # timings['detection_tracking'] = ...

        # --- Stage 2: Conditional Feature Extraction ---
        # Extracts features using ORIGINAL frames and track boxes relative to them.
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        for cam_id, tracks_data_list in tracks_to_extract_features_for.items():
            if tracks_data_list:
                frame_bgr = frames.get(cam_id)  # Get ORIGINAL frame again
                if frame_bgr is not None and frame_bgr.size > 0:
                    try:
                        tracks_data_np = np.array(tracks_data_list)
                        features_this_cam = self._extract_features_for_tracks(frame_bgr, tracks_data_np)
                        for track_id, feature in features_this_cam.items():  # track_id here is already int
                            extracted_features_this_frame[(cam_id, track_id)] = feature
                    except Exception as fe_err:
                        logger.error(f"[{cam_id}] Feature extraction call failed: {fe_err}")
        timings['feature_ext'] = time.time() - t_feat_start

        # --- Stage 3: Conditional Re-ID Association ---
        t_reid_start = time.time()
        assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        # --- Stage 4: Combine Tracking Results with Global IDs ---
        # Iterates through ALL tracker outputs from Stage 1d for this frame.
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    # BoxMOT format: [x1, y1, x2, y2, track_id, conf, class_id, ...]
                    if len(track_data) >= 7:  # Check if basic fields exist
                        try:
                            x1, y1, x2, y2 = map(float, track_data[0:4])
                            track_id = int(track_data[4])
                            conf = float(track_data[5])
                            cls = int(track_data[6])
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"[{cam_id}] Could not parse final track data {track_data}: {e}")
                            continue

                        current_track_key: TrackKey = (cam_id, track_id)
                        global_id: Optional[GlobalID]

                        # Check if ReID assigned an ID *in this cycle* for this track
                        if current_track_key in assigned_global_ids_this_cycle:
                            global_id = assigned_global_ids_this_cycle[current_track_key]
                        else:
                            # Otherwise, retrieve the last known ID from the persistent mapping
                            global_id = self.track_to_global_id.get(current_track_key)

                        final_results_per_camera[cam_id].append({
                            'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                            # These are already original coords
                            'track_id': track_id, 'global_id': global_id,
                            'conf': conf, 'class_id': cls
                        })

        # --- Stage 5: Update State and Cleanup ---
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch
        return ProcessedBatchResult(results_per_camera=dict(final_results_per_camera), timings=dict(timings))

    def draw_annotations(self, frames: Dict[CameraID, FrameData], processed_results: Dict[CameraID, List[TrackData]]) -> \
            Dict[CameraID, FrameData]:
        """Draws bounding boxes with track and global IDs on frames."""
        # This function should work as before, as processed_results contain bboxes
        # relative to the original frame dimensions provided in 'frames'.
        annotated_frames: Dict[CameraID, FrameData] = {}
        default_frame_h, default_frame_w = 1080, 1920  # Fallback size
        first_valid_frame_found = False
        for frame in frames.values():  # Determine typical size from input frames
            if frame is not None and frame.size > 0:
                default_frame_h, default_frame_w = frame.shape[:2]
                first_valid_frame_found = True;
                break
        if not first_valid_frame_found: logger.warning("No valid frames in batch for annotation sizing.")

        for cam_id, frame in frames.items():
            current_h, current_w = default_frame_h, default_frame_w
            if frame is None or frame.size == 0:
                # Create placeholder if frame is missing
                placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                            2, cv2.LINE_AA)
                annotated_frames[cam_id] = placeholder;
                continue
            else:
                annotated_frame = frame.copy()  # Work on a copy of the original frame
                current_h, current_w = frame.shape[:2]  # Use actual frame dimensions

            results_for_cam = processed_results.get(cam_id, [])
            for track_info in results_for_cam:
                bbox = track_info.get('bbox_xyxy')  # These are original frame coordinates
                track_id = track_info.get('track_id')
                global_id = track_info.get('global_id')
                if bbox is None: continue

                # Clamp box coordinates to ensure they are within frame bounds
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(current_w - 1, x2)
                y2 = min(current_h - 1, y2)

                if x1 >= x2 or y1 >= y2: continue  # Skip invalid boxes

                # --- Color and Label ---
                color = (200, 200, 200)  # Default grey
                if global_id is not None:
                    seed = int(global_id) * 3 + 5
                    color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)

                if self.config.draw_bounding_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label_parts = []
                if self.config.show_track_id and track_id is not None: label_parts.append(f"T:{track_id}")
                if self.config.show_global_id: label_parts.append(f"G:{global_id if global_id is not None else '?'}")
                label = " ".join(label_parts)

                if label:  # Draw label text and background
                    font_face, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, font_face, font_scale, thickness + 1)

                    # Position label slightly above the box, flipping below if near top edge
                    label_y_pos = y1 - baseline - 5
                    if label_y_pos < text_h: label_y_pos = y2 + text_h + 5  # Place below box if too high

                    # Clamp label position within frame boundaries
                    label_y_pos = max(text_h + baseline, min(label_y_pos, current_h - baseline - 1))
                    label_x_pos = max(0, x1)  # Start label at box's left edge

                    # Calculate background rectangle coordinates
                    bg_x1, bg_y1 = label_x_pos, label_y_pos - text_h - baseline
                    bg_x2, bg_y2 = label_x_pos + text_w, label_y_pos + baseline

                    # Ensure background rect is within bounds and valid
                    bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
                    bg_x2, bg_y2 = min(current_w - 1, bg_x2), min(current_h - 1, bg_y2)

                    if bg_x2 > bg_x1 and bg_y2 > bg_y1:  # Check if background rect is valid
                        # Draw filled background rectangle
                        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
                        # Draw text on top of background
                        cv2.putText(annotated_frame, label, (label_x_pos, label_y_pos), font_face, font_scale,
                                    (0, 0, 0), thickness, cv2.LINE_AA)  # Black text

            annotated_frames[cam_id] = annotated_frame
        return annotated_frames


# --- Main Execution Functions --- (Setup and loop structure identical to previous version)
def setup_paths_and_config() -> PipelineConfig:
    """Determines paths, creates, and returns the pipeline configuration."""
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve()
    config = PipelineConfig()  # Uses defaults defined in the dataclass
    config.device = get_compute_device()

    # Find Tracker Config dynamically
    tracker_filename = f"{config.tracker_type}.yaml"
    potential_paths = [
        BOXMOT_PATH / "configs" / tracker_filename,
        script_dir / "configs" / tracker_filename,
        script_dir / tracker_filename,
        Path.cwd() / "configs" / tracker_filename,
        Path.cwd() / tracker_filename
    ]
    found_path = next((p for p in potential_paths if p.is_file()), None)
    if not found_path:
        # Try finding any .yaml config in the same places if specific one fails
        logger.warning(f"Tracker config '{tracker_filename}' not found. Searching for any '.yaml'...")
        potential_paths = [p.parent for p in potential_paths]  # Get parent dirs
        found_path = next(
            (yaml_file for dir_path in potential_paths for yaml_file in dir_path.glob('*.yaml') if yaml_file.is_file()),
            None)
        if found_path:
            logger.warning(f"Using fallback tracker config: {found_path}")
        else:
            raise FileNotFoundError(f"No tracker config (.yaml) found in checked locations.")

    config.tracker_config_path = found_path
    logger.info(f"Using tracker config: {config.tracker_config_path}")

    # Validate ReID Weights (search common locations)
    if not config.reid_model_weights.is_file():
        potential_reid_paths = [
            script_dir / config.reid_model_weights.name,
            Path.cwd() / config.reid_model_weights.name,
            script_dir / "weights" / config.reid_model_weights.name,  # Common folders
            Path.cwd() / "weights" / config.reid_model_weights.name,
            config.reid_model_weights  # Original path if absolute
        ]
        found_reid_path = next((p for p in potential_reid_paths if p.is_file()), None)
        if not found_reid_path: raise FileNotFoundError(f"ReID weights '{config.reid_model_weights.name}' not found.")
        config.reid_model_weights = found_reid_path
    logger.info(f"Using ReID weights: {config.reid_model_weights}")

    # Validate Dataset Base Path
    if not config.dataset_base_path.is_dir(): raise FileNotFoundError(
        f"Dataset base path not found: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")

    logger.info("Configuration setup complete.")
    logger.info(f"Final Config: {config}")  # Log the final config values being used
    return config


def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    """Validates camera directories and determines the frame sequence."""
    logger.info("--- Loading Dataset Information ---")
    camera_dirs: Dict[CameraID, Path] = {}
    valid_cameras: List[CameraID] = []
    # Assuming structure like MTMMC/train/train/sXX/cYY/rgb/
    base_scene_path = config.dataset_base_path / "train" / "train" / config.selected_scene
    if not base_scene_path.is_dir(): raise FileNotFoundError(f"Scene directory not found: {base_scene_path}")
    logger.info(f"Using scene path: {base_scene_path}")

    for cam_id in config.selected_cameras:
        cam_rgb_dir = base_scene_path / cam_id / "rgb"
        if cam_rgb_dir.is_dir():
            # Check if directory contains image files
            image_files = list(cam_rgb_dir.glob('*.jpg')) + list(
                cam_rgb_dir.glob('*.png'))  # Add other extensions if needed
            if image_files:
                camera_dirs[cam_id] = cam_rgb_dir
                valid_cameras.append(cam_id)
                logger.info(f"Found valid image directory with {len(image_files)} images: {cam_rgb_dir}")
            else:
                logger.warning(
                    f"Image directory found for {cam_id} but contains no .jpg/.png files. Skipping: {cam_rgb_dir}")
        else:
            logger.warning(f"Image directory not found for {cam_id} at {cam_rgb_dir}. Skipping.")

    if not valid_cameras: raise RuntimeError("No valid camera data sources with images available.")
    config.selected_cameras = valid_cameras  # Update config with only the cameras found
    logger.info(f"Processing frames from cameras: {valid_cameras}")

    # Determine frame sequence based on the *first* valid camera found
    image_filenames: List[str] = []
    try:
        first_cam_dir = camera_dirs[valid_cameras[0]]
        image_filenames = sorted_alphanumeric([f.name for f in first_cam_dir.iterdir() if
                                               f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
        if not image_filenames: raise ValueError(
            f"No image files found in the first valid camera directory: {first_cam_dir}")
        logger.info(f"Found {len(image_filenames)} frames based on camera {valid_cameras[0]}.")
    except Exception as e:
        logger.critical(f"Failed to list image files from {valid_cameras[0]}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list image files: {e}") from e
    return camera_dirs, image_filenames


def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the image frame for the specified filename from each camera directory Path."""
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename
        img = None
        if image_path.is_file():
            try:
                img = cv2.imread(str(image_path))  # Reads in BGR format
                if img is None or img.size == 0:
                    logger.warning(f"[{cam_id}] Failed load or empty image: {image_path}")
                    img = None  # Ensure it's None if load fails
            except Exception as e:
                logger.error(f"[{cam_id}] Error reading image {image_path}: {e}")
                img = None  # Ensure it's None on error
        # If file doesn't exist, img remains None implicitly
        current_frames[cam_id] = img
    return current_frames


def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines annotated frames into a grid and displays them."""
    valid_annotated = [f for f in annotated_frames.values() if f is not None and f.size > 0]
    if not valid_annotated:
        # Display a blank screen or message if no frames are available
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated)
        # Try to make a grid layout (e.g., 2x2 for 4 cams, 3x2 for 5-6 cams, etc.)
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))

        # Use the shape of the first valid frame as the target size for the grid cells
        target_h, target_w = valid_annotated[0].shape[:2]
        combined_h, combined_w = rows * target_h, cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        frame_idx = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx < num_cams:
                    frame = valid_annotated[frame_idx]
                    # Resize frame ONLY if it doesn't match the target cell size
                    if frame.shape[:2] != (target_h, target_w):
                        try:
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err:
                            logger.warning(f"Grid resize error: {resize_err}. Using blank cell.")
                            # Create a blank frame of the target size if resize fails
                            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    try:
                        # Place the frame (or resized frame, or blank frame) into the grid
                        combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = frame
                    except ValueError as slice_err:
                        logger.error(
                            f"Grid placement error: {slice_err}. Frame shape {frame.shape}, Target slice [{r * target_h}:{(r + 1) * target_h}, {c * target_w}:{(c + 1) * target_w}]")
                    frame_idx += 1

        # Resize the combined display if it exceeds the maximum width
        disp_h, disp_w = combined_display.shape[:2]
        if disp_w > max_width:
            try:
                scale = max_width / disp_w
                disp_h_new, disp_w_new = int(disp_h * scale), max_width
                combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err:
                logger.error(f"Final display resize failed: {final_resize_err}")

    # Ensure the window exists before showing the image
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.imshow(window_name, combined_display)
    else:
        logger.warning(f"Cannot display frames, window '{window_name}' not found or closed.")


def main():
    """Main execution function."""
    pipeline = None  # Initialize pipeline to None
    try:
        config = setup_paths_and_config()
        camera_dirs, image_filenames = load_dataset_info(config)

        # --- Initialize Pipeline ---
        pipeline = MultiCameraPipeline(config)  # This loads models

        # --- Setup Display Window ---
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        # Attempt to resize window initially if needed, though behavior varies by backend
        # cv2.resizeWindow(config.window_name, config.max_display_width, int(config.max_display_width * 9 / 16)) # Example 16:9 aspect

        logger.info("--- Starting Frame Processing Loop ---")
        start_time = time.time()
        frames_processed_count = 0
        loop_start_time = time.perf_counter()  # Use perf_counter for more precise timing

        for frame_idx, current_filename in enumerate(image_filenames):
            iter_start_time = time.perf_counter()

            # --- Load Frames ---
            current_frames = load_frames_for_batch(camera_dirs, current_filename)
            if not any(f is not None for f in current_frames.values()):
                logger.warning(f"Frame {frame_idx}: No valid images loaded for '{current_filename}'. Skipping batch.")
                continue
            frames_processed_count += 1  # Count batches with at least one valid frame

            # --- Process Batch ---
            batch_result = pipeline.process_frame_batch(current_frames, frame_idx)

            # --- Annotate and Display ---
            # Use original frames (current_frames) for annotation background
            annotated_frames = pipeline.draw_annotations(current_frames, batch_result.results_per_camera)
            display_combined_frames(config.window_name, annotated_frames, config.max_display_width)

            # --- Logging and Timing ---
            iter_end_time = time.perf_counter()
            frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000
            # Log more frequently initially, then less often
            if frame_idx < 10 or frame_idx % 50 == 0:
                timing_str = " | ".join([f"{k}={v * 1000:.1f}ms" for k, v in batch_result.timings.items() if
                                         v > 0.0001])  # Only show significant timings
                track_count = sum(len(tracks) for tracks in batch_result.results_per_camera.values())
                current_loop_duration = iter_end_time - loop_start_time
                avg_fps_so_far = frames_processed_count / current_loop_duration if current_loop_duration > 0 else 0
                logger.info(
                    f"Frame {frame_idx:>4} | Batch Time: {frame_proc_time_ms:>6.1f}ms | AvgFPS: {avg_fps_so_far:5.2f} | "
                    f"Pipeline: {timing_str} | ActiveTracks: {track_count}")

            # --- User Input ---
            key = cv2.waitKey(config.display_wait_ms) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed. Exiting loop.")
                break
            elif key == ord('p'):
                logger.info("Paused. Press any key in the OpenCV window to resume...")
                cv2.waitKey(0)  # Wait indefinitely until a key is pressed in the window
                logger.info("Resuming.")
            # Check if window was closed by user
            if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Display window closed by user. Exiting loop.")
                break

        # --- End of Loop ---
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("--- Pipeline Finished ---")
        logger.info(f"Processed {frames_processed_count} frame batches.")
        if frames_processed_count > 0 and total_time > 0.01:
            logger.info(
                f"Total execution time: {total_time:.2f}s. Overall Avg FPS: {frames_processed_count / total_time:.2f}")
        else:
            logger.info("No frames processed or execution time too short for meaningful FPS.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e:
        logger.critical(f"Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        # Catch any other unexpected exceptions
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info("Closing OpenCV windows...")
        cv2.destroyAllWindows()
        # Force waitKey to process window closing events if needed, although destroyAllWindows should handle it.
        # for _ in range(5): cv2.waitKey(1)

        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            del pipeline  # Explicitly delete pipeline to release model references
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

        logger.info("Exiting script.")


if __name__ == "__main__":
    main()
