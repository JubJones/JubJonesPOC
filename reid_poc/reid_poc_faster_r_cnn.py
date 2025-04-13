# -*- coding: utf-8 -*-
import os
import sys
import time
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set, NamedTuple
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN
# Ensure torchvision.transforms.v2 is available or fallback if needed
try:
    from torchvision.transforms.v2 import Compose
except ImportError:
    from torchvision.transforms import Compose # Fallback for older torchvision
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
    # Example paths, adjust as needed for your system
    dataset_base_path: Path = Path(os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC")) # Example dynamic path
    reid_model_weights: Path = Path("osnet_x0_25_msmt17.pt")
    tracker_config_path: Optional[Path] = None # Set dynamically

    # Dataset Selection
    selected_scene: str = "s10"
    selected_cameras: List[str] = field(default_factory=lambda: ["c09", "c12", "c13", "c16"])

    # Model Parameters
    person_class_id: int = 1
    detection_confidence_threshold: float = 0.5
    reid_similarity_threshold: float = 0.65
    gallery_ema_alpha: float = 0.9
    reid_refresh_interval_frames: int = 10 # How often to re-run ReID for existing tracks

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
TrackID = int
GlobalID = int
TrackKey = Tuple[CameraID, TrackID]
FeatureVector = np.ndarray
BoundingBox = np.ndarray # xyxy format
Detection = Dict[str, Any] # {'bbox_xyxy': BoundingBox, 'conf': float, 'class_id': int}
TrackData = Dict[str, Any] # {'bbox_xyxy': BoundingBox, 'track_id': TrackID, 'global_id': Optional[GlobalID], ...}
FrameData = Optional[np.ndarray]
Timings = Dict[str, float]

class ProcessedBatchResult(NamedTuple):
    """Results structure for a processed batch of frames."""
    results_per_camera: Dict[CameraID, List[TrackData]]
    timings: Timings

# --- Helper Functions ---

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
            # Simple test
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA device confirmed.")
            return device
        except Exception as e:
            logger.warning(f"CUDA reported available, but test failed ({e}). Falling back...")
    # Add MPS check here if needed:
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     logger.info("Using Metal Performance Shaders (MPS) device.")
    #     return torch.device("mps")
    logger.info("Using CPU device.")
    return torch.device("cpu")

def calculate_cosine_similarity(feat1: Optional[FeatureVector], feat2: Optional[FeatureVector]) -> float:
    """Calculates cosine similarity between two feature vectors, handling None and invalid inputs."""
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    # Using np.any instead of np.all for slight optimization
    if not np.any(feat1) or not np.any(feat2): return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0

    try:
        # scipy.spatial.distance.cosine computes 1 - similarity
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0) # Clip distance before subtraction
        return float(np.clip(similarity, 0.0, 1.0)) # Clip final similarity
    except ValueError as e: # Catch potential dimension mismatch errors from cosine_distance
        logger.error(f"Error calculating cosine distance (likely shape mismatch): {e}")
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
    # Add small epsilon to avoid division by zero
    return embedding / (norm + 1e-6)

# --- Main Processing Class ---

class MultiCameraPipeline:
    """Handles multi-camera detection, tracking, and Re-Identification."""

    def __init__(self, config: PipelineConfig):
        """Initializes models, trackers, and state."""
        self.config = config
        self.device = config.device
        self.camera_ids = config.selected_cameras
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids} on device: {self.device}")

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
            # Use transforms associated with the weights
            transforms = weights.transforms()
            logger.info("Faster R-CNN Detector loaded successfully.")
            return model, transforms
        except Exception as e:
            logger.critical(f"FATAL ERROR loading Faster R-CNN: {e}")
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
            logger.info(f"Attempting ReID model load onto device specifier: '{reid_device_specifier}'")
            reid_model_handler = ReidAutoBackend(
                weights=reid_weights, device=reid_device_specifier, half=False # Assuming FP32 for stability
            )
            model = reid_model_handler.model
            if hasattr(model, "warmup"):
                logger.info("Warming up ReID model...")
                model.warmup()
            logger.info("OSNet ReID Model loaded successfully.")
            return model
        except Exception as e:
            logger.critical(f"FATAL ERROR loading ReID model: {e}")
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
            # Ensure device string is correctly formatted for BoxMOT
            tracker_device_str = _get_reid_device_specifier_string(self.device) # Use the same logic as ReID device
            for cam_id in self.camera_ids:
                tracker_instance = create_tracker(
                    tracker_type=self.config.tracker_type,
                    tracker_config=str(tracker_config_path), # Ensure path is string
                    reid_weights=None, # ReID is handled separately
                    device=tracker_device_str,
                    half=False, # Assuming FP32
                    per_class=False # Track all detected classes (should be only person based on detection)
                )
                # Attempt to reset tracker state if possible
                if hasattr(tracker_instance, 'reset'):
                     tracker_instance.reset()
                trackers[cam_id] = tracker_instance
                logger.info(f"Initialized {self.config.tracker_type} for camera {cam_id} on device '{tracker_device_str}'")
            logger.info(f"Initialized {len(trackers)} tracker instances.")
            return trackers
        except Exception as e:
            logger.critical(f"FATAL ERROR initializing trackers: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize trackers") from e

    def _detect_persons(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detects persons in a single frame using the loaded detector."""
        detections: List[Detection] = []
        if self.detector is None or self.detector_transforms is None or frame_bgr is None or frame_bgr.size == 0:
            return detections

        try:
            # Preprocess: Convert BGR (OpenCV) to RGB, then to PIL Image
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # PIL conversion needed if transforms expect PIL
            # If transforms handle tensors directly, this could be optimized
            img_pil = Image.fromarray(img_rgb)

            # Apply transformations
            input_tensor = self.detector_transforms(img_pil)
            input_batch = [input_tensor.to(self.device)]

            # Inference
            with torch.no_grad():
                predictions = self.detector(input_batch)

            # Postprocess: Extract relevant predictions
            pred_boxes = predictions[0]['boxes'].cpu().numpy()
            pred_labels = predictions[0]['labels'].cpu().numpy()
            pred_scores = predictions[0]['scores'].cpu().numpy()

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if label == self.config.person_class_id and score >= self.config.detection_confidence_threshold:
                    x1, y1, x2, y2 = box # Keep as float initially
                    # Ensure valid box dimensions before casting/using
                    if x2 > x1 + 1 and y2 > y1 + 1: # Add small margin for robustness
                         detections.append({
                             'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                             'conf': float(score),
                             'class_id': self.config.person_class_id
                         })
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=False) # Set exc_info=True for full traceback
        return detections

    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        """Extracts Re-ID features for a given set of tracked detections."""
        features: Dict[TrackID, FeatureVector] = {}
        if self.reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.shape[0] == 0:
            return features

        # Ensure input format is correct (BoxMOT expects list/array of XYXY)
        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids = tracked_dets_np[:, 4].astype(np.int32)

        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4:
            logger.warning(f"Invalid bbox shape for feature extraction. Shape: {bboxes_xyxy.shape}. Skipping.")
            return features

        try:
            # BoxMOT's get_features expects image (NumPy BGR) and list/array of bboxes (XYXY)
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)

            if batch_features is not None and len(batch_features) == len(track_ids):
                for i, det_feature in enumerate(batch_features):
                    # Validate feature vector
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        current_track_id = track_ids[i]
                        features[current_track_id] = det_feature # Store raw feature
                    # else: logger.debug(f"Invalid feature vector received for track {track_ids[i]}.") # Too verbose normally
            # else: logger.warning(f"Feature extraction mismatch or empty result. Expected {len(track_ids)}, got {len(batch_features) if batch_features is not None else 'None'}.")
        except Exception as e:
            logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features

    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[TrackKey, Optional[GlobalID]]:
        """Compares features of triggered tracks against the gallery to assign Global IDs."""
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {}
        if not features_per_track:
            return newly_assigned_global_ids

        # Prepare gallery for efficient comparison (filter invalid entries once)
        valid_gallery_items = [
            (gid, emb) for gid, emb in self.reid_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        if not valid_gallery_items:
             valid_gallery_ids = []
             valid_gallery_embeddings = []
        else:
            valid_gallery_ids, valid_gallery_embeddings = zip(*valid_gallery_items)

        for track_key, new_embedding in features_per_track.items():
            # track_key = (camera_id, track_id)
            newly_assigned_global_ids[track_key] = None # Default for this cycle

            if new_embedding is None or not np.isfinite(new_embedding).all() or new_embedding.size == 0:
                logger.warning(f"Skipping ReID for {track_key} due to invalid new embedding.")
                continue

            best_match_global_id: Optional[GlobalID] = None

            # 1. Compare against the existing gallery
            if valid_gallery_ids: # Only compare if gallery is not empty
                try:
                    similarities = np.array([calculate_cosine_similarity(new_embedding, gal_emb) for gal_emb in valid_gallery_embeddings])
                    max_similarity_idx = np.argmax(similarities)
                    max_similarity = similarities[max_similarity_idx]

                    if max_similarity >= self.config.reid_similarity_threshold:
                        best_match_global_id = valid_gallery_ids[max_similarity_idx]
                        # logger.debug(f"ReID Match: {track_key} -> GID {best_match_global_id} (Score: {max_similarity:.3f})")
                except Exception as sim_err:
                    logger.error(f"Error during similarity calculation for {track_key}: {sim_err}")
                    # Continue to next track without assigning ID based on similarity
                    pass

            # 2. Assign Global ID and Update Gallery/Mappings
            assigned_global_id: Optional[GlobalID] = None
            normalized_new_embedding = _normalize_embedding(new_embedding)

            if best_match_global_id is not None:
                # Case A: Matched an existing Global ID
                assigned_global_id = best_match_global_id
                # Update gallery embedding using EMA
                current_gallery_emb = self.reid_gallery.get(assigned_global_id) # Fetch the definitive current embedding
                if current_gallery_emb is not None:
                    updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb +
                                       (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                    self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                else:
                     # Should not happen if ID was in valid_gallery_ids, but handle defensively
                     logger.warning(f"Gallery embedding for matched GID {assigned_global_id} was unexpectedly None. Overwriting.")
                     self.reid_gallery[assigned_global_id] = normalized_new_embedding
            else:
                # Case B: No match found in the current gallery
                last_known_global_id = self.track_to_global_id.get(track_key)

                if last_known_global_id is not None and last_known_global_id in self.reid_gallery:
                    # Case B.1: Track had a previous Global ID. Re-assign it and update gallery.
                    assigned_global_id = last_known_global_id
                    # logger.debug(f"ReID Re-Assign: {track_key} -> GID {assigned_global_id} (No current match)")
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id) # Fetch again
                    if current_gallery_emb is not None:
                        updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb +
                                           (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                        self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                    else:
                        # Defensive handling
                        logger.warning(f"Gallery embedding for re-assigned GID {assigned_global_id} was unexpectedly None. Overwriting.")
                        self.reid_gallery[assigned_global_id] = normalized_new_embedding
                else:
                    # Case B.2: Truly new appearance or track lost its old ID. Assign a new Global ID.
                    assigned_global_id = self.next_global_id
                    self.next_global_id += 1
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding
                    # logger.info(f"ReID New ID: {track_key} -> GID {assigned_global_id}")

            # Update state maps if an ID was assigned/confirmed in this cycle
            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id # Ensure main mapping is always current

        return newly_assigned_global_ids

    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """Updates the 'last seen' state and removes stale entries from state dictionaries."""

        # 1. Update last seen tracks for the next frame's 'is_newly_seen' check
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys:
             new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        # 2. Cleanup stale Re-ID attempt timestamps
        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_reid:
            del self.track_last_reid_frame[key]
            # logger.debug(f"Removed stale ReID timestamp for {key}")

        # 3. Cleanup stale Track -> Global ID mappings
        keys_to_delete_global = set(self.track_to_global_id.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_global:
            del self.track_to_global_id[key]
            # logger.debug(f"Removed stale global ID mapping for {key}")

        # --- Optional Gallery Cleanup ---
        # If needed, add logic here to remove gallery entries for Global IDs
        # that haven't been associated with any active track for a while.
        # Example sketch (needs tracking last active frame per global_id):
        # active_global_ids_this_frame = {self.track_to_global_id[key] for key in current_frame_active_track_keys if key in self.track_to_global_id}
        # update_last_active_frame_for_global_ids(active_global_ids_this_frame, current_frame_idx)
        # stale_global_ids = find_global_ids_inactive_for_too_long(current_frame_idx)
        # for stale_gid in stale_global_ids:
        #     if stale_gid in self.reid_gallery:
        #         del self.reid_gallery[stale_gid]
        #         logger.info(f"Removed stale gallery entry for Global ID: {stale_gid}")


    def process_frame_batch(self, frames: Dict[CameraID, FrameData], frame_idx: int) -> ProcessedBatchResult:
        """Processes a batch of frames: Detect -> Track -> Conditionally Re-ID -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)

        # --- Stage 1: Detection and Tracking per Camera ---
        t_det_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        current_frame_active_track_keys: Set[TrackKey] = set()
        tracks_to_extract_features_for: Dict[CameraID, List[np.ndarray]] = defaultdict(list)

        for cam_id, frame_bgr in frames.items():
            tracker = self.trackers.get(cam_id)
            if not tracker:
                 current_frame_tracker_outputs[cam_id] = np.empty((0, 8))
                 continue # Skip if no tracker for this camera ID

            if frame_bgr is None or frame_bgr.size == 0:
                 # If frame is missing, update tracker with empty detections
                 # to allow it to manage lost tracks etc.
                 np_dets = np.empty((0, 6))
                 dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8) # Minimal dummy frame
            else:
                # Run detection on the valid frame
                detections = self._detect_persons(frame_bgr)
                np_dets = np.empty((0, 6)) # Format: [x1, y1, x2, y2, conf, class_id]
                if detections:
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in detections])
                dummy_frame = frame_bgr # Use the real frame if available

            # Update tracker
            try:
                # Pass detections and the frame (real or dummy)
                tracked_dets_np = tracker.update(np_dets, dummy_frame)
                # Ensure output is a numpy array, handle None or empty list
                if tracked_dets_np is None or len(tracked_dets_np) == 0:
                    current_frame_tracker_outputs[cam_id] = np.empty((0, 8))
                else:
                    current_frame_tracker_outputs[cam_id] = np.array(tracked_dets_np)
            except Exception as e:
                logger.error(f"Tracker update failed for camera {cam_id}: {e}")
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8)) # Ensure consistent empty state on error

            # Identify active tracks and triggers for Re-ID (only if we had a real frame for potential FE)
            if frame_bgr is not None and current_frame_tracker_outputs[cam_id].shape[0] > 0:
                previous_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in current_frame_tracker_outputs[cam_id]:
                    # Check standard tracker output format (e.g., xyxy, track_id, conf, class)
                    if len(track_data) >= 5: # Min: x1,y1,x2,y2,track_id
                        try:
                           track_id = int(track_data[4])
                        except (ValueError, IndexError):
                            logger.warning(f"Could not parse track_id from tracker output: {track_data}")
                            continue

                        current_track_key: TrackKey = (cam_id, track_id)
                        current_frame_active_track_keys.add(current_track_key)

                        # Re-ID Trigger Conditions
                        is_newly_seen = track_id not in previous_cam_track_ids
                        last_reid_attempt = self.track_last_reid_frame.get(current_track_key, -self.config.reid_refresh_interval_frames)
                        is_due_for_refresh = (frame_idx - last_reid_attempt) >= self.config.reid_refresh_interval_frames

                        if is_newly_seen or is_due_for_refresh:
                            tracks_to_extract_features_for[cam_id].append(track_data) # Store raw output for FE
                            self.track_last_reid_frame[current_track_key] = frame_idx # Mark attempt scheduled
            elif current_frame_tracker_outputs[cam_id].shape[0] > 0:
                # Frame was None, but tracker might have updated state (e.g., marked tracks as lost)
                # Log active tracks even without a frame for state cleanup purposes
                for track_data in current_frame_tracker_outputs[cam_id]:
                    if len(track_data) >= 5:
                        try: track_id = int(track_data[4])
                        except (ValueError, IndexError): continue
                        current_frame_active_track_keys.add((cam_id, track_id))


        timings['detection_tracking'] = time.time() - t_det_track_start

        # --- Stage 2: Conditional Feature Extraction ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        for cam_id, tracks_data_list in tracks_to_extract_features_for.items():
            if tracks_data_list: # Only process if list is not empty
                frame_bgr = frames.get(cam_id) # Get the original frame again
                if frame_bgr is not None and frame_bgr.size > 0:
                    # Convert list of track data arrays back to a single NumPy array
                    tracks_data_np = np.array(tracks_data_list)
                    features_this_cam = self._extract_features_for_tracks(frame_bgr, tracks_data_np)
                    for track_id, feature in features_this_cam.items():
                        extracted_features_this_frame[(cam_id, track_id)] = feature
        timings['feature_ext'] = time.time() - t_feat_start

        # --- Stage 3: Conditional Re-ID Association ---
        t_reid_start = time.time()
        assigned_global_ids = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        # --- Stage 4: Combine Tracking Results with Global IDs ---
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    # Check expected format: x1, y1, x2, y2, track_id, conf, class_id, [optional_index]
                    if len(track_data) >= 7:
                        try:
                            x1, y1, x2, y2 = map(float, track_data[0:4])
                            track_id = int(track_data[4])
                            conf = float(track_data[5])
                            cls = int(track_data[6])
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse final track data {track_data}: {e}")
                            continue

                        current_track_key: TrackKey = (cam_id, track_id)
                        global_id: Optional[GlobalID]

                        # Prioritize ID assigned in *this* frame's ReID cycle
                        if current_track_key in assigned_global_ids:
                            global_id = assigned_global_ids[current_track_key]
                        else:
                            # Fallback to the last known ID from the persistent mapping
                            global_id = self.track_to_global_id.get(current_track_key)

                        final_results_per_camera[cam_id].append({
                            'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                            'track_id': track_id,
                            'global_id': global_id, # Can be None if never identified
                            'conf': conf,
                            'class_id': cls
                        })
                    # else: logger.warning(f"Skipping final processing for track data with insufficient length: {len(track_data)}")


        # --- Stage 5: Update State and Cleanup ---
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch
        return ProcessedBatchResult(results_per_camera=dict(final_results_per_camera), timings=dict(timings))

    def draw_annotations(self, frames: Dict[CameraID, FrameData], processed_results: Dict[CameraID, List[TrackData]]) -> Dict[CameraID, FrameData]:
        """Draws bounding boxes with track and global IDs on frames."""
        annotated_frames: Dict[CameraID, FrameData] = {}
        default_frame_h, default_frame_w = 1080, 1920 # Default fallback size

        # Determine a sensible default size from the first valid frame encountered
        first_valid_frame_found = False
        for frame in frames.values():
            if frame is not None and frame.size > 0:
                default_frame_h, default_frame_w = frame.shape[:2]
                first_valid_frame_found = True
                break
        if not first_valid_frame_found:
             logger.warning("No valid frames found in batch to determine annotation size.")

        for cam_id, frame in frames.items():
            current_h, current_w = default_frame_h, default_frame_w
            if frame is None or frame.size == 0:
                # Create placeholder for missing frames
                placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                annotated_frames[cam_id] = placeholder
                continue
            else:
                annotated_frame = frame.copy()
                current_h, current_w = frame.shape[:2] # Use actual frame dimensions

            results_for_cam = processed_results.get(cam_id, [])

            for track_info in results_for_cam:
                bbox = track_info.get('bbox_xyxy')
                track_id = track_info.get('track_id')
                global_id = track_info.get('global_id') # Might be None

                if bbox is None: continue

                x1, y1, x2, y2 = map(int, bbox)
                # Clamp coordinates to frame boundaries defensively
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(current_w - 1, x2), min(current_h - 1, y2) # Use current frame dims
                if x1 >= x2 or y1 >= y2: continue # Skip boxes with no area

                # Determine color based on Global ID
                color = (200, 200, 200) # Light grey default for unknown/None global ID
                if global_id is not None:
                    # Simple deterministic color generation based on ID
                    seed = int(global_id) * 3 + 5
                    color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)

                # Draw bounding box
                if self.config.draw_bounding_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Prepare label text
                label_parts = []
                if self.config.show_track_id and track_id is not None:
                    label_parts.append(f"T:{track_id}")
                if self.config.show_global_id:
                    gid_str = str(global_id) if global_id is not None else "?"
                    label_parts.append(f"G:{gid_str}")
                label = " ".join(label_parts)

                # Draw label background and text
                if label:
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, font_face, font_scale, thickness + 1)

                    # Position label intelligently: prefer above, move below if near top edge
                    label_y_pos = y1 - baseline - 5 # Default position above box
                    if label_y_pos < text_h: # If too close to top edge
                        label_y_pos = y2 + text_h + 5 # Position below box

                    # Ensure label box doesn't go out of bounds vertically
                    label_y_pos = max(text_h + baseline, min(label_y_pos, current_h - baseline - 1))
                    label_x_pos = max(0, x1) # Start at box x1, ensure non-negative

                    # Calculate background rect coords, ensure within bounds
                    bg_x1 = label_x_pos
                    bg_y1 = label_y_pos - text_h - baseline
                    bg_x2 = label_x_pos + text_w
                    bg_y2 = label_y_pos + baseline
                    bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
                    bg_x2, bg_y2 = min(current_w - 1, bg_x2), min(current_h - 1, bg_y2)

                    # Check if background rect has valid dimensions before drawing
                    if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
                        # Adjust text position slightly for padding within the background
                        text_y = label_y_pos
                        text_x = label_x_pos
                        cv2.putText(annotated_frame, label, (text_x, text_y), font_face, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            annotated_frames[cam_id] = annotated_frame
        return annotated_frames


# --- Main Execution Functions ---

def setup_paths_and_config() -> PipelineConfig:
    """Determines paths, creates, and returns the pipeline configuration."""
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve()

    # Initialize config with defaults (including dynamic base path)
    config = PipelineConfig()

    # Determine and set compute device
    config.device = get_compute_device()

    # --- Find Tracker Config Path ---
    tracker_filename = f"{config.tracker_type}.yaml"
    potential_paths = [
        BOXMOT_PATH / "configs" / tracker_filename,
        script_dir / "configs" / tracker_filename, # Common location
        script_dir / tracker_filename,
        Path.cwd() / "configs" / tracker_filename, # Check current working dir
        Path.cwd() / tracker_filename,
    ]
    found_path = None
    logger.info(f"Searching for tracker config '{tracker_filename}'...")
    for i, path in enumerate(potential_paths):
        if path.is_file():
            logger.info(f"Found tracker config at: {path}")
            found_path = path
            break
    if not found_path:
        logger.critical(f"Could not find {tracker_filename} in expected locations: {[str(p) for p in potential_paths]}")
        raise FileNotFoundError(f"Tracker config '{tracker_filename}' not found.")
    config.tracker_config_path = found_path

    # --- Validate ReID Weights Path ---
    if not config.reid_model_weights.is_file():
        logger.info(f"ReID weights file '{config.reid_model_weights.name}' not found in default/specified path.")
        potential_reid_paths = [
            script_dir / config.reid_model_weights.name,
            Path.cwd() / config.reid_model_weights.name,
             # Add other potential locations if needed
        ]
        found_reid_path = None
        for path in potential_reid_paths:
             if path.is_file():
                  logger.info(f"Found ReID weights at alternative location: {path}")
                  found_reid_path = path
                  break
        if found_reid_path:
            config.reid_model_weights = found_reid_path
        else:
            logger.critical(f"ReID weights '{config.reid_model_weights.name}' not found in default path or alternatives.")
            raise FileNotFoundError(f"ReID weights file not found.")

    # --- Validate Dataset Base Path ---
    if not config.dataset_base_path.is_dir():
        logger.critical(f"Dataset base path not found or not a directory: {config.dataset_base_path}")
        raise FileNotFoundError(f"Dataset base path directory not found: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")

    logger.info("Configuration setup complete.")
    return config

def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    """Validates camera directories and determines the frame sequence."""
    logger.info("--- Loading Dataset Information ---")
    camera_dirs: Dict[CameraID, Path] = {} # Store Path objects
    valid_cameras: List[CameraID] = []

    # ***** MODIFIED PATH CONSTRUCTION *****
    # Construct the path including the nested "train" directories
    base_scene_path = config.dataset_base_path / "train" / "train" / config.selected_scene

    if not base_scene_path.is_dir():
        logger.critical(f"Scene directory not found: {base_scene_path}")
        raise FileNotFoundError(f"Scene directory not found: {base_scene_path}")
    logger.info(f"Using scene path: {base_scene_path}")

    logger.info("Validating camera directories...")
    for cam_id in config.selected_cameras:
        # ***** MODIFIED PATH CONSTRUCTION *****
        # Construct the full path to the image directory, including "rgb"
        cam_rgb_dir = base_scene_path / cam_id / "rgb"

        if cam_rgb_dir.is_dir():
            camera_dirs[cam_id] = cam_rgb_dir # Store the Path object directly
            valid_cameras.append(cam_id)
            logger.info(f"Found valid image directory: {cam_rgb_dir}")
        else:
            logger.warning(f"Image directory not found for camera {cam_id} at {cam_rgb_dir}. Skipping.")

    if not valid_cameras:
        logger.critical("No valid camera directories found for the selected cameras/scene.")
        raise RuntimeError("No valid camera data sources available.")

    # Update config to only use cameras found to be valid
    logger.info(f"Successfully validated cameras: {valid_cameras}")
    config.selected_cameras = valid_cameras

    # Determine frame sequence from the first valid camera
    image_filenames: List[str] = []
    try:
        first_cam_id = valid_cameras[0]
        first_cam_dir = camera_dirs[first_cam_id] # This is now the Path object to the rgb dir
        logger.info(f"Reading frame list from: {first_cam_dir}")
        # Filter for common image extensions using pathlib's glob or listdir+suffix check
        image_filenames = sorted_alphanumeric([
            f.name for f in first_cam_dir.iterdir() # Use iterdir() for Path objects
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])
        if not image_filenames:
            raise ValueError(f"No image files found in directory: {first_cam_dir}")
        logger.info(f"Found {len(image_filenames)} frames based on camera {first_cam_id}.")
    except Exception as e:
        logger.critical(f"Error listing image files in {camera_dirs.get(valid_cameras[0])}: {e}", exc_info=True)
        raise RuntimeError("Failed to determine frame sequence.") from e

    return camera_dirs, image_filenames

def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the image frame for the specified filename from each camera directory Path."""
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        # ***** MODIFIED PATH CONSTRUCTION *****
        # Use pathlib's / operator for robust path joining
        image_path = cam_dir_path / filename
        img = None
        # Check existence using pathlib's exists()
        if image_path.exists() and image_path.is_file():
            try:
                # Convert Path object to string for cv2.imread
                img = cv2.imread(str(image_path))
                if img is None or img.size == 0:
                    logger.warning(f"Failed to load or empty image at {image_path}")
                    img = None # Ensure it's None if loading failed
            except Exception as e:
                logger.error(f"Error reading image {image_path}: {e}")
                img = None
        # else: logger.debug(f"Image path not found or not a file: {image_path}") # Too verbose
        current_frames[cam_id] = img
    return current_frames


def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines annotated frames into a grid and displays them."""
    valid_annotated = [f for f in annotated_frames.values() if f is not None and f.size > 0]
    combined_display = None

    if not valid_annotated:
        # Display a placeholder if no frames are available
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames to Display", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated)
        # Simple grid layout calculation
        rows = int(np.ceil(np.sqrt(num_cams)))
        cols = int(np.ceil(num_cams / rows))

        # Use the shape of the first valid frame as the target size for consistency
        target_h, target_w = valid_annotated[0].shape[:2]
        combined_h, combined_w = rows * target_h, cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        frame_idx = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx < num_cams:
                    frame = valid_annotated[frame_idx]
                    # Resize only if necessary (can happen if cameras have different resolutions)
                    if frame.shape[0]!= target_h or frame.shape[1]!= target_w:
                        try:
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err:
                            logger.warning(f"Could not resize frame {frame_idx}: {resize_err}")
                            # Create a black placeholder if resize fails
                            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                            cv2.putText(frame, "Resize Err", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    # Place frame into the grid
                    try:
                        combined_display[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame
                    except ValueError as slice_err:
                         logger.error(f"Error placing frame {frame_idx} in grid (shape: {frame.shape}, target: {target_h}x{target_w}): {slice_err}")

                    frame_idx += 1

        # Resize the final combined display if it exceeds the maximum width
        disp_h, disp_w = combined_display.shape[:2]
        if disp_w > max_width:
            try:
                scale = max_width / disp_w
                disp_h_new = int(disp_h * scale)
                disp_w_new = max_width
                combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err:
                 logger.error(f"Failed to resize final display grid: {final_resize_err}")
                 # Fallback: show the unresized grid if possible
                 pass # Keep the potentially oversized combined_display


    cv2.imshow(window_name, combined_display)

def main():
    """Main execution function."""
    try:
        config = setup_paths_and_config()
        camera_dirs, image_filenames = load_dataset_info(config) # camera_dirs now Dict[CameraID, Path]
        pipeline = MultiCameraPipeline(config)

        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL) # Allow resizing

        logger.info("--- Starting Frame Processing Loop ---")
        start_time = time.time()
        frames_processed_count = 0
        loop_start_time = time.perf_counter()

        for frame_idx, current_filename in enumerate(image_filenames):
            iter_start_time = time.perf_counter()

            # --- Load Frames ---
            current_frames = load_frames_for_batch(camera_dirs, current_filename)
            if not any(f is not None for f in current_frames.values()):
                logger.warning(f"Frame {frame_idx}: No valid images loaded for filename '{current_filename}'. Skipping batch.")
                continue # Skip processing if no frames were loaded for any camera
            frames_processed_count += 1

            # --- Process Batch ---
            batch_result = pipeline.process_frame_batch(current_frames, frame_idx)

            # --- Draw Annotations ---
            annotated_frames = pipeline.draw_annotations(current_frames, batch_result.results_per_camera)

            # --- Visualization ---
            display_combined_frames(config.window_name, annotated_frames, config.max_display_width)

            # --- Print Timings (Periodically) ---
            iter_end_time = time.perf_counter()
            frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000
            if frame_idx % 20 == 0 or frame_idx < 5: # Log less frequently
                 pipeline_timing_str = " | ".join([f"{k}={v*1000:.1f}" for k, v in batch_result.timings.items()])
                 track_count = sum(len(tracks) for tracks in batch_result.results_per_camera.values())
                 current_loop_time = iter_end_time - loop_start_time
                 avg_fps_so_far = (frame_idx + 1) / current_loop_time if current_loop_time > 0 else 0
                 logger.info(
                     f"Frame {frame_idx:>4} [{current_filename}] | FrameTime: {frame_proc_time_ms: >6.1f}ms | "
                     f"AvgFPS: {avg_fps_so_far:.2f} | "
                     f"Pipeline (ms): {pipeline_timing_str} | ActiveTracks: {track_count}"
                 )

            # --- User Interaction ---
            key = cv2.waitKey(config.display_wait_ms) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed. Exiting loop.")
                break
            elif key == ord('p'):
                logger.info("Pause key pressed. Press any key in the OpenCV window to continue...")
                cv2.waitKey(0) # Wait indefinitely until another key is pressed
                logger.info("Resuming...")


        # --- Cleanup ---
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("--- Pipeline Finished ---")
        logger.info(f"Processed {frames_processed_count} frame indices.")
        if frames_processed_count > 0 and total_time > 0:
            avg_fps = frames_processed_count / total_time
            logger.info(f"Total execution time: {total_time:.2f} seconds.")
            logger.info(f"Overall Average FPS: {avg_fps:.2f}")
        else:
            logger.info("No frames were processed or execution time was zero.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e:
        logger.critical(f"A critical error occurred during setup or execution: {e}", exc_info=True)
    except KeyboardInterrupt:
         logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed.")
        # Explicitly clear CUDA cache if applicable
        if torch.cuda.is_available():
             logger.info("Clearing CUDA cache.")
             torch.cuda.empty_cache()
        logger.info("Exiting.")


if __name__ == "__main__":
    main()