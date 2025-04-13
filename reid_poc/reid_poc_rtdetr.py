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
# Removed torchvision detector-specific imports

# --- Ultralytics Import ---
try:
    from ultralytics import RTDETR # Use RTDETR
except ImportError as e:
    # Use logging for critical errors before logger is fully configured if needed
    logging.critical(f"Failed to import ultralytics. Is it installed (pip install ultralytics)? Error: {e}")
    sys.exit(1)
# --- End Ultralytics Import ---

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

# Removed PIL import as RTDETR takes numpy directly
from scipy.spatial.distance import cosine as cosine_distance

# --- Basic Logging Setup ---
# Setup logging early so it's available for all messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class PipelineConfig:
    """Configuration settings for the multi-camera pipeline."""
    # Paths - Use Path objects for cross-platform compatibility
    dataset_base_path: Path = Path(os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"))
    # *** Detector Path ***
    detector_model_path: Path = Path("rtdetr-l.pt") # Default RT-DETR model path
    reid_model_weights: Path = Path("osnet_x0_25_msmt17.pt")
    tracker_config_path: Optional[Path] = None # Set dynamically

    # Dataset Selection
    selected_scene: str = "s10"
    selected_cameras: List[str] = field(default_factory=lambda: ["c09", "c12", "c13", "c16"])

    # Model Parameters
    # *** Person Class ID for COCO/Ultralytics models ***
    person_class_id: int = 0 # Typically 0 for 'person' in models trained on COCO
    detection_confidence_threshold: float = 0.5
    reid_similarity_threshold: float = 0.65
    gallery_ema_alpha: float = 0.9 # Weighting for historical embedding in EMA update
    reid_refresh_interval_frames: int = 10

    # Tracker Type
    tracker_type: str = 'bytetrack'

    # Execution
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Visualization
    draw_bounding_boxes: bool = True
    show_track_id: bool = True
    show_global_id: bool = True
    # *** Updated Window Name ***
    window_name: str = "Multi-Camera Tracking & Re-ID (RTDETR + ByteTrack + OSNet)"
    display_wait_ms: int = 1
    max_display_width: int = 1920

# --- Type Aliases and Structures ---
CameraID = str
TrackID = int # Use Python's native int for type hinting keys
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
    """Selects and returns the best available compute device (MPS, CUDA, CPU)."""
    logger.info("--- Determining Compute Device ---")
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
         try:
             device = torch.device("mps")
             logger.info("Attempting to use Metal Performance Shaders (MPS) device.")
             _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps") # Quick test
             logger.info("MPS device confirmed.")
             return device
         except Exception as e: logger.warning(f"MPS reported available, but test failed ({e}). Falling back...")
    # Check for CUDA
    elif torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"Attempting to use CUDA device: {torch.cuda.get_device_name(device)}")
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA device confirmed.")
            return device
        except Exception as e:
            logger.warning(f"CUDA reported available, but test failed ({e}). Falling back...")
    # Fallback to CPU
    logger.info("Using CPU device.")
    return torch.device("cpu")


def calculate_cosine_similarity(feat1: Optional[FeatureVector], feat2: Optional[FeatureVector]) -> float:
    """Calculates cosine similarity between two feature vectors, handling None and invalid inputs."""
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten(); feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if not np.any(feat1) or not np.any(feat2): return 0.0 # Check if either is all zeros
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0
    try:
        # distance = 1 - similarity
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0) # Clip distance 0-2
        return float(np.clip(similarity, 0.0, 1.0)) # Clip final similarity 0-1
    except ValueError as e:
        # More specific error for shape mismatches likely caught by cosine_distance
        logger.error(f"Cosine distance error (likely shape mismatch {feat1.shape} vs {feat2.shape}): {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error calculating cosine distance: {e}")
        return 0.0

def _get_reid_device_specifier_string(device: torch.device) -> str:
    """Determines the device string specifier needed by ReidAutoBackend/BoxMOT."""
    if device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        return str(idx)
    elif device.type == 'mps': return 'mps'
    return 'cpu'

def _normalize_embedding(embedding: FeatureVector) -> FeatureVector:
    """Normalizes a feature vector using L2 norm."""
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-6) # Add epsilon for numerical stability

# --- Main Processing Class ---

class MultiCameraPipeline:
    """Handles multi-camera detection (RTDETR), tracking, and Re-Identification."""

    def __init__(self, config: PipelineConfig):
        """Initializes models, trackers, and state."""
        self.config = config
        self.device = config.device
        self.camera_ids = config.selected_cameras
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids} on device: {self.device}")

        # *** Load RT-DETR Detector ***
        self.detector = self._load_detector()
        # No separate detector_transforms needed for RTDETR predict interface

        self.reid_model = self._load_reid_model()
        self.trackers = self._initialize_trackers()

        # State Management (Identical)
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.next_global_id: GlobalID = 1
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        self.track_last_reid_frame: Dict[TrackKey, int] = {}

    # *** MODIFIED: Load RT-DETR ***
    def _load_detector(self) -> RTDETR:
        """Loads the RT-DETR object detector model."""
        logger.info(f"Loading RTDETR detector from: {self.config.detector_model_path}")
        if not self.config.detector_model_path.is_file():
            logger.critical(f"FATAL ERROR: Detector weights file not found at {self.config.detector_model_path}")
            raise FileNotFoundError(f"Detector weights not found: {self.config.detector_model_path}")
        try:
            # Ultralytics models usually take string path
            model = RTDETR(str(self.config.detector_model_path))
            # Move model to appropriate device (Ultralytics handles this internally too, but explicit is fine)
            model.to(self.device)
            # Optional: Warmup (can improve timing of first inference)
            # logger.info("Warming up RTDETR detector...")
            # model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=self.device)
            logger.info("RTDETR Detector loaded successfully.")
            return model
        except Exception as e:
            logger.critical(f"FATAL ERROR loading RTDETR detector: {e}", exc_info=True)
            raise RuntimeError("Failed to load object detector") from e

    def _load_reid_model(self) -> Optional[BaseModelBackend]:
        """Loads the OSNet Re-Identification model."""
        # (Identical to previous refactored version)
        reid_weights = self.config.reid_model_weights
        logger.info(f"Loading SHARED OSNet ReID model from: {reid_weights}")
        if not reid_weights.is_file():
             logger.critical(f"FATAL ERROR: ReID weights file not found at {reid_weights}")
             raise FileNotFoundError(f"ReID weights not found: {reid_weights}")
        try:
            reid_device_specifier = _get_reid_device_specifier_string(self.device)
            reid_model_handler = ReidAutoBackend(
                weights=reid_weights, device=reid_device_specifier, half=False
            )
            model = reid_model_handler.model
            if hasattr(model, "warmup"): model.warmup() # Warmup if method exists
            logger.info("OSNet ReID Model loaded successfully.")
            return model
        except Exception as e:
            logger.critical(f"FATAL ERROR loading ReID model: {e}", exc_info=True)
            raise RuntimeError("Failed to load ReID model") from e

    def _initialize_trackers(self) -> Dict[CameraID, BaseTracker]:
        """Initializes a tracker instance for each camera."""
        # (Identical to previous refactored version)
        logger.info(f"Initializing {self.config.tracker_type} trackers...")
        trackers: Dict[CameraID, BaseTracker] = {}
        tracker_config_path = self.config.tracker_config_path
        if not tracker_config_path or not tracker_config_path.is_file():
             logger.critical(f"FATAL ERROR: Tracker config not found: {tracker_config_path}")
             raise FileNotFoundError(f"Tracker config not found: {tracker_config_path}")
        try:
            tracker_device_str = _get_reid_device_specifier_string(self.device)
            for cam_id in self.camera_ids:
                tracker_instance = create_tracker(
                    tracker_type=self.config.tracker_type,
                    tracker_config=str(tracker_config_path),
                    reid_weights=None, device=tracker_device_str, half=False, per_class=False
                )
                if hasattr(tracker_instance, 'reset'): tracker_instance.reset()
                trackers[cam_id] = tracker_instance
                logger.info(f"Initialized {self.config.tracker_type} for camera {cam_id} on device '{tracker_device_str}'")
            logger.info(f"Initialized {len(trackers)} tracker instances.")
            return trackers
        except Exception as e:
            logger.critical(f"FATAL ERROR initializing trackers: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize trackers") from e

    # *** MODIFIED: Use RT-DETR for detection ***
    def _detect_persons(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detects persons in a single frame using the loaded RT-DETR detector."""
        detections: List[Detection] = []
        if self.detector is None or frame_bgr is None or frame_bgr.size == 0:
            return detections

        try:
            # RTDETR's predict method takes BGR NumPy array directly
            results = self.detector.predict(
                frame_bgr,
                classes=[self.config.person_class_id],  # Filter for persons during detection
                conf=self.config.detection_confidence_threshold,
                verbose=False,                          # Suppress ultralytics console output
                device=self.device                      # Specify device for inference
                # Consider adding other args like 'imgsz' if needed for specific input size
            )

            # Process results (Ultralytics Results object)
            # results is a list (usually one element for single image)
            if results and results[0].boxes is not None:
                # Extract boxes (xyxy format), confidences, and optionally classes
                boxes_data = results[0].boxes
                boxes_xyxy = boxes_data.xyxy.cpu().numpy() # Get boxes as numpy array
                confs = boxes_data.conf.cpu().numpy()      # Get confidences as numpy array
                # cls_ids = boxes_data.cls.cpu().numpy() # Uncomment if you need class IDs returned by detector

                for box, score in zip(boxes_xyxy, confs):
                    # No need to check class ID here as we filtered in predict()
                    x1, y1, x2, y2 = box.astype(np.float32) # Keep as float
                    # Basic validity check
                    if x2 > x1 + 1 and y2 > y1 + 1: # Ensure minimum width/height
                         detections.append({
                             'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                             'conf': float(score),
                             'class_id': self.config.person_class_id # Use the configured person class ID
                         })
        except Exception as e:
            logger.error(f"RTDETR Detection error: {e}", exc_info=True) # Log full traceback for errors
        return detections


    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        """Extracts Re-ID features for a given set of tracked detections."""
        # (Identical to previous corrected version)
        features: Dict[TrackID, FeatureVector] = {} # Hint uses TrackID=int
        if self.reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.shape[0] == 0:
            return features
        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids = tracked_dets_np[:, 4] # Keep as numpy array initially
        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4:
            logger.warning(f"Invalid bbox shape for FE. Shape: {bboxes_xyxy.shape}. Skipping.")
            return features
        try:
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)
            if batch_features is not None and len(batch_features) == len(track_ids):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        current_track_id: TrackID = int(track_ids[i]) # Cast key to int
                        features[current_track_id] = det_feature
        except Exception as e: logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features

    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[TrackKey, Optional[GlobalID]]:
        """Compares features of triggered tracks against the gallery to assign Global IDs."""
        # (Identical to previous corrected version)
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {}
        if not features_per_track: return newly_assigned_global_ids
        valid_gallery_items = [(gid, emb) for gid, emb in self.reid_gallery.items() if emb is not None and np.isfinite(emb).all() and emb.size > 0]
        if not valid_gallery_items: valid_gallery_ids, valid_gallery_embeddings = [], []
        else: valid_gallery_ids, valid_gallery_embeddings = zip(*valid_gallery_items)

        for track_key, new_embedding in features_per_track.items():
            newly_assigned_global_ids[track_key] = None
            if new_embedding is None or not np.isfinite(new_embedding).all() or new_embedding.size == 0: continue
            best_match_global_id: Optional[GlobalID] = None
            if valid_gallery_ids:
                try:
                    similarities = np.array([calculate_cosine_similarity(new_embedding, gal_emb) for gal_emb in valid_gallery_embeddings])
                    max_similarity_idx = np.argmax(similarities)
                    if similarities[max_similarity_idx] >= self.config.reid_similarity_threshold:
                        best_match_global_id = valid_gallery_ids[max_similarity_idx]
                except Exception as sim_err: logger.error(f"Similarity calc error {track_key}: {sim_err}")

            assigned_global_id: Optional[GlobalID] = None
            normalized_new_embedding = _normalize_embedding(new_embedding)

            if best_match_global_id is not None: # Case A: Match found
                assigned_global_id = best_match_global_id
                current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                if current_gallery_emb is not None:
                    updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb + (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                    self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                else: self.reid_gallery[assigned_global_id] = normalized_new_embedding
            else: # Case B: No match
                last_known_global_id = self.track_to_global_id.get(track_key)
                if last_known_global_id is not None and last_known_global_id in self.reid_gallery: # Case B.1: Re-assign old
                    assigned_global_id = last_known_global_id
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb + (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                        self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                    else: self.reid_gallery[assigned_global_id] = normalized_new_embedding
                else: # Case B.2: Assign new
                    assigned_global_id = self.next_global_id; self.next_global_id += 1
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding

            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id
        return newly_assigned_global_ids

    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """Updates the 'last seen' state and removes stale entries from state dictionaries."""
        # (Identical to previous refactored version)
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys: new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen
        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_reid: del self.track_last_reid_frame[key]
        keys_to_delete_global = set(self.track_to_global_id.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_global: del self.track_to_global_id[key]

    def process_frame_batch(self, frames: Dict[CameraID, FrameData], frame_idx: int) -> ProcessedBatchResult:
        """Processes a batch of frames: Detect (RTDETR) -> Track -> Conditionally Re-ID -> Associate."""
        # (Logic identical to previous corrected version, relies on internal method changes)
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)
        t_det_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        current_frame_active_track_keys: Set[TrackKey] = set()
        tracks_to_extract_features_for: Dict[CameraID, List[np.ndarray]] = defaultdict(list)

        # Stage 1: Detection & Tracking
        for cam_id, frame_bgr in frames.items():
            tracker = self.trackers.get(cam_id)
            if not tracker: current_frame_tracker_outputs[cam_id] = np.empty((0, 8)); continue
            np_dets = np.empty((0, 6))
            dummy_frame = frame_bgr
            if frame_bgr is None or frame_bgr.size == 0: dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                 detections = self._detect_persons(frame_bgr) # Uses RTDETR now
                 if detections: np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in detections])
            try:
                tracked_dets_np = tracker.update(np_dets, dummy_frame)
                current_frame_tracker_outputs[cam_id] = np.array(tracked_dets_np) if tracked_dets_np is not None and len(tracked_dets_np) > 0 else np.empty((0, 8))
            except Exception as e: logger.error(f"Tracker update failed cam {cam_id}: {e}"); current_frame_tracker_outputs[cam_id] = np.empty((0, 8))

            if current_frame_tracker_outputs[cam_id].shape[0] > 0:
                previous_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in current_frame_tracker_outputs[cam_id]:
                    if len(track_data) >= 5:
                        try: track_id = int(track_data[4])
                        except (ValueError, IndexError): continue
                        current_track_key: TrackKey = (cam_id, track_id)
                        current_frame_active_track_keys.add(current_track_key)
                        if frame_bgr is not None and frame_bgr.size > 0: # Check ReID trigger only if frame was valid
                            is_newly_seen = track_id not in previous_cam_track_ids
                            last_reid = self.track_last_reid_frame.get(current_track_key, -self.config.reid_refresh_interval_frames)
                            is_due = (frame_idx - last_reid) >= self.config.reid_refresh_interval_frames
                            if is_newly_seen or is_due:
                                tracks_to_extract_features_for[cam_id].append(track_data)
                                self.track_last_reid_frame[current_track_key] = frame_idx
        timings['detection_tracking'] = time.time() - t_det_track_start

        # Stage 2: Feature Extraction (Conditional)
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        for cam_id, tracks_data_list in tracks_to_extract_features_for.items():
            if tracks_data_list:
                frame_bgr = frames.get(cam_id)
                if frame_bgr is not None and frame_bgr.size > 0:
                    features_this_cam = self._extract_features_for_tracks(frame_bgr, np.array(tracks_data_list))
                    for track_id, feature in features_this_cam.items(): extracted_features_this_frame[(cam_id, track_id)] = feature
        timings['feature_ext'] = time.time() - t_feat_start

        # Stage 3: Re-ID Association (Conditional)
        t_reid_start = time.time()
        assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        # Stage 4: Combine Results
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                 for track_data in tracked_dets_np:
                    if len(track_data) >= 7: # xyxy, tid, conf, cls
                        try: x1,y1,x2,y2=map(float,track_data[0:4]); track_id=int(track_data[4]); conf=float(track_data[5]); cls=int(track_data[6])
                        except (ValueError, IndexError) as e: logger.warning(f"Parse error final data {track_data}: {e}"); continue
                        current_track_key: TrackKey = (cam_id, track_id)
                        global_id = assigned_global_ids_this_cycle.get(current_track_key, self.track_to_global_id.get(current_track_key))
                        final_results_per_camera[cam_id].append({'bbox_xyxy': np.array([x1, y1, x2, y2],dtype=np.float32), 'track_id': track_id, 'global_id': global_id, 'conf': conf, 'class_id': cls})

        # Stage 5: State Update & Cleanup
        self._update_and_cleanup_state(current_frame_active_track_keys)

        timings['total'] = time.time() - t_start_batch
        return ProcessedBatchResult(results_per_camera=dict(final_results_per_camera), timings=dict(timings))

    def draw_annotations(self, frames: Dict[CameraID, FrameData], processed_results: Dict[CameraID, List[TrackData]]) -> Dict[CameraID, FrameData]:
        """Draws bounding boxes with track and global IDs on frames."""
        # (Identical to previous refactored version)
        annotated_frames: Dict[CameraID, FrameData] = {}
        default_frame_h, default_frame_w = 1080, 1920
        first_valid_frame_found = False
        for frame in frames.values():
            if frame is not None and frame.size > 0: default_frame_h, default_frame_w = frame.shape[:2]; first_valid_frame_found = True; break
        if not first_valid_frame_found: logger.warning("No valid frames for annotation sizing.")

        for cam_id, frame in frames.items():
            current_h, current_w = default_frame_h, default_frame_w
            if frame is None or frame.size == 0:
                placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                annotated_frames[cam_id] = placeholder; continue
            else: annotated_frame = frame.copy(); current_h, current_w = frame.shape[:2]

            results_for_cam = processed_results.get(cam_id, [])
            for track_info in results_for_cam:
                bbox, track_id, global_id = track_info.get('bbox_xyxy'), track_info.get('track_id'), track_info.get('global_id')
                if bbox is None: continue
                x1, y1, x2, y2 = map(int, bbox); x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(current_w - 1, x2), min(current_h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue
                color = (200, 200, 200);
                if global_id is not None: seed = int(global_id)*3+5; color = ((seed*41)%200+55, (seed*17)%200+55, (seed*29)%200+55)
                if self.config.draw_bounding_boxes: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label_parts = []
                if self.config.show_track_id and track_id is not None: label_parts.append(f"T:{track_id}")
                if self.config.show_global_id: label_parts.append(f"G:{global_id if global_id is not None else '?'}")
                label = " ".join(label_parts)
                if label:
                    font_face, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    (tw, th), bl = cv2.getTextSize(label, font_face, font_scale, thickness+1)
                    ty = y1 - bl - 5;
                    if ty < th : ty = y2 + th + 5
                    ty = max(th+bl, min(ty, current_h-bl-1)); tx = max(0, x1)
                    bg1 = (tx, ty - th - bl); bg2 = (tx + tw, ty + bl)
                    bg1 = (max(0,bg1[0]), max(0,bg1[1])); bg2 = (min(current_w-1,bg2[0]), min(current_h-1,bg2[1]))
                    if bg2[0] > bg1[0] and bg2[1] > bg1[1]:
                        cv2.rectangle(annotated_frame, bg1, bg2, color, cv2.FILLED)
                        cv2.putText(annotated_frame, label, (tx, ty), font_face, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            annotated_frames[cam_id] = annotated_frame
        return annotated_frames

# --- Main Execution Functions ---

def setup_paths_and_config() -> PipelineConfig:
    """Determines paths, creates, and returns the pipeline configuration."""
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve()
    config = PipelineConfig() # Load defaults
    config.device = get_compute_device() # Determine device

    # --- Find Detector Model Path ---
    det_path = config.detector_model_path
    if not det_path.is_file():
        logger.warning(f"Detector weights not found at default: {det_path}")
        potential_det_paths = [ script_dir/det_path.name, Path.cwd()/det_path.name ]
        found_det_path = next((p for p in potential_det_paths if p.is_file()), None)
        if not found_det_path: raise FileNotFoundError(f"Detector weights '{det_path.name}' not found.")
        config.detector_model_path = found_det_path; logger.info(f"Found detector weights: {found_det_path}")
    else: logger.info(f"Using detector weights: {det_path}")

    # --- Find Tracker Config Path ---
    tracker_filename = f"{config.tracker_type}.yaml"
    potential_paths = [ BOXMOT_PATH/"configs"/tracker_filename, script_dir/"configs"/tracker_filename, script_dir/tracker_filename, Path.cwd()/"configs"/tracker_filename, Path.cwd()/tracker_filename ]
    found_path = next((p for p in potential_paths if p.is_file()), None)
    if not found_path: raise FileNotFoundError(f"Tracker config '{tracker_filename}' not found in {potential_paths}")
    config.tracker_config_path = found_path; logger.info(f"Found tracker config: {found_path}")

    # --- Validate ReID Weights Path ---
    reid_path = config.reid_model_weights
    if not reid_path.is_file():
        logger.warning(f"ReID weights not found at default: {reid_path}")
        potential_reid_paths = [ script_dir/reid_path.name, Path.cwd()/reid_path.name ]
        found_reid_path = next((p for p in potential_reid_paths if p.is_file()), None)
        if not found_reid_path: raise FileNotFoundError(f"ReID weights '{reid_path.name}' not found.")
        config.reid_model_weights = found_reid_path; logger.info(f"Found ReID weights: {found_reid_path}")
    else: logger.info(f"Using ReID weights: {reid_path}")

    # --- Validate Dataset Base Path ---
    if not config.dataset_base_path.is_dir(): raise FileNotFoundError(f"Dataset base path not found: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")

    logger.info("Configuration setup complete.")
    return config

def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    """Validates camera directories and determines the frame sequence."""
    # (Identical to previous refactored version)
    logger.info("--- Loading Dataset Information ---")
    camera_dirs: Dict[CameraID, Path] = {}
    valid_cameras: List[CameraID] = []
    base_scene_path = config.dataset_base_path / "train" / "train" / config.selected_scene
    if not base_scene_path.is_dir(): raise FileNotFoundError(f"Scene directory not found: {base_scene_path}")
    logger.info(f"Using scene path: {base_scene_path}")
    for cam_id in config.selected_cameras:
        cam_rgb_dir = base_scene_path / cam_id / "rgb"
        if cam_rgb_dir.is_dir():
            camera_dirs[cam_id] = cam_rgb_dir; valid_cameras.append(cam_id)
            logger.info(f"Found valid image directory: {cam_rgb_dir}")
        else: logger.warning(f"Image directory not found for {cam_id} at {cam_rgb_dir}. Skipping.")
    if not valid_cameras: raise RuntimeError("No valid camera data sources available.")
    config.selected_cameras = valid_cameras; logger.info(f"Processing frames from cameras: {valid_cameras}")
    image_filenames: List[str] = []
    try:
        first_cam_dir = camera_dirs[valid_cameras[0]]
        image_filenames = sorted_alphanumeric([ f.name for f in first_cam_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"] ])
        if not image_filenames: raise ValueError(f"No image files found in: {first_cam_dir}")
        logger.info(f"Found {len(image_filenames)} frames based on camera {valid_cameras[0]}.")
    except Exception as e: raise RuntimeError(f"Failed to list image files: {e}") from e
    return camera_dirs, image_filenames

def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    """Loads the image frame for the specified filename from each camera directory Path."""
    # (Identical to previous refactored version)
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename; img = None
        if image_path.is_file():
            try: img = cv2.imread(str(image_path))
            except Exception as e: logger.error(f"Error reading image {image_path}: {e}")
            if img is None or img.size == 0: logger.warning(f"Failed load or empty image: {image_path}"); img = None
        current_frames[cam_id] = img
    return current_frames

def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines annotated frames into a grid and displays them."""
    # (Identical to previous refactored version)
    valid_annotated = [f for f in annotated_frames.values() if f is not None and f.size > 0]
    if not valid_annotated:
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated)
        rows, cols = int(np.ceil(np.sqrt(num_cams))), int(np.ceil(num_cams / int(np.ceil(np.sqrt(num_cams)))))
        target_h, target_w = valid_annotated[0].shape[:2]
        combined_h, combined_w = rows * target_h, cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        frame_idx = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx < num_cams:
                    frame = valid_annotated[frame_idx]
                    if frame.shape[:2] != (target_h, target_w):
                        try: frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err: logger.warning(f"Resize err: {resize_err}"); frame = np.zeros_like(combined_display[0:target_h, 0:target_w])
                    try: combined_display[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame
                    except ValueError as slice_err: logger.error(f"Grid placement error: {slice_err}")
                    frame_idx += 1
        disp_h, disp_w = combined_display.shape[:2]
        if disp_w > max_width:
            try: scale = max_width / disp_w; disp_h_new, disp_w_new = int(disp_h * scale), max_width; combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err: logger.error(f"Final resize failed: {final_resize_err}")
    cv2.imshow(window_name, combined_display)

def main():
    """Main execution function."""
    try:
        config = setup_paths_and_config() # Gets config with validated paths
        camera_dirs, image_filenames = load_dataset_info(config) # Gets data sources

        # *** Instantiate pipeline using the config ***
        # No need to pass paths individually if they are in config
        pipeline = MultiCameraPipeline(config)

        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        logger.info("--- Starting Frame Processing Loop ---")
        start_time = time.time(); frames_processed_count = 0; loop_start_time = time.perf_counter()

        for frame_idx, current_filename in enumerate(image_filenames):
            iter_start_time = time.perf_counter()
            current_frames = load_frames_for_batch(camera_dirs, current_filename)
            if not any(f is not None for f in current_frames.values()):
                logger.warning(f"Frame {frame_idx}: No valid images loaded for '{current_filename}'. Skipping.")
                continue
            frames_processed_count += 1

            batch_result = pipeline.process_frame_batch(current_frames, frame_idx)
            annotated_frames = pipeline.draw_annotations(current_frames, batch_result.results_per_camera)
            display_combined_frames(config.window_name, annotated_frames, config.max_display_width)

            iter_end_time = time.perf_counter()
            frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000
            if frame_idx % 20 == 0 or frame_idx < 5: # Log periodically
                 timing_str = " | ".join([f"{k}={v*1000:.1f}" for k, v in batch_result.timings.items()])
                 track_count = sum(len(tracks) for tracks in batch_result.results_per_camera.values())
                 avg_fps_so_far = frames_processed_count / (iter_end_time - loop_start_time) if (iter_end_time - loop_start_time) > 0 else 0
                 logger.info( f"Frame {frame_idx:>4} | Time: {frame_proc_time_ms:>6.1f}ms | AvgFPS: {avg_fps_so_far:5.2f} | "
                              f"Pipeline: {timing_str} | Tracks: {track_count}" )

            key = cv2.waitKey(config.display_wait_ms) & 0xFF
            if key == ord('q'): logger.info("Quit key pressed."); break
            elif key == ord('p'): logger.info("Paused. Press any key in window..."); cv2.waitKey(0); logger.info("Resuming.")

        end_time = time.time(); total_time = end_time - start_time
        logger.info("--- Pipeline Finished ---")
        logger.info(f"Processed {frames_processed_count} frame indices.")
        if frames_processed_count > 0 and total_time > 0: logger.info(f"Total time: {total_time:.2f}s. Overall Avg FPS: {frames_processed_count / total_time:.2f}")
        else: logger.info("No frames processed or time was zero.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e: logger.critical(f"Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt: logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e: logger.critical(f"Unexpected error: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows(); logger.info("OpenCV windows closed.")
        if torch.cuda.is_available(): logger.info("Clearing CUDA cache."); torch.cuda.empty_cache()
        logger.info("Exiting.")

if __name__ == "__main__":
    main()