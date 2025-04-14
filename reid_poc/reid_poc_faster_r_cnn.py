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

# <<< Set Logging Level to INFO >>>
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


# --- Configuration ---
@dataclass
class PipelineConfig:
    """Configuration settings for the multi-camera pipeline."""
    # Paths
    dataset_base_path: Path = Path(
        os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"))
    reid_model_weights: Path = Path("osnet_x0_25_msmt17.pt")
    tracker_config_path: Optional[Path] = None

    # Dataset
    selected_scene: str = "s10"
    selected_cameras: List[str] = field(default_factory=lambda: ["c09", "c12", "c13", "c16"])

    # Model Params
    person_class_id: int = 1
    detection_confidence_threshold: float = 0.5
    reid_similarity_threshold: float = 0.65
    gallery_ema_alpha: float = 0.9
    reid_refresh_interval_frames: int = 10 # Processed frames

    # Performance
    detection_input_width: Optional[int] = 640
    use_amp: bool = True

    # Tracker
    tracker_type: str = 'bytetrack'

    # Frame Skipping
    frame_skip_rate: int = 1 # Set > 1 to enable skipping

    # Execution
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Visualization
    draw_bounding_boxes: bool = True
    show_track_id: bool = True
    show_global_id: bool = True
    window_name: str = "Multi-Camera Tracking & Re-ID"
    display_wait_ms: int = 1
    max_display_width: int = 1920

    # <<< DEBUG FLAGS - Set to False >>>
    enable_debug_logging: bool = False
    log_raw_detections: bool = False


# --- Type Aliases and Structures --- (Unchanged)
CameraID = str
TrackID = int
GlobalID = int
TrackKey = Tuple[CameraID, TrackID]
FeatureVector = np.ndarray
BoundingBox = np.ndarray # xyxy
Detection = Dict[str, Any]
TrackData = Dict[str, Any]
FrameData = Optional[np.ndarray]
Timings = Dict[str, float]
ScaleFactors = Tuple[float, float]

class ProcessedBatchResult(NamedTuple):
    results_per_camera: Dict[CameraID, List[TrackData]]
    timings: Timings
    processed_this_frame: bool


# --- Helper Functions ---
def sorted_alphanumeric(data: List[str]) -> List[str]:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_compute_device() -> torch.device:
    logger.info("--- Determining Compute Device ---")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"Attempting CUDA: {torch.cuda.get_device_name(device)}")
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA confirmed.")
        except Exception as e:
            logger.warning(f"CUDA failed ({e}). Falling back...")
            device = torch.device("cpu")
    logger.info(f"Using device: {device.type}")
    return device

def calculate_cosine_similarity(feat1: Optional[FeatureVector], feat2: Optional[FeatureVector]) -> float:
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if not np.any(feat1) or not np.any(feat2): return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0
    try:
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0)
        return float(np.clip(similarity, 0.0, 1.0))
    except ValueError: return 0.0
    except Exception as e: logger.error(f"Cosine distance error: {e}"); return 0.0

def _get_reid_device_specifier_string(device: torch.device) -> str:
    if device.type == 'cuda': return str(device.index if device.index is not None else 0)
    if device.type == 'mps': return 'mps'
    return 'cpu'

def _normalize_embedding(embedding: FeatureVector) -> FeatureVector:
    norm = np.linalg.norm(embedding)
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
        # --- Logging config details (INFO level) ---
        if config.detection_input_width: logger.info(f"Detection resizing ENABLED to width: {config.detection_input_width}")
        else: logger.info("Detection resizing DISABLED.")
        if config.use_amp and self.device.type == 'cuda': logger.info("AMP (FP16) for detection ENABLED.")
        elif config.use_amp: logger.warning("AMP requested but device is not CUDA. AMP disabled.")
        else: logger.info("AMP (FP16) for detection DISABLED.")
        if self.config.frame_skip_rate > 1: logger.info(f"Frame skipping ENABLED (1/{self.config.frame_skip_rate} frames).")
        else: logger.info("Frame skipping DISABLED.")
        # Removed debug log flags from INFO level

        # --- State Initialization ---
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.next_global_id: GlobalID = 1
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        self.track_last_reid_frame: Dict[TrackKey, int] = {}
        self.processed_frame_counter: int = 0

        # --- Load Models and Trackers ---
        self.detector, self.detector_transforms = self._load_detector()
        self.reid_model = self._load_reid_model()
        self.trackers = self._initialize_trackers()

    def _load_detector(self) -> Tuple[FasterRCNN, Compose]:
        logger.info("Loading SHARED Faster R-CNN detector...")
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            model.to(self.device)
            model.eval()
            transforms = weights.transforms()
            logger.info("Faster R-CNN Detector loaded successfully.")
            return model, transforms
        except Exception as e:
            logger.critical(f"FATAL ERROR loading Faster R-CNN: {e}", exc_info=True)
            raise RuntimeError("Failed to load object detector") from e

    def _load_reid_model(self) -> Optional[BaseModelBackend]:
        reid_weights = self.config.reid_model_weights
        logger.info(f"Loading SHARED OSNet ReID model from: {reid_weights}")
        if not reid_weights.is_file():
            logger.critical(f"FATAL ERROR: ReID weights file not found: {reid_weights}")
            raise FileNotFoundError(f"ReID weights not found: {reid_weights}")
        try:
            reid_device_specifier = _get_reid_device_specifier_string(self.device)
            reid_model_handler = ReidAutoBackend(weights=reid_weights, device=reid_device_specifier, half=False)
            model = reid_model_handler.model
            if hasattr(model, "warmup"):
                logger.info("Warming up ReID model...")
                model.warmup()
                logger.info("ReID model warmup complete.")
            logger.info("OSNet ReID Model loaded successfully.")
            return model
        except Exception as e:
            logger.critical(f"FATAL ERROR loading ReID model: {e}", exc_info=True)
            raise RuntimeError("Failed to load ReID model") from e

    def _initialize_trackers(self) -> Dict[CameraID, BaseTracker]:
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
                    reid_weights=None,
                    device=tracker_device_str,
                    half=False,
                    per_class=False
                )
                if hasattr(tracker_instance, 'reset'):
                    tracker_instance.reset()
                trackers[cam_id] = tracker_instance
                # Keep this essential INFO log
                logger.info(f"Initialized {self.config.tracker_type} for camera {cam_id} on device '{tracker_device_str}'")
            logger.info(f"Initialized {len(trackers)} tracker instances.")
            return trackers
        except Exception as e:
            logger.critical(f"FATAL ERROR initializing trackers: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize trackers") from e

    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        features: Dict[TrackID, FeatureVector] = {}
        if self.reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.shape[0] == 0:
            return features

        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids = tracked_dets_np[:, 4]

        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4:
            # Keep warnings for potential issues
            logger.warning(f"[FE_EXTRACT] Invalid bbox shape {bboxes_xyxy.shape}. Skip.")
            return features

        try:
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)
            if batch_features is not None and len(batch_features) == len(track_ids):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        features[int(track_ids[i])] = det_feature
        except Exception as e:
            logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features

    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[TrackKey, Optional[GlobalID]]:
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {}
        if not features_per_track:
            return newly_assigned_global_ids

        # Removed debug log start message

        valid_gallery_items = [(gid, emb) for gid, emb in self.reid_gallery.items() if emb is not None and np.isfinite(emb).all() and emb.size > 0]
        valid_gallery_ids, valid_gallery_embeddings = ([], []) if not valid_gallery_items else zip(*valid_gallery_items)
        # Removed debug log gallery size message

        for track_key, new_embedding in features_per_track.items():
            newly_assigned_global_ids[track_key] = None
            if new_embedding is None or not np.isfinite(new_embedding).all() or new_embedding.size == 0:
                logger.warning(f"Skipping ReID for {track_key}: invalid embedding.")
                continue

            best_match_global_id: Optional[GlobalID] = None
            max_similarity = -1.0
            if valid_gallery_ids:
                try:
                    similarities = np.array([calculate_cosine_similarity(new_embedding, gal_emb) for gal_emb in valid_gallery_embeddings])
                    if similarities.size > 0:
                        max_similarity_idx = np.argmax(similarities)
                        max_similarity = similarities[max_similarity_idx]
                        if max_similarity >= self.config.reid_similarity_threshold:
                            best_match_global_id = valid_gallery_ids[max_similarity_idx]
                except Exception as sim_err:
                    logger.error(f"Similarity calculation error for {track_key}: {sim_err}")

            assigned_global_id: Optional[GlobalID] = None
            normalized_new_embedding = _normalize_embedding(new_embedding)
            # Removed debug log prefix variable

            if best_match_global_id is not None: # Case A: Match
                assigned_global_id = best_match_global_id
                current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                if current_gallery_emb is not None:
                    updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb + (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                    self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                else:
                    logger.warning(f"Track {track_key}: Matched GID {assigned_global_id} None in gallery? Overwriting.")
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding
                # Removed debug log decision message
            else: # Case B: No Match
                last_known_global_id = self.track_to_global_id.get(track_key)
                if last_known_global_id is not None and last_known_global_id in self.reid_gallery: # Case B.1: Re-assign
                    assigned_global_id = last_known_global_id
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb + (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                        self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                    else:
                        logger.warning(f"Track {track_key}: Re-assigned GID {assigned_global_id} None in gallery? Overwriting.")
                        self.reid_gallery[assigned_global_id] = normalized_new_embedding
                    # Removed debug log decision message
                else: # Case B.2: New ID
                    assigned_global_id = self.next_global_id
                    self.next_global_id += 1
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding
                    # Removed debug log decision message

            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id

        # Removed debug log finished message
        return newly_assigned_global_ids

    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        # Removed debug log start message
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys:
            new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        # Removed debug log delete message
        for key in keys_to_delete_reid:
            del self.track_last_reid_frame[key]

        keys_to_delete_global = set(self.track_to_global_id.keys()) - current_frame_active_track_keys
        # Removed debug log delete message
        for key in keys_to_delete_global:
            del self.track_to_global_id[key]

    def process_frame_batch_full(self, frames: Dict[CameraID, FrameData], frame_idx: int) -> ProcessedBatchResult:
        """Processes a batch of frames FULLY: Detect -> Track -> Re-ID -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)
        self.processed_frame_counter += 1
        # proc_frame_id = self.processed_frame_counter # No longer needed for logging
        # Removed debug log start message

        # --- Stage 1a: Preprocess ---
        t_prep_start = time.time()
        batch_input_tensors: List[torch.Tensor] = []
        batch_cam_ids: List[CameraID] = []
        batch_original_shapes: List[Tuple[int, int]] = []
        batch_scale_factors: List[ScaleFactors] = []
        for cam_id, frame_bgr in frames.items():
            if frame_bgr is not None and frame_bgr.size > 0:
                original_h, original_w = frame_bgr.shape[:2]
                frame_for_det = frame_bgr
                scale_x, scale_y = 1.0, 1.0
                if self.config.detection_input_width and original_w > self.config.detection_input_width:
                    target_w = self.config.detection_input_width
                    scale = target_w / original_w
                    target_h = int(original_h * scale)
                    try:
                        frame_for_det = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        scale_x = original_w / target_w
                        scale_y = original_h / target_h
                    except Exception as resize_err:
                        logger.warning(f"[{cam_id}] Resize failed: {resize_err}. Using original.")
                        frame_for_det = frame_bgr
                        scale_x, scale_y = 1.0, 1.0
                try:
                    img_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    input_tensor = self.detector_transforms(img_pil)
                    batch_input_tensors.append(input_tensor.to(self.device))
                    batch_cam_ids.append(cam_id)
                    batch_original_shapes.append((original_h, original_w))
                    batch_scale_factors.append((scale_x, scale_y))
                except Exception as transform_err:
                    logger.error(f"[{cam_id}] Preprocessing failed: {transform_err}")
        timings['preprocess'] = time.time() - t_prep_start

        # --- Stage 1b: Batched Detection ---
        t_detect_start = time.time()
        all_predictions: List[Dict[str, torch.Tensor]] = []
        if batch_input_tensors:
            try:
                with torch.no_grad():
                    use_amp_runtime = self.config.use_amp and self.device.type == 'cuda'
                    with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                        all_predictions = self.detector(batch_input_tensors)
            except Exception as e:
                logger.error(f"Detection failed: {e}", exc_info=False)
                all_predictions = []
        timings['detection_batched'] = time.time() - t_detect_start
        # Removed debug log raw detection count

        # --- Stage 1c: Postprocess Detections ---
        t_postproc_start = time.time()
        detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)
        # raw_detections_log variable removed
        if len(all_predictions) == len(batch_cam_ids):
            for i, prediction_dict in enumerate(all_predictions):
                cam_id = batch_cam_ids[i]
                original_h, original_w = batch_original_shapes[i]
                scale_x, scale_y = batch_scale_factors[i]
                try:
                    pred_boxes = prediction_dict['boxes'].cpu().numpy()
                    pred_labels = prediction_dict['labels'].cpu().numpy()
                    pred_scores = prediction_dict['scores'].cpu().numpy()
                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        if label == self.config.person_class_id:
                            x1, y1, x2, y2 = box
                            orig_x1 = max(0.0, x1 * scale_x)
                            orig_y1 = max(0.0, y1 * scale_y)
                            orig_x2 = min(float(original_w - 1), x2 * scale_x)
                            orig_y2 = min(float(original_h - 1), y2 * scale_y)
                            bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32)
                            # Removed raw detection logging
                            if score >= self.config.detection_confidence_threshold:
                                if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1:
                                    detections_per_camera[cam_id].append({'bbox_xyxy': bbox_orig, 'conf': float(score), 'class_id': self.config.person_class_id})
                except Exception as postproc_err:
                    logger.error(f"[{cam_id}] Postprocessing error: {postproc_err}")
        else:
             if batch_input_tensors:
                 logger.error(f"Detection output mismatch: {len(all_predictions)} vs {len(batch_cam_ids)}")
        timings['postprocess_scale'] = time.time() - t_postproc_start
        # Removed debug log filtered detection count

        # --- Stage 1d: Tracking per Camera ---
        t_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        current_frame_active_track_keys: Set[TrackKey] = set()
        tracks_to_extract_features_for: Dict[CameraID, List[np.ndarray]] = defaultdict(list)
        for cam_id in self.camera_ids:
            tracker = self.trackers.get(cam_id)
            if not tracker:
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8))
                continue
            cam_detections = detections_per_camera.get(cam_id, [])
            np_dets = np.empty((0, 6))
            if cam_detections:
                try:
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                except Exception as format_err:
                    logger.error(f"[{cam_id}] Formatting detections error: {format_err}")
            original_frame_bgr = frames.get(cam_id)
            dummy_frame_shape = (1080, 1920)
            if cam_id in batch_cam_ids:
                try:
                    dummy_frame_shape = batch_original_shapes[batch_cam_ids.index(cam_id)]
                except (ValueError, IndexError):
                    pass
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*dummy_frame_shape, 3), dtype=np.uint8)

            # Removed debug log tracker input

            try:
                tracked_dets_np = tracker.update(np_dets, dummy_frame)
                tracked_dets_np = np.array(tracked_dets_np) if tracked_dets_np is not None and len(tracked_dets_np) > 0 else np.empty((0, 8))
                current_frame_tracker_outputs[cam_id] = tracked_dets_np
                # Removed debug log tracker output details
            except Exception as e:
                logger.error(f"[{cam_id}] Tracker update failed: {e}", exc_info=True)
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8))

            # --- Identify active tracks and Re-ID Triggers ---
            if current_frame_tracker_outputs[cam_id].shape[0] > 0:
                previous_processed_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in current_frame_tracker_outputs[cam_id]:
                    if len(track_data) >= 5:
                        try:
                            track_id = int(track_data[4])
                        except (ValueError, IndexError):
                            continue
                        current_track_key: TrackKey = (cam_id, track_id)
                        current_frame_active_track_keys.add(current_track_key)
                        if original_frame_bgr is not None and original_frame_bgr.size > 0:
                            is_newly_seen_since_last_proc = track_id not in previous_processed_cam_track_ids
                            last_reid_attempt_proc_idx = self.track_last_reid_frame.get(current_track_key, -self.config.reid_refresh_interval_frames - 1)
                            # Use current processed frame counter (incremented at the start of this method)
                            proc_frame_id = self.processed_frame_counter
                            is_due_for_refresh = (proc_frame_id - last_reid_attempt_proc_idx) >= self.config.reid_refresh_interval_frames
                            trigger_reid = is_newly_seen_since_last_proc or is_due_for_refresh
                            # Removed debug log reid trigger reason
                            if trigger_reid:
                                tracks_to_extract_features_for[cam_id].append(track_data)
                                self.track_last_reid_frame[current_track_key] = proc_frame_id # Use current counter
        timings['tracking'] = time.time() - t_track_start

        # --- Stage 2: Conditional Feature Extraction ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        if tracks_to_extract_features_for:
             # Removed debug log feature extraction trigger count
            for cam_id, tracks_data_list in tracks_to_extract_features_for.items():
                if tracks_data_list:
                    frame_bgr = frames.get(cam_id)
                    if frame_bgr is not None and frame_bgr.size > 0:
                        try:
                            tracks_data_np = np.array(tracks_data_list)
                            features_this_cam = self._extract_features_for_tracks(frame_bgr, tracks_data_np)
                            for track_id, feature in features_this_cam.items():
                                extracted_features_this_frame[(cam_id, track_id)] = feature
                        except Exception as fe_err:
                            logger.error(f"[{cam_id}] FE call failed: {fe_err}")
        timings['feature_ext'] = time.time() - t_feat_start

        # --- Stage 3: Re-ID Association ---
        t_reid_start = time.time()
        assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        # --- Stage 4: Combine Results ---
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    if len(track_data) >= 7:
                        try:
                            x1, y1, x2, y2 = map(float, track_data[0:4])
                            track_id = int(track_data[4])
                            conf = float(track_data[5])
                            cls = int(track_data[6])
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"[{cam_id}] Parse track data failed {track_data}: {e}")
                            continue
                        current_track_key: TrackKey = (cam_id, track_id)
                        global_id: Optional[GlobalID]
                        if current_track_key in assigned_global_ids_this_cycle:
                            global_id = assigned_global_ids_this_cycle[current_track_key]
                        else:
                            global_id = self.track_to_global_id.get(current_track_key)
                        final_results_per_camera[cam_id].append({
                            'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                            'track_id': track_id, 'global_id': global_id,
                            'conf': conf, 'class_id': cls })

        # --- Stage 5: Update State ---
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch
        # Removed debug log finish message
        return ProcessedBatchResult(
            results_per_camera=dict(final_results_per_camera),
            timings=dict(timings),
            processed_this_frame=True
        )


    def draw_annotations(self, frames: Dict[CameraID, FrameData], processed_results: Dict[CameraID, List[TrackData]]) -> Dict[CameraID, FrameData]:
        annotated_frames: Dict[CameraID, FrameData] = {}
        default_frame_h, default_frame_w = 1080, 1920
        first_valid_frame_found = False
        for frame in frames.values():
            if frame is not None and frame.size > 0:
                default_frame_h, default_frame_w = frame.shape[:2]
                first_valid_frame_found = True
                break

        for cam_id, frame in frames.items():
            current_h, current_w = default_frame_h, default_frame_w
            if frame is None or frame.size == 0:
                placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                annotated_frames[cam_id] = placeholder
                continue
            else:
                annotated_frame = frame.copy()
                current_h, current_w = frame.shape[:2]

            results_for_cam = processed_results.get(cam_id, [])
            for track_info in results_for_cam:
                bbox = track_info.get('bbox_xyxy')
                track_id = track_info.get('track_id')
                global_id = track_info.get('global_id')
                if bbox is None: continue

                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(current_w - 1, x2); y2 = min(current_h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue

                color = (200, 200, 200)
                if global_id is not None:
                    seed = int(global_id) * 3 + 5
                    color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)

                if self.config.draw_bounding_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label_parts = []
                if self.config.show_track_id and track_id is not None:
                    label_parts.append(f"T:{track_id}")
                if self.config.show_global_id:
                    label_parts.append(f"G:{global_id if global_id is not None else '?'}")
                label = " ".join(label_parts)

                if label:
                    font_face, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, font_face, font_scale, thickness + 1)
                    label_y_pos = y1 - baseline - 5
                    if label_y_pos < text_h: label_y_pos = y2 + text_h + 5
                    label_y_pos = max(text_h + baseline, min(label_y_pos, current_h - baseline - 1))
                    label_x_pos = max(0, x1)
                    bg_x1 = label_x_pos; bg_y1 = label_y_pos - text_h - baseline
                    bg_x2 = label_x_pos + text_w; bg_y2 = label_y_pos + baseline
                    bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
                    bg_x2, bg_y2 = min(current_w - 1, bg_x2), min(current_h - 1, bg_y2)
                    if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
                        cv2.putText(annotated_frame, label, (label_x_pos, label_y_pos), font_face, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            annotated_frames[cam_id] = annotated_frame
        return annotated_frames


# --- Main Execution Functions ---
def setup_paths_and_config() -> PipelineConfig:
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve()
    config = PipelineConfig()
    config.device = get_compute_device()
    if config.frame_skip_rate < 1:
        logger.warning(f"Invalid frame_skip_rate ({config.frame_skip_rate}). Setting to 1.")
        config.frame_skip_rate = 1
    tracker_filename = f"{config.tracker_type}.yaml"
    potential_paths = [ BOXMOT_PATH / "configs" / tracker_filename, script_dir / "configs" / tracker_filename, script_dir / tracker_filename, Path.cwd() / "configs" / tracker_filename, Path.cwd() / tracker_filename ]
    found_path = next((p for p in potential_paths if p.is_file()), None)
    if not found_path:
        logger.warning(f"Tracker config '{tracker_filename}' not found. Searching...")
        potential_config_dirs = { p.parent for p in potential_paths }
        found_yaml = next( (yaml_file for dir_path in potential_config_dirs if dir_path.exists() for yaml_file in dir_path.glob('*.yaml') if yaml_file.is_file()), None)
        if found_yaml:
            logger.warning(f"Using fallback tracker config: {found_yaml}.")
            found_path = found_yaml
        else:
            checked_dirs_str = "\n - ".join(map(str, potential_config_dirs))
            raise FileNotFoundError(f"No tracker config (.yaml) found for '{config.tracker_type}' or fallback in:\n - {checked_dirs_str}")
    config.tracker_config_path = found_path
    logger.info(f"Using tracker config: {config.tracker_config_path}")
    if not config.reid_model_weights.is_file():
        logger.info(f"ReID weights '{config.reid_model_weights.name}' not found. Searching...")
        potential_reid_paths = [ script_dir / config.reid_model_weights.name, Path.cwd() / config.reid_model_weights.name, script_dir / "weights" / config.reid_model_weights.name, Path.cwd() / "weights" / config.reid_model_weights.name, config.reid_model_weights ]
        found_reid_path = next((p for p in potential_reid_paths if p.is_file()), None)
        if not found_reid_path:
            raise FileNotFoundError(f"ReID weights '{config.reid_model_weights.name}' not found.")
        config.reid_model_weights = found_reid_path.resolve()
    logger.info(f"Using ReID weights: {config.reid_model_weights}")
    if not config.dataset_base_path.is_dir():
        raise FileNotFoundError(f"Dataset base path not found: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")
    logger.info("Configuration setup complete.")
    # logger.info(f"Final Config: {config}") # Removed final config log for brevity
    return config

def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    logger.info("--- Loading Dataset Information ---")
    camera_dirs: Dict[CameraID, Path] = {}
    valid_cameras: List[CameraID] = []
    base_scene_path = config.dataset_base_path / "train" / "train" / config.selected_scene
    if not base_scene_path.is_dir():
        alt_scene_path = config.dataset_base_path / config.selected_scene
        if alt_scene_path.is_dir(): base_scene_path = alt_scene_path; logger.warning(f"Using alternative scene path: {base_scene_path}")
        elif (config.dataset_base_path / "train" / config.selected_scene).is_dir(): base_scene_path = config.dataset_base_path / "train" / config.selected_scene; logger.warning(f"Using alternative scene path: {base_scene_path}")
        else: raise FileNotFoundError(f"Scene directory not found: Check paths like {config.dataset_base_path / 'train' / 'train' / config.selected_scene} or {config.dataset_base_path / config.selected_scene}")
    logger.info(f"Using scene path: {base_scene_path}")

    for cam_id in config.selected_cameras:
        potential_img_dirs = [ base_scene_path / cam_id / "img1", base_scene_path / cam_id / "rgb", base_scene_path / cam_id ]
        found_dir = None
        for img_dir in potential_img_dirs:
            if img_dir.is_dir():
                image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                if image_files: found_dir = img_dir; logger.info(f"Found valid dir for {cam_id}: {found_dir} ({len(image_files)} images)"); break
        if found_dir: camera_dirs[cam_id] = found_dir; valid_cameras.append(cam_id)
        else: logger.warning(f"No valid image directory found for {cam_id}. Skipping.")
    if not valid_cameras: raise RuntimeError("No valid camera data sources found.")
    config.selected_cameras = valid_cameras
    logger.info(f"Processing cameras: {valid_cameras}")

    image_filenames: List[str] = []
    first_cam_dir = camera_dirs[valid_cameras[0]]
    try:
        image_filenames = sorted_alphanumeric([f.name for f in first_cam_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
        if not image_filenames: raise ValueError(f"No images found in {first_cam_dir}")
        logger.info(f"Found {len(image_filenames)} frames based on {valid_cameras[0]}.")
    except Exception as e: logger.critical(f"Failed list images from {first_cam_dir}: {e}", exc_info=True); raise RuntimeError(f"Failed list images: {e}") from e
    return camera_dirs, image_filenames

def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
    current_frames: Dict[CameraID, FrameData] = {}
    for cam_id, cam_dir_path in camera_dirs.items():
        image_path = cam_dir_path / filename
        img = None
        if image_path.is_file():
            try:
                img = cv2.imread(str(image_path))
                if img is None or img.size == 0: logger.warning(f"[{cam_id}] Load failed: {image_path}"); img = None
            except Exception as e: logger.error(f"[{cam_id}] Read error {image_path}: {e}"); img = None
        current_frames[cam_id] = img
    return current_frames

def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    valid_annotated = [f for f in annotated_frames.values() if f is not None and f.size > 0]
    if not valid_annotated:
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated)
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))
        target_h, target_w = valid_annotated[0].shape[:2]
        combined_h, combined_w = rows * target_h, cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        frame_idx_display = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx_display < num_cams:
                    frame = valid_annotated[frame_idx_display]
                    if frame.shape[:2] != (target_h, target_w):
                        try: frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err: logger.warning(f"Grid resize error: {resize_err}."); frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    try: combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = frame
                    except ValueError as slice_err: logger.error( f"Grid placement error: {slice_err}.")
                    frame_idx_display += 1
        disp_h, disp_w = combined_display.shape[:2]
        if disp_w > max_width:
            try: scale = max_width / disp_w; disp_h_new, disp_w_new = int(disp_h * scale), max_width; combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err: logger.error(f"Final display resize failed: {final_resize_err}")
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
         cv2.imshow(window_name, combined_display)


def main():
    """Main execution function."""
    pipeline: Optional[MultiCameraPipeline] = None
    last_batch_result: Optional[ProcessedBatchResult] = None
    try:
        config = setup_paths_and_config()
        # Set log level based on config AFTER config is loaded
        # logger.setLevel(logging.DEBUG if config.enable_debug_logging else logging.INFO) # Already set at top
        camera_dirs, image_filenames = load_dataset_info(config)
        pipeline = MultiCameraPipeline(config)
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

        logger.info("--- Starting Frame Processing Loop ---")
        total_frames_loaded = 0
        total_frames_processed = 0
        loop_start_time = time.perf_counter()

        for frame_idx, current_filename in enumerate(image_filenames):
            iter_start_time = time.perf_counter()

            # --- Load Frames ---
            current_frames = load_frames_for_batch(camera_dirs, current_filename)
            if not any(f is not None for f in current_frames.values()):
                # Keep warning for load failures
                logger.warning(f"Frame {frame_idx}: No valid images loaded for '{current_filename}'. Skipping.")
                continue
            total_frames_loaded += 1

            # --- Frame Skipping Logic ---
            process_this_frame = (frame_idx % config.frame_skip_rate == 0)
            current_batch_timings = defaultdict(float)
            # processing_status = "PROC" if process_this_frame else "SKIP" # Removed

            if process_this_frame:
                if pipeline is not None:
                    batch_result = pipeline.process_frame_batch_full(current_frames, frame_idx)
                    last_batch_result = batch_result
                    current_batch_timings = batch_result.timings # Store timings from processing
                    total_frames_processed += 1
                else:
                    logger.error("Pipeline not initialized!")
                    continue
            else: # Skipped frame
                 # Minimal timing for skipped frame overhead
                 current_batch_timings['skipped_frame_overhead'] = (time.perf_counter() - iter_start_time)

            # --- Annotate and Display ---
            display_frames = current_frames
            results_to_draw = {}
            if last_batch_result:
                results_to_draw = last_batch_result.results_per_camera
            # Removed debug log about missing results

            if pipeline:
                annotated_frames = pipeline.draw_annotations(display_frames, results_to_draw)
                display_combined_frames(config.window_name, annotated_frames, config.max_display_width)
            else:
                display_combined_frames(config.window_name, current_frames, config.max_display_width)


            # --- Logging and Timing ---
            iter_end_time = time.perf_counter()
            frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000
            current_loop_duration = iter_end_time - loop_start_time
            avg_display_fps = total_frames_loaded / current_loop_duration if current_loop_duration > 0 else 0
            avg_processing_fps = total_frames_processed / current_loop_duration if current_loop_duration > 0 else 0

            # <<< Simplified Periodic Logging >>>
            if frame_idx < 10 or frame_idx % 50 == 0:
                track_count = 0
                if last_batch_result:
                    track_count = sum(len(tracks) for tracks in last_batch_result.results_per_camera.values())

                # Construct the pipeline timing string only if the frame was processed
                pipeline_timing_str = ""
                if process_this_frame and current_batch_timings:
                    # Selectively include main pipeline stage timings
                    stages_to_log = ['preprocess', 'detection_batched', 'postprocess_scale', 'tracking', 'feature_ext', 'reid', 'total']
                    pipeline_timings = {k: v for k, v in current_batch_timings.items() if k in stages_to_log}
                    pipeline_timing_str = " | Pipeline: " + " | ".join([f"{k}={v * 1000:.1f}ms" for k, v in pipeline_timings.items() if v > 0.0001])

                logger.info(
                    f"Frame {frame_idx:<4} Batch Time: {frame_proc_time_ms:>6.1f}ms | AvgFPS: {avg_display_fps:5.2f} | Tracks: {track_count:<3}{pipeline_timing_str}"
                )

            # --- User Input ---
            key = cv2.waitKey(config.display_wait_ms) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed.")
                break
            elif key == ord('p'):
                logger.info("Paused.")
                cv2.waitKey(0)
                logger.info("Resuming.")
            if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Display window closed.")
                break

        # --- End of Loop ---
        loop_end_time = time.perf_counter()
        total_time = loop_end_time - loop_start_time
        logger.info("--- Pipeline Finished ---")
        logger.info(f"Loaded {total_frames_loaded} batches, Processed {total_frames_processed} batches.") # Changed log message slightly
        if total_frames_loaded > 0 and total_time > 0.01:
             final_avg_display_fps = total_frames_loaded / total_time
             final_avg_processing_fps = total_frames_processed / total_time
             logger.info(f"Total time: {total_time:.2f}s. Overall Avg Display FPS: {final_avg_display_fps:.2f}, Avg Processing FPS: {final_avg_processing_fps:.2f}") # Combined FPS log
        else:
            logger.info("No frames processed / time too short for FPS.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e:
        logger.critical(f"Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...") # Changed log message
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
        if torch.cuda.is_available():
            logger.info("Releasing GPU resources...")
            del pipeline
            del last_batch_result
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        logger.info("Exiting script.")

if __name__ == "__main__":
    main()