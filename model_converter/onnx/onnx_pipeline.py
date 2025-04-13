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
# torchvision FasterRCNN related imports are now only needed for transforms, not the model itself
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights # Keep for transforms
# Remove FasterRCNN model import if not needed elsewhere
# from torchvision.models.detection import FasterRCNN
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

# --- ONNX Runtime Import ---
try:
    import onnxruntime as ort
except ImportError:
    logger.critical("ONNX Runtime not found. Please install it: pip install onnxruntime-gpu (for GPU) or onnxruntime (for CPU)")
    sys.exit(1)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class PipelineConfig:
    """Configuration settings for the multi-camera pipeline using ONNX Detector."""
    # Paths
    dataset_base_path: Path = Path(os.getenv("MTMMC_PATH", "D:/MTMMC" if sys.platform == "win32" else "/Volumes/HDD/MTMMC"))
    reid_model_weights: Path = Path("../osnet_x0_25_msmt17.pt")
    tracker_config_path: Optional[Path] = None # Set dynamically
    onnx_model_path: Path = Path("onnx/onnx_models/detector_dynamic.onnx") # Path to the exported ONNX model

    # Dataset Selection
    selected_scene: str = "s10"
    selected_cameras: List[str] = field(default_factory=lambda: ["c09", "c12", "c13", "c16"])

    # Model Parameters
    person_class_id: int = 1
    detection_confidence_threshold: float = 0.5
    reid_similarity_threshold: float = 0.65
    gallery_ema_alpha: float = 0.9
    reid_refresh_interval_frames: int = 10

    # --- Performance Optimizations ---
    # Input width used for resizing BEFORE transforms and ONNX inference
    detection_input_width: Optional[int] = 640
    # use_amp is no longer directly applicable to ONNX Runtime in this simple way
    # ORT manages its own optimizations (FP16 may be used via TensorRT EP if enabled)
    # use_amp: bool = True

    # Tracker Type
    tracker_type: str = 'bytetrack'

    # Execution Device (Mainly for ReID and potentially preprocessing)
    # ONNX Runtime will use its own providers (CUDA, CPU)
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Visualization
    draw_bounding_boxes: bool = True
    show_track_id: bool = True
    show_global_id: bool = True
    window_name: str = "Multi-Camera Tracking & Re-ID (ONNX Detector)"
    display_wait_ms: int = 1
    max_display_width: int = 1920

# --- Type Aliases and Structures --- (Mostly unchanged)
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
ScaleFactors = Tuple[float, float]

class ProcessedBatchResult(NamedTuple):
    results_per_camera: Dict[CameraID, List[TrackData]]
    timings: Timings

# --- Helper Functions --- (sorted_alphanumeric, calculate_cosine_similarity, etc. are unchanged) ---
def sorted_alphanumeric(data: List[str]) -> List[str]:
    """Sorts a list of strings alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_compute_device() -> torch.device:
    """Selects and returns the best available compute device (primarily for non-detector parts)."""
    logger.info("--- Determining Compute Device ---")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            logger.info(f"Attempting to use CUDA device: {torch.cuda.get_device_name(device)}")
            _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
            logger.info("CUDA device confirmed (available for PyTorch ops like ReID).")
            return device
        except Exception as e:
            logger.warning(f"CUDA reported available, but test failed ({e}). Falling back...")

    logger.info("Using CPU device for PyTorch operations.")
    return torch.device("cpu")

# --- _get_reid_device_specifier_string, _normalize_embedding remain the same ---
def _get_reid_device_specifier_string(device: torch.device) -> str:
    if device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        return str(idx)
    elif device.type == 'mps': return 'mps'
    return 'cpu'

def _normalize_embedding(embedding: FeatureVector) -> FeatureVector:
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-6)

def calculate_cosine_similarity(feat1: Optional[FeatureVector], feat2: Optional[FeatureVector]) -> float:
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten(); feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if not np.any(feat1) or not np.any(feat2): return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0
    try:
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0)
        return float(np.clip(similarity, 0.0, 1.0))
    except ValueError as e: return 0.0
    except Exception as e: return 0.0
# --- Main Processing Class ---

class MultiCameraPipeline:
    """Handles multi-camera pipeline using ONNX for detection."""

    def __init__(self, config: PipelineConfig):
        """Initializes models, trackers, ONNX session, and state."""
        self.config = config
        self.device = config.device # Device for ReID, etc.
        self.camera_ids = config.selected_cameras
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids}")
        logger.info(f"Primary PyTorch device (for ReID): {self.device}")
        if config.detection_input_width: logger.info(f"Detection resizing ENABLED to width: {config.detection_input_width}")
        else: logger.info("Detection resizing DISABLED.")

        # Load ONNX detector session and required transforms
        self.detector_session, self.detector_input_names, self.detector_output_names, self.detector_transforms = self._load_onnx_detector()

        # Load ReID model (still PyTorch/BoxMOT)
        self.reid_model = self._load_reid_model()
        # Initialize Trackers
        self.trackers = self._initialize_trackers()

        # State Management (unchanged)
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.next_global_id: GlobalID = 1
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        self.track_last_reid_frame: Dict[TrackKey, int] = {}

    def _load_onnx_detector(self) -> Tuple[ort.InferenceSession, List[str], List[str], Compose]:
        """Loads the ONNX detector model and the necessary preprocessing transforms."""
        logger.info(f"Loading ONNX detector from: {self.config.onnx_model_path}")
        if not self.config.onnx_model_path.is_file():
            logger.critical(f"FATAL ERROR: ONNX model file not found at {self.config.onnx_model_path}")
            raise FileNotFoundError(f"ONNX model not found: {self.config.onnx_model_path}")

        # --- Load ONNX Runtime Session ---
        providers = []
        # Prioritize CUDA Execution Provider if GPU is available and onnxruntime-gpu is installed
        if self.device.type == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
             providers.append(('CUDAExecutionProvider', {'device_id': self.device.index or 0}))
             logger.info("Using CUDAExecutionProvider for ONNX Runtime.")
        providers.append('CPUExecutionProvider') # Always include CPU as fallback
        logger.info(f"ONNX Runtime Providers: {providers}")

        try:
            session = ort.InferenceSession(str(self.config.onnx_model_path), providers=providers)
            logger.info("ONNX Runtime session created successfully.")
        except Exception as e:
            logger.critical(f"FATAL ERROR loading ONNX model: {e}", exc_info=True)
            logger.error("Ensure the ONNX model is valid and compatible with the installed ONNX Runtime version.")
            logger.error(f"Available providers: {ort.get_available_providers()}")
            raise RuntimeError("Failed to load ONNX detector session") from e

        # --- Get Input/Output Names ---
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        logger.info(f"ONNX Model Inputs: {input_names}")
        logger.info(f"ONNX Model Outputs: {output_names}")
        # Basic check: Ensure standard Faster R-CNN outputs are present
        if not all(name in output_names for name in ["boxes", "labels", "scores"]):
             logger.warning("Standard output names ('boxes', 'labels', 'scores') not found in ONNX model outputs. Postprocessing might fail.")
        # Check input expectations
        if len(input_names) != 1:
             logger.warning(f"Expected 1 ONNX input, but found {len(input_names)}: {input_names}. Inference logic might need adjustment.")
        else:
            input_shape = session.get_inputs()[0].shape # Get expected shape [usually includes dynamic axes like 'batch', 'C', 'H', 'W']
            logger.info(f"ONNX Model Expected Input ('{input_names[0]}') Shape: {input_shape}")
            # Check if the rank seems correct based on the error we previously saw
            if len(input_shape) != 3:
                 logger.warning(f"ONNX input shape {input_shape} does not have rank 3 (C, H, W). The previous fix might be incorrect if the model expects batching.")


        # --- Get Preprocessing Transforms ---
        # We still need the *same* transforms that were used *before* exporting the model
        # Assuming these transforms prepare the image correctly for the exported model
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            transforms = weights.transforms()
            logger.info("Loaded original Faster R-CNN preprocessing transforms.")
        except Exception as e:
            logger.error(f"Failed to load default transforms: {e}. Using basic ToTensor.")
            # Fallback or define necessary transforms manually if needed
            transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


        return session, input_names, output_names, transforms

    def _load_reid_model(self) -> Optional[BaseModelBackend]:
        """Loads the OSNet Re-Identification model (Unchanged)."""
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
            if hasattr(model, "warmup"): model.warmup()
            logger.info("OSNet ReID Model loaded successfully.")
            return model
        except Exception as e:
            logger.critical(f"FATAL ERROR loading ReID model: {e}", exc_info=True)
            raise RuntimeError("Failed to load ReID model") from e

    def _initialize_trackers(self) -> Dict[CameraID, BaseTracker]:
        """Initializes a tracker instance for each camera (Unchanged)."""
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


    # --- Feature Extraction, ReID Association, State Update (Unchanged) ---
    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        """Extracts Re-ID features for a given set of tracked detections."""
        features: Dict[TrackID, FeatureVector] = {}
        if self.reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.shape[0] == 0:
            return features
        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids = tracked_dets_np[:, 4]
        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4: return features
        try:
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)
            if batch_features is not None and len(batch_features) == len(track_ids):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        current_track_id: TrackID = int(track_ids[i])
                        features[current_track_id] = det_feature
        except Exception as e: logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features

    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[TrackKey, Optional[GlobalID]]:
        """Compares features of triggered tracks against the gallery to assign Global IDs."""
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
                    max_similarity = similarities[max_similarity_idx]
                    if max_similarity >= self.config.reid_similarity_threshold:
                        best_match_global_id = valid_gallery_ids[max_similarity_idx]
                except Exception as sim_err: logger.error(f"Similarity calculation error for {track_key}: {sim_err}")
            assigned_global_id: Optional[GlobalID] = None
            normalized_new_embedding = _normalize_embedding(new_embedding)
            if best_match_global_id is not None:
                assigned_global_id = best_match_global_id
                current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                if current_gallery_emb is not None:
                    updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb + (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                    self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                else: self.reid_gallery[assigned_global_id] = normalized_new_embedding
            else:
                last_known_global_id = self.track_to_global_id.get(track_key)
                if last_known_global_id is not None and last_known_global_id in self.reid_gallery:
                    assigned_global_id = last_known_global_id
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb + (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                        self.reid_gallery[assigned_global_id] = _normalize_embedding(updated_embedding)
                    else: self.reid_gallery[assigned_global_id] = normalized_new_embedding
                else:
                    assigned_global_id = self.next_global_id; self.next_global_id += 1
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding
            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id
        return newly_assigned_global_ids

    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """Updates the 'last seen' state and removes stale entries."""
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys: new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen
        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_reid: del self.track_last_reid_frame[key]
        keys_to_delete_global = set(self.track_to_global_id.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_global: del self.track_to_global_id[key]

    def process_frame_batch(self, frames: Dict[CameraID, FrameData], frame_idx: int) -> ProcessedBatchResult:
        """Processes a batch of frames using ONNX detector."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)

        # --- Stage 1a: Preprocess Frames for Batch Detection (Similar to before) ---
        t_prep_start = time.time()
        batch_input_list_np: List[np.ndarray] = [] # List of NumPy arrays for ONNX
        batch_cam_ids: List[CameraID] = []
        batch_original_shapes: List[Tuple[int, int]] = []
        batch_scale_factors: List[ScaleFactors] = []

        for cam_id, frame_bgr in frames.items():
            if frame_bgr is not None and frame_bgr.size > 0:
                original_h, original_w = frame_bgr.shape[:2]
                frame_for_det = frame_bgr
                scale_x, scale_y = 1.0, 1.0

                # --- Resizing (Optional) ---
                if self.config.detection_input_width and original_w > self.config.detection_input_width:
                    target_w = self.config.detection_input_width
                    scale = target_w / original_w
                    target_h = int(original_h * scale)
                    try:
                        frame_for_det = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        scale_x = original_w / target_w; scale_y = original_h / target_h
                    except Exception as resize_err:
                        logger.warning(f"[{cam_id}] Resizing failed: {resize_err}. Using original frame.")
                        frame_for_det = frame_bgr; scale_x, scale_y = 1.0, 1.0

                # --- Preprocessing (Transforms -> NumPy) ---
                try:
                    img_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    input_tensor = self.detector_transforms(img_pil)
                    input_np = input_tensor.cpu().numpy() # Shape [C, H, W]

                    batch_input_list_np.append(input_np) # Add the [C, H, W] numpy array

                    batch_cam_ids.append(cam_id)
                    batch_original_shapes.append((original_h, original_w))
                    batch_scale_factors.append((scale_x, scale_y))
                except Exception as transform_err:
                     logger.error(f"[{cam_id}] Preprocessing/Transform failed: {transform_err}")

        timings['preprocess'] = time.time() - t_prep_start

        # --- Stage 1b: Batched Detection with ONNX Runtime ---
        t_detect_start = time.time()
        all_onnx_outputs: List[List[np.ndarray]] = [] # List of outputs per image

        if batch_input_list_np and self.detector_input_names:
            input_name = self.detector_input_names[0] # Assume the first input name is the one we need

            # Run inference one image at a time because the ONNX model likely expects [C, H, W] input
            logger.debug(f"Running ONNX inference for {len(batch_input_list_np)} images individually.")
            for i, input_np_single in enumerate(batch_input_list_np):
                # input_np_single already has the expected shape [C, H, W] (rank 3)
                if input_np_single.ndim != 3:
                     logger.error(f"[{batch_cam_ids[i]}] Unexpected input dimension {input_np_single.ndim} before ONNX run. Expected 3.")
                     continue # Skip this frame

                # Feed the rank 3 tensor directly
                input_feed_single = {input_name: input_np_single}
                try:
                    # Run returns list: [boxes_np, labels_np, scores_np] for this single image
                    onnx_outputs_single = self.detector_session.run(self.detector_output_names, input_feed_single)
                    all_onnx_outputs.append(onnx_outputs_single)
                except ort.capi.onnxruntime_pybind11_state.InvalidArgument as e_rank:
                     logger.error(f"[{batch_cam_ids[i]}] ONNX Rank Error during inference: {e_rank}")
                     logger.error(f"Input shape provided: {input_np_single.shape}")
                     # This error should not happen with the fix, but good to log if it does
                     break # Stop processing batch if rank error persists
                except Exception as e:
                     logger.error(f"[{batch_cam_ids[i]}] ONNX Runtime inference failed: {e}", exc_info=True)
                     # Don't append if error occurs for this frame

        timings['detection_batched'] = time.time() - t_detect_start

        # --- Stage 1c: Postprocess Detections & Scale Boxes ---
        t_postproc_start = time.time()
        detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)

        # Check if number of outputs matches number of *successfully processed* inputs
        if len(all_onnx_outputs) <= len(batch_cam_ids): # Allow for errors during inference loop
            processed_inputs_count = len(all_onnx_outputs)
            for i in range(processed_inputs_count): # Process outputs for successfully inferred images
                onnx_outputs_single = all_onnx_outputs[i]
                # Get corresponding cam_id, shapes, scales for the successful inference
                cam_id = batch_cam_ids[i]
                original_h, original_w = batch_original_shapes[i]
                scale_x, scale_y = batch_scale_factors[i]

                try:
                    output_dict = {name: data for name, data in zip(self.detector_output_names, onnx_outputs_single)}
                    pred_boxes = output_dict.get("boxes", np.empty((0, 4)))
                    pred_labels = output_dict.get("labels", np.empty((0,)))
                    pred_scores = output_dict.get("scores", np.empty((0,)))

                    if not (pred_boxes.ndim == 2 and pred_boxes.shape[1] == 4):
                        logger.warning(f"[{cam_id}] Unexpected ONNX boxes shape: {pred_boxes.shape}")
                        continue
                    if not (pred_labels.ndim == 1 and pred_scores.ndim == 1 and pred_boxes.shape[0] == pred_labels.shape[0] == pred_scores.shape[0]):
                         logger.warning(f"[{cam_id}] ONNX output shape mismatch or empty: Boxes {pred_boxes.shape}, Labels {pred_labels.shape}, Scores {pred_scores.shape}")
                         continue

                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        if label == self.config.person_class_id and score >= self.config.detection_confidence_threshold:
                            x1, y1, x2, y2 = box
                            orig_x1 = max(0.0, x1 * scale_x); orig_y1 = max(0.0, y1 * scale_y)
                            orig_x2 = min(float(original_w - 1), x2 * scale_x); orig_y2 = min(float(original_h - 1), y2 * scale_y)
                            if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1:
                                detections_per_camera[cam_id].append({
                                    'bbox_xyxy': np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32),
                                    'conf': float(score), 'class_id': self.config.person_class_id
                                })
                except Exception as postproc_err:
                     logger.error(f"[{cam_id}] Error postprocessing ONNX detections: {postproc_err}", exc_info=False)
        else:
            # This case should ideally not be reached if errors are handled in the loop
            logger.error(f"ONNX output count mismatch: Processed {len(batch_cam_ids)} inputs, got {len(all_onnx_outputs)} output sets.")

        timings['postprocess_scale'] = time.time() - t_postproc_start

        # --- Stage 1d: Tracking per Camera (Unchanged) ---
        t_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        current_frame_active_track_keys: Set[TrackKey] = set()
        tracks_to_extract_features_for: Dict[CameraID, List[np.ndarray]] = defaultdict(list)

        for cam_id in self.camera_ids:
            tracker = self.trackers.get(cam_id)
            if not tracker: current_frame_tracker_outputs[cam_id] = np.empty((0, 8)); continue
            cam_detections = detections_per_camera.get(cam_id, [])
            np_dets = np.empty((0, 6))
            if cam_detections:
                try: np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                except Exception as format_err: logger.error(f"[{cam_id}] Error formatting detections for tracker: {format_err}")
            original_frame_bgr = frames.get(cam_id)
            # Find original shape info safely
            try: original_shape_idx = batch_cam_ids.index(cam_id)
            except ValueError: dummy_shape_h, dummy_shape_w = 100, 100 # Fallback if cam_id wasn't processed
            else: dummy_shape_h, dummy_shape_w = batch_original_shapes[original_shape_idx]
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((dummy_shape_h, dummy_shape_w, 3), dtype=np.uint8)
            try:
                tracked_dets_np = tracker.update(np_dets, dummy_frame)
                current_frame_tracker_outputs[cam_id] = np.array(tracked_dets_np) if tracked_dets_np is not None and len(tracked_dets_np) > 0 else np.empty((0, 8))
            except Exception as e:
                logger.error(f"[{cam_id}] Tracker update failed: {e}"); current_frame_tracker_outputs[cam_id] = np.empty((0, 8))
            if current_frame_tracker_outputs[cam_id].shape[0] > 0:
                previous_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in current_frame_tracker_outputs[cam_id]:
                    if len(track_data) >= 5:
                        try: track_id = int(track_data[4])
                        except (ValueError, IndexError): continue
                        current_track_key: TrackKey = (cam_id, track_id)
                        current_frame_active_track_keys.add(current_track_key)
                        if original_frame_bgr is not None and original_frame_bgr.size > 0:
                            is_newly_seen = track_id not in previous_cam_track_ids
                            last_reid_attempt = self.track_last_reid_frame.get(current_track_key, -self.config.reid_refresh_interval_frames)
                            is_due_for_refresh = (frame_idx - last_reid_attempt) >= self.config.reid_refresh_interval_frames
                            if is_newly_seen or is_due_for_refresh:
                                tracks_to_extract_features_for[cam_id].append(track_data)
                                self.track_last_reid_frame[current_track_key] = frame_idx
        timings['tracking'] = time.time() - t_track_start

        # --- Stage 2: Conditional Feature Extraction (Unchanged) ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        for cam_id, tracks_data_list in tracks_to_extract_features_for.items():
            if tracks_data_list:
                frame_bgr = frames.get(cam_id)
                if frame_bgr is not None and frame_bgr.size > 0:
                    try:
                        tracks_data_np = np.array(tracks_data_list)
                        features_this_cam = self._extract_features_for_tracks(frame_bgr, tracks_data_np)
                        for track_id, feature in features_this_cam.items(): extracted_features_this_frame[(cam_id, track_id)] = feature
                    except Exception as fe_err: logger.error(f"[{cam_id}] Feature extraction call failed: {fe_err}")
        timings['feature_ext'] = time.time() - t_feat_start

        # --- Stage 3: Conditional Re-ID Association (Unchanged) ---
        t_reid_start = time.time()
        assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        # --- Stage 4: Combine Tracking Results with Global IDs (Unchanged) ---
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    if len(track_data) >= 7:
                        try: x1, y1, x2, y2 = map(float, track_data[0:4]); track_id = int(track_data[4]); conf = float(track_data[5]); cls = int(track_data[6])
                        except (ValueError, IndexError, TypeError) as e: continue
                        current_track_key: TrackKey = (cam_id, track_id)
                        global_id = assigned_global_ids_this_cycle.get(current_track_key, self.track_to_global_id.get(current_track_key))
                        final_results_per_camera[cam_id].append({
                            'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                            'track_id': track_id, 'global_id': global_id, 'conf': conf, 'class_id': cls
                        })

        # --- Stage 5: Update State and Cleanup (Unchanged) ---
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch
        return ProcessedBatchResult(results_per_camera=dict(final_results_per_camera), timings=dict(timings))

    # --- draw_annotations (Unchanged) ---
    def draw_annotations(self, frames: Dict[CameraID, FrameData], processed_results: Dict[CameraID, List[TrackData]]) -> Dict[CameraID, FrameData]:
        annotated_frames: Dict[CameraID, FrameData] = {}
        default_frame_h, default_frame_w = 1080, 1920
        first_valid_frame_found = False
        for frame in frames.values():
            if frame is not None and frame.size > 0: default_frame_h, default_frame_w = frame.shape[:2]; first_valid_frame_found = True; break
        if not first_valid_frame_found: logger.warning("No valid frames in batch for annotation sizing.")
        for cam_id, frame in frames.items():
            current_h, current_w = default_frame_h, default_frame_w
            if frame is None or frame.size == 0:
                placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8); cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA); annotated_frames[cam_id] = placeholder; continue
            else: annotated_frame = frame.copy(); current_h, current_w = frame.shape[:2]
            results_for_cam = processed_results.get(cam_id, [])
            for track_info in results_for_cam:
                bbox = track_info.get('bbox_xyxy'); track_id = track_info.get('track_id'); global_id = track_info.get('global_id')
                if bbox is None: continue
                x1, y1, x2, y2 = map(int, bbox); x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(current_w - 1, x2), min(current_h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue
                color = (200, 200, 200)
                if global_id is not None: seed = int(global_id) * 3 + 5; color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)
                if self.config.draw_bounding_boxes: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label_parts = [];
                if self.config.show_track_id and track_id is not None: label_parts.append(f"T:{track_id}")
                if self.config.show_global_id: label_parts.append(f"G:{global_id if global_id is not None else '?'}")
                label = " ".join(label_parts)
                if label:
                    font_face, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1; (text_w, text_h), baseline = cv2.getTextSize(label, font_face, font_scale, thickness + 1)
                    label_y_pos = y1 - baseline - 5;
                    if label_y_pos < text_h: label_y_pos = y2 + text_h + 5
                    label_y_pos = max(text_h + baseline, min(label_y_pos, current_h - baseline - 1)); label_x_pos = max(0, x1)
                    bg_x1, bg_y1 = label_x_pos, label_y_pos - text_h - baseline; bg_x2, bg_y2 = label_x_pos + text_w, label_y_pos + baseline
                    bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1); bg_x2, bg_y2 = min(current_w - 1, bg_x2), min(current_h - 1, bg_y2)
                    if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED); cv2.putText(annotated_frame, label, (label_x_pos, label_y_pos), font_face, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            annotated_frames[cam_id] = annotated_frame
        return annotated_frames

# --- Main Execution Functions --- (Setup and loop structure identical, reads new config)
def setup_paths_and_config() -> PipelineConfig:
    """Determines paths, creates, and returns the pipeline configuration."""
    logger.info("--- Setting up Configuration and Paths ---")
    script_dir = Path(__file__).parent.resolve()
    config = PipelineConfig() # Uses defaults defined in the dataclass
    config.device = get_compute_device()

    # Find Tracker Config dynamically (Unchanged)
    tracker_filename = f"{config.tracker_type}.yaml"
    potential_paths = [ BOXMOT_PATH / "configs" / tracker_filename, script_dir / "configs" / tracker_filename, script_dir / tracker_filename, Path.cwd() / "configs" / tracker_filename, Path.cwd() / tracker_filename ]
    found_path = next((p for p in potential_paths if p.is_file()), None)
    if not found_path:
        logger.warning(f"Tracker config '{tracker_filename}' not found. Searching for any '.yaml'...")
        potential_paths = [ p.parent for p in potential_paths ]
        found_path = next((yaml_file for dir_path in potential_paths for yaml_file in dir_path.glob('*.yaml') if yaml_file.is_file()), None)
        if found_path: logger.warning(f"Using fallback tracker config: {found_path}")
        else: raise FileNotFoundError(f"No tracker config (.yaml) found in checked locations.")
    config.tracker_config_path = found_path
    logger.info(f"Using tracker config: {config.tracker_config_path}")

    # Validate ReID Weights (Unchanged)
    if not config.reid_model_weights.is_file():
        potential_reid_paths = [ script_dir / config.reid_model_weights.name, Path.cwd() / config.reid_model_weights.name, script_dir / "weights" / config.reid_model_weights.name, Path.cwd() / "weights" / config.reid_model_weights.name, config.reid_model_weights ]
        found_reid_path = next((p for p in potential_reid_paths if p.is_file()), None)
        if not found_reid_path: raise FileNotFoundError(f"ReID weights '{config.reid_model_weights.name}' not found.")
        config.reid_model_weights = found_reid_path
    logger.info(f"Using ReID weights: {config.reid_model_weights}")

    # Validate ONNX Model Path
    if not config.onnx_model_path.is_file():
         potential_onnx_paths = [ script_dir / config.onnx_model_path.name, Path.cwd() / config.onnx_model_path.name, config.onnx_model_path ]
         found_onnx_path = next((p for p in potential_onnx_paths if p.is_file()), None)
         if not found_onnx_path: raise FileNotFoundError(f"ONNX detector model '{config.onnx_model_path.name}' not found in {potential_onnx_paths}.")
         config.onnx_model_path = found_onnx_path
    logger.info(f"Using ONNX detector model: {config.onnx_model_path}")

    # Validate Dataset Base Path (Unchanged)
    if not config.dataset_base_path.is_dir(): raise FileNotFoundError(f"Dataset base path not found: {config.dataset_base_path}")
    logger.info(f"Using dataset base path: {config.dataset_base_path}")

    logger.info("Configuration setup complete.")
    logger.info(f"Final Config: {config}")
    return config

# --- load_dataset_info, load_frames_for_batch, display_combined_frames (Unchanged) ---
def load_dataset_info(config: PipelineConfig) -> Tuple[Dict[CameraID, Path], List[str]]:
    logger.info("--- Loading Dataset Information ---")
    camera_dirs: Dict[CameraID, Path] = {}; valid_cameras: List[CameraID] = []
    base_scene_path = config.dataset_base_path / "train" / "train" / config.selected_scene
    if not base_scene_path.is_dir(): raise FileNotFoundError(f"Scene directory not found: {base_scene_path}")
    logger.info(f"Using scene path: {base_scene_path}")
    for cam_id in config.selected_cameras:
        cam_rgb_dir = base_scene_path / cam_id / "rgb"
        if cam_rgb_dir.is_dir():
            image_files = list(cam_rgb_dir.glob('*.jpg')) + list(cam_rgb_dir.glob('*.png'))
            if image_files:
                 camera_dirs[cam_id] = cam_rgb_dir; valid_cameras.append(cam_id)
                 logger.info(f"Found valid image directory with {len(image_files)} images: {cam_rgb_dir}")
            else: logger.warning(f"Image directory found for {cam_id} but contains no .jpg/.png files. Skipping: {cam_rgb_dir}")
        else: logger.warning(f"Image directory not found for {cam_id} at {cam_rgb_dir}. Skipping.")
    if not valid_cameras: raise RuntimeError("No valid camera data sources with images available.")
    config.selected_cameras = valid_cameras
    logger.info(f"Processing frames from cameras: {valid_cameras}")
    image_filenames: List[str] = []
    try:
        first_cam_dir = camera_dirs[valid_cameras[0]]
        image_filenames = sorted_alphanumeric([ f.name for f in first_cam_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"] ])
        if not image_filenames: raise ValueError(f"No image files found in the first valid camera directory: {first_cam_dir}")
        logger.info(f"Found {len(image_filenames)} frames based on camera {valid_cameras[0]}.")
    except Exception as e: raise RuntimeError(f"Failed to list image files: {e}") from e
    return camera_dirs, image_filenames

def load_frames_for_batch(camera_dirs: Dict[CameraID, Path], filename: str) -> Dict[CameraID, FrameData]:
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
    valid_annotated = [f for f in annotated_frames.values() if f is not None and f.size > 0]
    if not valid_annotated:
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8); cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated); cols = int(np.ceil(np.sqrt(num_cams))); rows = int(np.ceil(num_cams / cols))
        target_h, target_w = valid_annotated[0].shape[:2]; combined_h, combined_w = rows * target_h, cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        frame_idx = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx < num_cams:
                    frame = valid_annotated[frame_idx]
                    if frame.shape[:2] != (target_h, target_w):
                        try: frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err: frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    try: combined_display[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame
                    except ValueError as slice_err: logger.error(f"Grid placement error: {slice_err}")
                    frame_idx += 1
        disp_h, disp_w = combined_display.shape[:2]
        if disp_w > max_width:
            try: scale = max_width / disp_w; disp_h_new, disp_w_new = int(disp_h * scale), max_width; combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err: logger.error(f"Final display resize failed: {final_resize_err}")
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1: cv2.imshow(window_name, combined_display)
    else: logger.warning(f"Cannot display frames, window '{window_name}' not found or closed.")

def main():
    """Main execution function."""
    pipeline = None
    try:
        config = setup_paths_and_config()
        camera_dirs, image_filenames = load_dataset_info(config)
        pipeline = MultiCameraPipeline(config) # Initializes ONNX session, ReID, Trackers
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        logger.info("--- Starting Frame Processing Loop (using ONNX Detector) ---")
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
            if frame_idx < 10 or frame_idx % 50 == 0:
                 timing_str = " | ".join([f"{k}={v*1000:.1f}ms" for k, v in batch_result.timings.items() if v > 0.0001])
                 track_count = sum(len(tracks) for tracks in batch_result.results_per_camera.values())
                 current_loop_duration = iter_end_time - loop_start_time
                 avg_fps_so_far = frames_processed_count / current_loop_duration if current_loop_duration > 0 else 0
                 logger.info( f"Frame {frame_idx:>4} | Batch Time: {frame_proc_time_ms:>6.1f}ms | AvgFPS: {avg_fps_so_far:5.2f} | "
                              f"Pipeline: {timing_str} | ActiveTracks: {track_count}" )

            key = cv2.waitKey(config.display_wait_ms) & 0xFF
            if key == ord('q'): logger.info("Quit key pressed."); break
            elif key == ord('p'): logger.info("Paused. Press any key..."); cv2.waitKey(0); logger.info("Resuming.")
            if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1: logger.info("Display window closed."); break

        end_time = time.time(); total_time = end_time - start_time
        logger.info("--- Pipeline Finished ---")
        logger.info(f"Processed {frames_processed_count} frame batches.")
        if frames_processed_count > 0 and total_time > 0.01: logger.info(f"Total time: {total_time:.2f}s. Overall Avg FPS: {frames_processed_count / total_time:.2f}")
        else: logger.info("No frames processed or time was zero.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e: logger.critical(f"Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt: logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e: logger.critical(f"Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("Closing OpenCV windows...")
        cv2.destroyAllWindows()
        if torch.cuda.is_available(): logger.info("Clearing CUDA cache."); del pipeline; torch.cuda.empty_cache()
        logger.info("Exiting.")

if __name__ == "__main__":
    main()