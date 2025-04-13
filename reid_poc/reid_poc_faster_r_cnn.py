# -*- coding: utf-8 -*-
import os
import sys
import time
import re # Added for sorting
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image # Needed for FasterRCNN preprocessing

# --- BoxMOT Imports ---
# Ensure boxmot is installed and accessible (e.g., pip install boxmot)
try:
    from boxmot import create_tracker
    # Find the directory where boxmot is installed to locate configs
    import boxmot as boxmot_root_module
    BOXMOT_PATH = Path(boxmot_root_module.__file__).parent
    print(f"DEBUG: Found BoxMOT installation at: {BOXMOT_PATH}")
except ImportError as e:
    print(f"FATAL ERROR: Failed to import boxmot. Is it installed? Error: {e}")
    sys.exit(1)

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.trackers.basetracker import BaseTracker # Base class for tracker instances
# --- End BoxMOT Imports ---

from scipy.spatial.distance import cosine as cosine_distance

# --- Configuration ---
# Detector Configuration (Faster R-CNN)
PERSON_CLASS_ID = 1 # COCO class ID for 'person' in TorchVision models is 1
CONFIDENCE_THRESHOLD = 0.5 # Minimum detection confidence for Faster R-CNN

# Re-ID Configuration (OSNet)
# Make sure this file exists relative to your script or provide an absolute path
REID_MODEL_WEIGHTS = Path("osnet_x0_25_msmt17.pt")
REID_SIMILARITY_THRESHOLD = 0.65 # Threshold for matching embeddings
GALLERY_EMA_ALPHA = 0.9 # Exponential Moving Average alpha for updating gallery embeddings

# --- Tracker Configuration (ByteTrack) ---
TRACKER_TYPE = 'bytetrack'
# Config path will be set dynamically after finding boxmot installation
BYTETRACK_CONFIG_PATH: Optional[Path] = None
# Example: BYTETRACK_CONFIG_PATH = Path("/path/to/your/jwizzed-boxmot/boxmot/configs/bytetrack.yaml")
# TRACK_THRESH = 0.6 # Can be overridden if needed, otherwise read from yaml
# TRACK_BUFFER = 30
# MATCH_THRESH = 0.9
# --- End Tracker Configuration ---


# --- MTMMC Dataset Configuration ---
# !! MODIFY THESE PATHS !!
# Set the root path of the MTMMC dataset
DATASET_BASE_PATH = r"D:\MTMMC" # Example for Windows (Use raw string or double backslashes)
# DATASET_BASE_PATH = "/Volumes/HDD/MTMMC" # Example for macOS/Linux
SELECTED_SCENE = "s10"
SELECTED_CAMERAS = ["c09", "c12", "c13", "c16"] # e.g., ["c01", "c02", "c09", "c16"]

# --- General Configuration ---
# Ensure device selection happens early
SELECTED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Default device selection
# SELECTED_DEVICE = torch.device("cpu") # Force CPU
# SELECTED_DEVICE = torch.device("mps") # Force MPS if available


# --- Visualization ---
DRAW_BOUNDING_BOXES = True
SHOW_TRACK_ID = True # Show the per-camera tracker ID
SHOW_GLOBAL_ID = True # Show the cross-camera global ID
WINDOW_NAME = "Multi-Camera Tracking & Re-ID POC (MTMMC - FasterRCNN + ByteTrack + OSNet)"
DISPLAY_WAIT_MS = 1 # Wait time for cv2.waitKey (1 for fast playback, 0 for pause on each frame)
MAX_DISPLAY_WIDTH = 1920 # Max width for the combined display window (optional resizing)

# --- Helper Functions ---

def sorted_alphanumeric(data):
    """Sorts a list of strings alphanumerically (handling numbers correctly)."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_and_set_device() -> torch.device:
    """Selects the best available compute device and sets the global SELECTED_DEVICE."""
    global SELECTED_DEVICE # Declare intent to modify the global variable
    print("--- Determining Device ---")
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            print(f"Attempting to use CUDA device: {torch.cuda.get_device_name(device)}")
            _ = torch.tensor([1.0], device="cuda") + torch.tensor([1.0], device="cuda") # Quick test
            print("CUDA device confirmed.")
            SELECTED_DEVICE = device
            return device
        except Exception as e:
             print(f"CUDA reported available, but test failed ({e}). Falling back...")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
         try:
             device = torch.device("mps")
             print("Attempting to use Metal Performance Shaders (MPS) device.")
             _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps") # Quick test
             print("MPS device confirmed.")
             SELECTED_DEVICE = device
             return device
         except Exception as e:
             print(f"MPS reported available, but test failed ({e}). Falling back...")
    else:
        print("CUDA / MPS not available.")

    device = torch.device("cpu")
    print("Using CPU device.")
    SELECTED_DEVICE = device
    return device


def calculate_cosine_similarity(feat1: Optional[np.ndarray], feat2: Optional[np.ndarray]) -> float:
    """Calculates cosine similarity between two feature vectors. Handles None inputs."""
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten(); feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if np.all(feat1 == 0) or np.all(feat2 == 0): return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0
    try:
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        distance = max(0.0, float(distance)); distance = min(2.0, distance)
    except Exception as e: print(f"Error calculating cosine distance: {e}"); return 0.0
    similarity = 1.0 - distance
    return float(np.clip(similarity, 0.0, 1.0))


# --- Main Processing Class ---

class MultiCameraPipeline:
    def __init__(self,
                 reid_weights_path: Path,
                 tracker_config_path: Path,
                 camera_ids: List[str],
                 device: torch.device): # Takes device object
        self.device = device
        self.camera_ids = camera_ids
        print(f"\n--- Loading Models on Device: {self.device} ---")

        # 1. Load Shared Faster R-CNN Detector
        self.detector = None
        self.detector_transforms = None
        try:
            print("Loading SHARED Faster R-CNN (ResNet50 FPN) detector...")
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.detector.to(self.device)
            self.detector.eval()
            self.detector_transforms = weights.transforms()
            print("Faster R-CNN Detector loaded successfully.")
        except Exception as e: print(f"FATAL ERROR loading Faster R-CNN detector: {e}"); sys.exit(1)

        # 2. Load Shared OSNet ReID Model
        self.reid_model: Optional[BaseModelBackend] = None
        try:
            print(f"Loading SHARED OSNet ReID model from: {reid_weights_path}")
            # Determine the device string specifier required by ReidAutoBackend
            # based on the torch.device object
            reid_device_specifier = self._get_reid_device_specifier_string(self.device)
            print(f"Attempting to load ReID model onto device specifier: '{reid_device_specifier}'")

            # Pass the *string specifier* to ReidAutoBackend
            # (based on typical BoxMOT usage)
            reid_model_handler = ReidAutoBackend(
                weights=reid_weights_path,
                device=reid_device_specifier, # Pass the string specifier ('0', 'cpu', 'mps')
                half=False
            )
            self.reid_model = reid_model_handler.model # Get the actual backend model

            # Optional warmup call
            if hasattr(self.reid_model, "warmup"):
                print("Warming up OSNet ReID model...")
                self.reid_model.warmup()
            print("OSNet ReID Model loaded successfully.")

        except ImportError as e: print(f"FATAL ERROR loading ReID model: BoxMOT/ReID dependencies might be missing. {e}"); sys.exit(1)
        except FileNotFoundError as e: print(f"FATAL ERROR loading ReID model: Weights file not found at {reid_weights_path}. {e}"); sys.exit(1)
        except Exception as e: print(f"FATAL ERROR loading ReID model with device specifier '{reid_device_specifier}': {e}"); sys.exit(1)


        # 3. Initialize Trackers (One per Camera)
        self.trackers: Dict[str, BaseTracker] = {}
        print(f"\n--- Initializing {TRACKER_TYPE} Trackers (One per Camera) ---")
        if not tracker_config_path or not tracker_config_path.is_file():
             print(f"FATAL ERROR: Tracker config file not found at {tracker_config_path}"); sys.exit(1)
        try:
            # Convert torch.device to string for create_tracker if needed by underlying tracker init
            tracker_device_str = str(self.device) # e.g., 'cuda:0', 'cpu', 'mps'
            for cam_id in self.camera_ids:
                # Pass per_class=False as we filter detections *before* passing to tracker
                # ByteTrack itself doesn't seem to use device/half args directly in create_tracker
                # but they are passed for potential future use or other trackers.
                tracker_instance = create_tracker(
                    tracker_type=TRACKER_TYPE,
                    tracker_config=tracker_config_path,
                    reid_weights=None, # Not used by ByteTrack directly here
                    device=tracker_device_str, # Pass device string representation
                    half=False,
                    per_class=False
                )
                self.trackers[cam_id] = tracker_instance
                # Reset tracker count for each instance if necessary (BoxMOT might handle this)
                if hasattr(tracker_instance, 'reset'): tracker_instance.reset()
                print(f"Initialized {TRACKER_TYPE} for camera {cam_id}")
            print(f"Successfully initialized {len(self.trackers)} tracker instances.")
        except Exception as e: print(f"FATAL ERROR initializing trackers: {e}"); sys.exit(1)


        # State Management
        self.reid_gallery: Dict[int, np.ndarray] = {} # {global_id: embedding}
        self.track_to_global_id: Dict[Tuple[str, int], int] = {} # {(camera_id, track_id): global_id}
        self.next_global_id = 1

    def _get_reid_device_specifier_string(self, device: torch.device) -> str:
        """Determines the device string specifier needed by ReidAutoBackend."""
        if device.type == 'cuda':
            cuda_index = device.index if device.index is not None else 0
            specifier = str(cuda_index)
            print(f"Determined ReID device specifier: CUDA index '{specifier}'")
        elif device.type == 'mps':
            specifier = 'mps'
            print(f"Determined ReID device specifier: MPS '{specifier}'")
        else: # CPU
            specifier = 'cpu'
            print(f"Determined ReID device specifier: CPU '{specifier}'")
        return specifier

    def _detect_persons(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detects persons in a single frame using Faster R-CNN."""
        detections: List[Dict[str, Any]] = []
        if self.detector is None or self.detector_transforms is None or frame_bgr is None or frame_bgr.size == 0:
            return detections

        try:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            input_tensor = self.detector_transforms(img_pil)
            input_batch = [input_tensor.to(self.device)]

            with torch.no_grad():
                predictions = self.detector(input_batch)

            pred_boxes_xyxy = predictions[0]['boxes'].cpu().numpy()
            pred_labels = predictions[0]['labels'].cpu().numpy()
            pred_scores = predictions[0]['scores'].cpu().numpy()

            for box_xyxy, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                if label == PERSON_CLASS_ID and score >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box_xyxy
                    if x2 > x1 and y2 > y1:
                         detections.append({
                             'bbox_xyxy': box_xyxy.astype(np.float32),
                             'conf': float(score),
                             'class_id': PERSON_CLASS_ID
                         })

        except Exception as e:
            print(f"E: Detection error: {e}")
            # import traceback; traceback.print_exc()

        return detections

    def _extract_features_for_tracks(self,
                                     frame_bgr: np.ndarray,
                                     tracked_dets_np: np.ndarray
                                     ) -> Dict[int, np.ndarray]:
        """
        Extracts Re-ID features for tracked persons in a single frame.
        Returns a dictionary mapping track_id (int) to feature embedding (np.ndarray).
        """
        features: Dict[int, np.ndarray] = {} # {track_id: embedding}
        if self.reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.size == 0:
            return features

        # Assuming tracked_dets_np format: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        bboxes_xyxy_list = tracked_dets_np[:, 0:4]
        # Extract track IDs and ensure they are integers
        track_ids = tracked_dets_np[:, 4]

        if not bboxes_xyxy_list.size: return features

        bboxes_np = np.array(bboxes_xyxy_list).astype(np.float32)
        if bboxes_np.ndim != 2 or bboxes_np.shape[1] != 4:
             print(f"W: Invalid bbox shape for feature extraction. Shape: {bboxes_np.shape}")
             return features

        try:
            batch_features = self.reid_model.get_features(bboxes_np, frame_bgr)

            if batch_features is not None and len(batch_features) == len(track_ids):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all():
                        # --- FIX: Cast track_id to int ---
                        current_track_id = int(track_ids[i]) # Explicitly cast to Python int
                        # ---------------------------------
                        features[current_track_id] = det_feature
                    # else:
                        # current_track_id_val = track_ids[i] # Get value before potentially failing int cast
                        # print(f"W: Invalid feature extracted for track {current_track_id_val}. Skipping.")
            # else:
                 # print(f"W: Feature extraction mismatch/failure. Expected {len(track_ids)}, got {len(batch_features) if batch_features is not None else 'None'}.")

        except Exception as e:
            print(f"E: Feature extraction call failed: {e}")
            # import traceback; traceback.print_exc()

        return features


    def _perform_reid_on_tracks(self,
                                features_per_track: Dict[Tuple[str, int], np.ndarray]
                               ) -> Dict[Tuple[str, int], Optional[int]]:
        """
        Compares features of currently tracked objects against the global gallery
        and assigns global IDs. Updates the global gallery.

        Args:
            features_per_track: Dict mapping (camera_id, track_id) to the latest feature embedding.

        Returns:
            Dict mapping (camera_id, track_id) to the assigned global_id (or None).
        """
        assigned_global_ids: Dict[Tuple[str, int], Optional[int]] = {}
        if not features_per_track:
            return assigned_global_ids

        valid_gallery_ids = [gid for gid, emb in self.reid_gallery.items() if emb is not None and np.isfinite(emb).all()]
        valid_gallery_embeddings = [self.reid_gallery[gid] for gid in valid_gallery_ids]

        match_counts = 0
        new_id_counts = 0
        update_counts = 0

        for track_key, new_embedding in features_per_track.items():
            camera_id, track_id = track_key
            assigned_global_ids[track_key] = None # Default

            if new_embedding is None or not np.isfinite(new_embedding).all():
                if track_key in self.track_to_global_id:
                     assigned_global_ids[track_key] = self.track_to_global_id[track_key]
                continue

            best_match_global_id = None
            best_match_score = 0.0

            if valid_gallery_ids:
                similarities = [calculate_cosine_similarity(new_embedding, gal_emb) for gal_emb in valid_gallery_embeddings]
                if similarities:
                    max_similarity = max(similarities)
                    if max_similarity >= REID_SIMILARITY_THRESHOLD:
                        best_match_index = np.argmax(similarities)
                        best_match_global_id = valid_gallery_ids[best_match_index]
                        best_match_score = max_similarity

            if best_match_global_id is not None:
                assigned_global_id = best_match_global_id
                assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id
                match_counts += 1

                current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                if current_gallery_emb is not None:
                    updated_embedding = (GALLERY_EMA_ALPHA * current_gallery_emb +
                                       (1 - GALLERY_EMA_ALPHA) * new_embedding)
                    norm = np.linalg.norm(updated_embedding)
                    self.reid_gallery[assigned_global_id] = updated_embedding / norm if norm > 1e-6 else updated_embedding
                    update_counts +=1

            else:
                last_known_global_id = self.track_to_global_id.get(track_key, None)
                if last_known_global_id is not None and last_known_global_id in self.reid_gallery:
                    assigned_global_id = last_known_global_id
                    assigned_global_ids[track_key] = assigned_global_id

                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (GALLERY_EMA_ALPHA * current_gallery_emb +
                                           (1 - GALLERY_EMA_ALPHA) * new_embedding)
                        norm = np.linalg.norm(updated_embedding)
                        self.reid_gallery[assigned_global_id] = updated_embedding / norm if norm > 1e-6 else updated_embedding
                        update_counts +=1
                else:
                    new_global_id = self.next_global_id
                    self.next_global_id += 1
                    assigned_global_ids[track_key] = new_global_id
                    self.track_to_global_id[track_key] = new_global_id

                    norm = np.linalg.norm(new_embedding)
                    self.reid_gallery[new_global_id] = new_embedding / norm if norm > 1e-6 else new_embedding
                    new_id_counts += 1

        # print(f"  Re-ID: Matched={match_counts}, New IDs={new_id_counts}, Updated Gall={update_counts}")

        current_track_keys = set(features_per_track.keys())
        keys_to_remove = set(self.track_to_global_id.keys()) - current_track_keys
        for key in keys_to_remove:
            del self.track_to_global_id[key]
            # print(f"  Cleaned up mapping for lost track: {key}")

        return assigned_global_ids


    def process_frame_batch(self, frames: Dict[str, Optional[np.ndarray]], frame_idx: int
                            ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
        """
        Main processing function: Detect -> Track -> Extract Features -> Re-ID.
        Returns processed tracks with global IDs and timings.
        """
        t_start_batch = time.time()
        timings = {'detection_tracking': 0.0, 'feature_ext': 0.0, 'reid': 0.0, 'total': 0.0}
        final_results_per_camera: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        all_current_track_features: Dict[Tuple[str, int], np.ndarray] = {} # (cam_id, track_id) -> feature

        # --- 1. Detection & Tracking (Per Camera) ---
        t_det_track_start = time.time()
        tracked_outputs_per_camera: Dict[str, np.ndarray] = {}

        for cam_id, frame_bgr in frames.items():
            tracker = self.trackers.get(cam_id)
            if not tracker:
                 print(f"E: No tracker found for camera {cam_id}")
                 continue

            if frame_bgr is None or frame_bgr.size == 0:
                try:
                    # Provide a dummy frame if None, ensures consistent input type for tracker
                    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8) if frame_bgr is None else frame_bgr
                    empty_dets = np.empty((0, 6)) # x1,y1,x2,y2,conf,cls
                    _ = tracker.update(empty_dets, dummy_frame) # Update tracker state
                except Exception as e:
                    print(f"E: Updating tracker state for cam {cam_id} with empty dets failed: {e}")
                continue # Skip actual processing for this camera

            # Detect persons
            detections = self._detect_persons(frame_bgr)

            # Prepare detections for tracker
            if detections:
                np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in detections])
            else:
                np_dets = np.empty((0, 6))

            # Update tracker
            try:
                tracked_dets_np = tracker.update(np_dets, frame_bgr)
                # Ensure tracked_dets_np is always a numpy array, even if empty
                tracked_outputs_per_camera[cam_id] = np.array(tracked_dets_np) if tracked_dets_np is not None and len(tracked_dets_np) > 0 else np.empty((0, 8))
            except Exception as e:
                print(f"E: Tracker update failed for camera {cam_id}: {e}")
                tracked_outputs_per_camera[cam_id] = np.empty((0, 8))

        timings['detection_tracking'] = time.time() - t_det_track_start

        # --- 2. Extract Features for Tracked Objects ---
        t_feat_start = time.time()
        for cam_id, tracked_dets_np in tracked_outputs_per_camera.items():
             # Check if array has data (more robust than just size > 0)
             if tracked_dets_np.shape[0] > 0:
                frame_bgr = frames.get(cam_id) # Get original frame again
                if frame_bgr is not None and frame_bgr.size > 0:
                    features_this_cam = self._extract_features_for_tracks(frame_bgr, tracked_dets_np)
                    for track_id, feature in features_this_cam.items():
                        all_current_track_features[(cam_id, track_id)] = feature
        timings['feature_ext'] = time.time() - t_feat_start

        # --- 3. Perform Re-ID on Tracked Features ---
        t_reid_start = time.time()
        assigned_global_ids = self._perform_reid_on_tracks(all_current_track_features)
        timings['reid'] = time.time() - t_reid_start

        # --- 4. Combine Tracking and Re-ID Results ---
        for cam_id, tracked_dets_np in tracked_outputs_per_camera.items():
            if tracked_dets_np.shape[0] > 0:
                 for track_data in tracked_dets_np:
                    # Ensure slicing doesn't go out of bounds if format changes unexpectedly
                    if len(track_data) >= 7:
                        x1, y1, x2, y2, track_id_float, conf, cls = track_data[0:7]
                        track_id = int(track_id_float) # Cast track ID
                        cls = int(cls)                  # Cast class ID

                        global_id = assigned_global_ids.get((cam_id, track_id), None)

                        final_results_per_camera[cam_id].append({
                            'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                            'track_id': track_id,
                            'global_id': global_id,
                            'conf': float(conf),
                            'class_id': cls
                        })
                    else:
                        print(f"W: Unexpected track data format for cam {cam_id}: {track_data}")


        # --- 5. Final Timing ---
        timings['total'] = time.time() - t_start_batch

        return dict(final_results_per_camera), timings


    def draw_annotations(self,
                         frames: Dict[str, Optional[np.ndarray]],
                         processed_results: Dict[str, List[Dict[str, Any]]]
                         ) -> Dict[str, Optional[np.ndarray]]:
        """Draws bounding boxes with track_id and global_id on frames."""
        annotated_frames: Dict[str, Optional[np.ndarray]] = {}
        default_frame_h, default_frame_w = 1080, 1920
        first_valid_frame_dims_set = False

        for frame in frames.values():
            if frame is not None and frame.size > 0:
                 default_frame_h, default_frame_w = frame.shape[:2]
                 first_valid_frame_dims_set = True
                 break

        for cam_id, frame in frames.items():
             if frame is None or frame.size == 0:
                  placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                  cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                  annotated_frames[cam_id] = placeholder
                  continue

             current_h, current_w = frame.shape[:2]
             annotated_frame = frame.copy()
             results_for_cam = processed_results.get(cam_id, [])

             for track_info in results_for_cam:
                  bbox_xyxy = track_info.get('bbox_xyxy')
                  track_id = track_info.get('track_id')
                  global_id = track_info.get('global_id')
                  conf = track_info.get('conf', 0.0)

                  if bbox_xyxy is None: continue

                  x1, y1, x2, y2 = map(int, bbox_xyxy)
                  x1 = max(0, x1); y1 = max(0, y1); x2 = min(current_w, x2); y2 = min(current_h, y2)
                  if x1 >= x2 or y1 >= y2: continue

                  color = (255, 182, 193) # Default Pink
                  if global_id is not None:
                       # Consistent color based on global_id
                       seed = int(global_id) * 3 + 5 # Ensure global_id is treated as int for seeding
                       color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)

                  if DRAW_BOUNDING_BOXES:
                       cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                  label_parts = []
                  if SHOW_TRACK_ID and track_id is not None: label_parts.append(f"T:{track_id}")
                  if SHOW_GLOBAL_ID: label_parts.append(f"G:{global_id}" if global_id is not None else "G:?")
                  # label_parts.append(f"C:{conf:.2f}") # Optional: add confidence
                  label = " ".join(label_parts)

                  if label:
                       (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                       ly = y1 - 10 if y1 > (lh + 10) else y1 + lh + 5
                       ly = max(lh + 5, ly) # Ensure not cut off at top
                       lx = x1

                       # Draw background rectangle and text
                       cv2.rectangle(annotated_frame, (lx, ly - lh - bl), (lx + lw, ly), color, cv2.FILLED)
                       cv2.putText(annotated_frame, label, (lx, ly - bl//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

             annotated_frames[cam_id] = annotated_frame

        return annotated_frames


# --- Main Execution ---
if __name__ == "__main__":
    # Determine device first
    selected_device = get_and_set_device() # Sets the global SELECTED_DEVICE

    # --- Find BoxMOT Path and Config ---
    # BOXMOT_PATH is defined globally after imports
    BYTETRACK_CONFIG_PATH = BOXMOT_PATH / "configs" / f"{TRACKER_TYPE}.yaml"
    if not BYTETRACK_CONFIG_PATH.is_file():
         print(f"W: Could not find {TRACKER_TYPE}.yaml at {BYTETRACK_CONFIG_PATH}")
         script_dir = Path(__file__).parent.resolve() # Use absolute path of script dir
         # Fallback 1: ./boxmot/configs/
         fallback_config_path = script_dir / "boxmot" / "configs" / f"{TRACKER_TYPE}.yaml"
         # Fallback 2: ./configs/
         fallback_config_path_2 = script_dir / "configs" / f"{TRACKER_TYPE}.yaml"
         # Fallback 3: ./*.yaml (directly in script dir)
         fallback_config_path_3 = script_dir / f"{TRACKER_TYPE}.yaml"

         if fallback_config_path.is_file():
              BYTETRACK_CONFIG_PATH = fallback_config_path
              print(f"Found config at fallback location 1: {BYTETRACK_CONFIG_PATH}")
         elif fallback_config_path_2.is_file():
             BYTETRACK_CONFIG_PATH = fallback_config_path_2
             print(f"Found config at fallback location 2: {BYTETRACK_CONFIG_PATH}")
         elif fallback_config_path_3.is_file():
             BYTETRACK_CONFIG_PATH = fallback_config_path_3
             print(f"Found config at fallback location 3: {BYTETRACK_CONFIG_PATH}")
         else:
              print(f"FATAL ERROR: Could not find {TRACKER_TYPE}.yaml config file in standard location or fallbacks.")
              sys.exit(1)

    # Validate ReID weights path
    if not REID_MODEL_WEIGHTS.is_file():
        print(f"W: ReID weights file not found at the configured path: {REID_MODEL_WEIGHTS}")
        script_dir = Path(__file__).parent.resolve()
        fallback_reid_path = script_dir / REID_MODEL_WEIGHTS.name
        if fallback_reid_path.is_file():
            REID_MODEL_WEIGHTS = fallback_reid_path
            print(f"Found ReID weights at fallback location: {REID_MODEL_WEIGHTS}")
        else:
            print(f"FATAL ERROR: ReID weights file not found at {REID_MODEL_WEIGHTS} or {fallback_reid_path}.")
            sys.exit(1)


    # --- Construct Camera Paths and Validate ---
    camera_base_dirs = {}
    valid_cameras = []
    base_scene_path = Path(DATASET_BASE_PATH) / "train" / "train" / SELECTED_SCENE
    if not base_scene_path.is_dir(): print(f"FATAL ERROR: Scene directory not found: {base_scene_path}"); sys.exit(1)

    print("\n--- Validating Camera Directories ---")
    for cam_id in SELECTED_CAMERAS:
        rgb_dir = base_scene_path / cam_id / "rgb"
        if rgb_dir.is_dir():
            camera_base_dirs[cam_id] = str(rgb_dir)
            valid_cameras.append(cam_id)
            print(f"Found valid camera directory: {rgb_dir}")
        else: print(f"W: RGB directory not found for {cam_id} in scene {SELECTED_SCENE}. Skipping.")

    if not valid_cameras: print(f"FATAL ERROR: No valid cameras found for {SELECTED_CAMERAS} in {SELECTED_SCENE}."); sys.exit(1)
    print(f"Processing cameras: {valid_cameras}")

    # --- Determine Frame Sequence and Count ---
    image_filenames = []
    max_proc_frames = 0
    try:
        first_cam_id = valid_cameras[0]
        first_cam_dir = camera_base_dirs[first_cam_id]
        image_filenames = sorted_alphanumeric([f for f in os.listdir(first_cam_dir) if f.lower().endswith(".jpg")])
        if not image_filenames: raise ValueError(f"No JPG files found in {first_cam_dir}")
        max_proc_frames = len(image_filenames)
        print(f"\nFound {max_proc_frames} frames based on {first_cam_id}.")
    except Exception as e: print(f"FATAL ERROR: Could not list image files: {e}"); sys.exit(1)

    # --- Initialize the Pipeline ---
    # Pass the globally set selected_device
    pipeline = MultiCameraPipeline(
        reid_weights_path=REID_MODEL_WEIGHTS,
        tracker_config_path=BYTETRACK_CONFIG_PATH,
        camera_ids=valid_cameras,
        device=selected_device
    )

    # --- Initialize Display Window ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --- Processing Loop ---
    start_time = time.time(); actual_frames_processed = 0
    print("\n--- Starting Frame Processing Loop ---")

    for frame_idx, current_filename in enumerate(image_filenames):
        t_frame_start = time.time() # Start timing for this frame index

        # --- Load Frames for the Current Index ---
        current_frames_bgr: Dict[str, Optional[np.ndarray]] = {}
        valid_frame_loaded_this_iter = False
        for cam_id in valid_cameras:
            image_path = os.path.join(camera_base_dirs[cam_id], current_filename)
            img = None
            if os.path.exists(image_path):
                try: img = cv2.imread(image_path)
                except Exception as e: print(f"E: reading {image_path}: {e}")
                if img is not None and img.size > 0:
                    current_frames_bgr[cam_id] = img
                    valid_frame_loaded_this_iter = True
                else: current_frames_bgr[cam_id] = None
            else: current_frames_bgr[cam_id] = None

        if not valid_frame_loaded_this_iter:
             print(f"W: No valid frames loaded for index {frame_idx} ({current_filename}). Processing empty.")
             for cam_id in valid_cameras:
                 if cam_id not in current_frames_bgr:
                     current_frames_bgr[cam_id] = None

        actual_frames_processed += 1

        # --- Process the Batch ---
        processed_results, timings = pipeline.process_frame_batch(current_frames_bgr, frame_idx)

        # --- Print Timings ---
        t_frame_end = time.time()
        frame_proc_time = t_frame_end - t_frame_start
        if frame_idx % 10 == 0 or frame_idx < 5:
            print(f"\n--- Frame {frame_idx} ({current_filename}) --- Frame Proc Time: {frame_proc_time*1000:.1f} ms ---")
            print(f"  Pipeline Timings (ms): Total={timings['total']*1000:.1f} | "
                  f"Detect+Track={timings['detection_tracking']*1000:.1f} | "
                  f"FeatExtract={timings['feature_ext']*1000:.1f} | "
                  f"ReID={timings['reid']*1000:.1f}")
            track_count = sum(len(tracks) for tracks in processed_results.values())
            print(f"  Active Tracks: {track_count} persons across {len(processed_results)} cameras.")


        # --- Draw Annotations ---
        annotated_frames = pipeline.draw_annotations(current_frames_bgr, processed_results)

        # --- Visualization ---
        valid_annotated = [f for cam_id in valid_cameras for f in [annotated_frames.get(cam_id)] if f is not None and f.size > 0]
        combined_display = None
        if valid_annotated:
            num_cams = len(valid_annotated)
            rows = int(np.ceil(np.sqrt(num_cams))); cols = int(np.ceil(num_cams / rows))
            try:
                target_h, target_w = valid_annotated[0].shape[:2]
            except IndexError:
                 target_h, target_w = 480, 640

            combined_h, combined_w = rows * target_h, cols * target_w
            combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

            frame_num_in_batch = 0
            for r in range(rows):
                for c in range(cols):
                    if frame_num_in_batch < num_cams:
                        frame_to_place = valid_annotated[frame_num_in_batch]
                        if frame_to_place.shape[0]!= target_h or frame_to_place.shape[1]!= target_w:
                             try: frame_to_place = cv2.resize(frame_to_place, (target_w, target_h), interpolation=cv2.INTER_AREA)
                             except Exception: frame_to_place = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                        combined_display[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame_to_place
                        frame_num_in_batch += 1

            display_h, display_w = combined_display.shape[:2]
            if display_w > MAX_DISPLAY_WIDTH:
                scale = MAX_DISPLAY_WIDTH / display_w
                display_h = int(display_h * scale)
                display_w = MAX_DISPLAY_WIDTH
                combined_display = cv2.resize(combined_display, (display_w, display_h), interpolation=cv2.INTER_AREA)

        else:
             combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
             cv2.putText(combined_display, "No Frames to Display", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, combined_display)

        key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('p'):
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)


    # --- Cleanup ---
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- Pipeline Finished ---")
    print(f"Processed {actual_frames_processed} frame indices.")
    print(f"Total execution time: {total_time:.2f} seconds.")
    if actual_frames_processed > 0 and total_time > 0:
        print(f"Average FPS: {actual_frames_processed / total_time:.2f}")
    else:
         print("Average FPS: N/A")

    cv2.destroyAllWindows()
    print("Resources released.")
