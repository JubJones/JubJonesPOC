# -*- coding: utf-8 -*-
import os
import sys
import time
import re # Added for sorting
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union, Set # Added Set

import cv2
import numpy as np
import torch
# Removed torchvision detector import

# --- Ultralytics Import ---
try:
    from ultralytics import RTDETR
except ImportError as e:
    print(f"FATAL ERROR: Failed to import ultralytics. Is it installed (pip install ultralytics)? Error: {e}")
    sys.exit(1)
# --- End Ultralytics Import ---

# --- BoxMOT Imports ---
try:
    from boxmot import create_tracker
    import boxmot as boxmot_root_module
    BOXMOT_PATH = Path(boxmot_root_module.__file__).parent
    print(f"DEBUG: Found BoxMOT installation at: {BOXMOT_PATH}")
except ImportError as e:
    print(f"FATAL ERROR: Failed to import boxmot. Is it installed? Error: {e}")
    sys.exit(1)

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.trackers.basetracker import BaseTracker
# --- End BoxMOT Imports ---

from scipy.spatial.distance import cosine as cosine_distance

# --- Configuration ---
# Detector Configuration (RT-DETR)
DETECTOR_MODEL_PATH = "rtdetr-l.pt" # Path to your RT-DETR model weights
PERSON_CLASS_ID = 0 # COCO class ID for 'person' in Ultralytics models is typically 0
CONFIDENCE_THRESHOLD = 0.5 # Minimum detection confidence

# Re-ID Configuration (OSNet)
REID_MODEL_WEIGHTS = Path("osnet_x0_25_msmt17.pt")
REID_SIMILARITY_THRESHOLD = 0.65
GALLERY_EMA_ALPHA = 0.9
REID_REFRESH_INTERVAL = 10 # How often (in frames) to re-run ReID for existing tracks

# --- Tracker Configuration (ByteTrack) ---
TRACKER_TYPE = 'bytetrack'
BYTETRACK_CONFIG_PATH: Optional[Path] = None # Set dynamically

# --- MTMMC Dataset Configuration ---
DATASET_BASE_PATH = r"D:\MTMMC"
# DATASET_BASE_PATH = "/Volumes/HDD/MTMMC"
SELECTED_SCENE = "s10"
SELECTED_CAMERAS = ["c09", "c12", "c13", "c16"]

# --- General Configuration ---
SELECTED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Visualization ---
DRAW_BOUNDING_BOXES = True
SHOW_TRACK_ID = True
SHOW_GLOBAL_ID = True
WINDOW_NAME = "Multi-Camera Tracking & Re-ID (RTDETR + ByteTrack + OSNet)" # Updated Window Name
DISPLAY_WAIT_MS = 1
MAX_DISPLAY_WIDTH = 1920

# --- Helper Functions ---

def sorted_alphanumeric(data):
    """Sorts a list of strings alphanumerically (handling numbers correctly)."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_and_set_device() -> torch.device:
    """Selects the best available compute device and sets the global SELECTED_DEVICE."""
    global SELECTED_DEVICE
    print("--- Determining Device ---")
    # Use MPS if available (Apple Silicon), otherwise CUDA, fallback to CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
         try:
             device = torch.device("mps")
             print("Attempting to use Metal Performance Shaders (MPS) device.")
             _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps") # Quick test
             print("MPS device confirmed.")
             SELECTED_DEVICE = device
             return device
         except Exception as e: print(f"MPS reported available, but test failed ({e}). Falling back...")
    elif torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            print(f"Attempting to use CUDA device: {torch.cuda.get_device_name(device)}")
            _ = torch.tensor([1.0], device="cuda") + torch.tensor([1.0], device="cuda")
            print("CUDA device confirmed.")
            SELECTED_DEVICE = device
            return device
        except Exception as e: print(f"CUDA reported available, but test failed ({e}). Falling back...")

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
        distance = np.clip(float(distance), 0.0, 2.0) # Clip distance
    except Exception as e: print(f"Error calculating cosine distance: {e}"); return 0.0
    similarity = 1.0 - distance
    return float(np.clip(similarity, 0.0, 1.0)) # Clip similarity


# --- Main Processing Class ---

class MultiCameraPipeline:
    def __init__(self,
                 detector_weights_path: str, # Path for RTDETR model
                 reid_weights_path: Path,
                 tracker_config_path: Path,
                 camera_ids: List[str],
                 device: torch.device):
        self.device = device
        self.camera_ids = camera_ids
        print(f"\n--- Loading Models on Device: {self.device} ---")

        # 1. Load Detector (RT-DETR)
        self.detector = None
        try:
            print(f"Loading RTDETR detector from: {detector_weights_path}")
            self.detector = RTDETR(detector_weights_path)
            self.detector.to(self.device)
            # Optional: Warmup RT-DETR
            # print("Warming up RTDETR detector...")
            # self.detector.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=self.device)
            print("RTDETR Detector loaded successfully.")
        except Exception as e: print(f"FATAL ERROR loading RTDETR detector: {e}"); sys.exit(1)

        # 2. Load ReID Model
        self.reid_model: Optional[BaseModelBackend] = None
        try:
            print(f"Loading SHARED OSNet ReID model from: {reid_weights_path}")
            reid_device_specifier = self._get_reid_device_specifier_string(self.device)
            print(f"Attempting ReID model load onto device specifier: '{reid_device_specifier}'")
            reid_model_handler = ReidAutoBackend(
                weights=reid_weights_path, device=reid_device_specifier, half=False
            )
            self.reid_model = reid_model_handler.model
            if hasattr(self.reid_model, "warmup"):
                print("Warming up OSNet ReID model...")
                self.reid_model.warmup()
            print("OSNet ReID Model loaded.")
        except Exception as e: print(f"FATAL ERROR loading ReID model: {e}"); sys.exit(1)

        # 3. Initialize Trackers
        self.trackers: Dict[str, BaseTracker] = {}
        print(f"\n--- Initializing {TRACKER_TYPE} Trackers ---")
        if not tracker_config_path or not tracker_config_path.is_file():
             print(f"FATAL ERROR: Tracker config not found: {tracker_config_path}"); sys.exit(1)
        try:
            tracker_device_str = str(self.device)
            for cam_id in self.camera_ids:
                tracker_instance = create_tracker(
                    tracker_type=TRACKER_TYPE, tracker_config=tracker_config_path,
                    reid_weights=None, device=tracker_device_str, half=False, per_class=False
                )
                self.trackers[cam_id] = tracker_instance
                if hasattr(tracker_instance, 'reset'): tracker_instance.reset()
                print(f"Initialized {TRACKER_TYPE} for camera {cam_id}")
            print(f"Initialized {len(self.trackers)} tracker instances.")
        except Exception as e: print(f"FATAL ERROR initializing trackers: {e}"); sys.exit(1)

        # State Management (Same as before)
        self.reid_gallery: Dict[int, np.ndarray] = {}
        self.track_to_global_id: Dict[Tuple[str, int], int] = {}
        self.next_global_id = 1
        self.last_seen_track_ids: Dict[str, Set[int]] = defaultdict(set)
        self.track_last_reid_frame: Dict[Tuple[str, int], int] = {}


    def _get_reid_device_specifier_string(self, device: torch.device) -> str:
        """Determines the device string specifier needed by ReidAutoBackend."""
        if device.type == 'cuda':
            idx = device.index if device.index is not None else 0
            return str(idx)
        elif device.type == 'mps':
            return 'mps'
        else: return 'cpu'

    def _detect_persons(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detects persons in a single frame using RT-DETR."""
        detections: List[Dict[str, Any]] = []
        if self.detector is None or frame_bgr is None or frame_bgr.size == 0:
            return detections
        try:
            # RTDETR predict call - uses BGR numpy array directly
            results = self.detector.predict(
                frame_bgr,
                classes=[PERSON_CLASS_ID],      # Filter for persons during detection
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,                  # Suppress ultralytics console output
                device=self.device              # Specify device for inference
            )

            # Parse RTDETR results object
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                # cls_ids = results[0].boxes.cls.cpu().numpy() # Class IDs if needed

                for box, score in zip(boxes_xyxy, confs):
                    # Class filtering already done by predict()
                    x1, y1, x2, y2 = box
                    if x2 > x1 and y2 > y1: # Basic box validity check
                         detections.append({
                             'bbox_xyxy': box.astype(np.float32),
                             'conf': float(score),
                             'class_id': PERSON_CLASS_ID # Store the class ID we requested
                         })
        except Exception as e:
            print(f"E: RTDETR Detection error: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed errors
        return detections

    def _extract_features_for_tracks(self,
                                     frame_bgr: np.ndarray,
                                     tracked_dets_np: np.ndarray
                                     ) -> Dict[int, np.ndarray]:
        """Extracts Re-ID features for the *provided subset* of tracked persons."""
        # (Identical to previous version)
        features: Dict[int, np.ndarray] = {}
        if self.reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.shape[0] == 0:
            return features
        bboxes_xyxy_list = tracked_dets_np[:, 0:4]
        track_ids = tracked_dets_np[:, 4]
        bboxes_np = np.array(bboxes_xyxy_list).astype(np.float32)
        if bboxes_np.ndim != 2 or bboxes_np.shape[1] != 4:
             print(f"W: Invalid bbox shape for FE. Shape: {bboxes_np.shape}")
             return features
        try:
            batch_features = self.reid_model.get_features(bboxes_np, frame_bgr)
            if batch_features is not None and len(batch_features) == len(track_ids):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all():
                        current_track_id = int(track_ids[i])
                        features[current_track_id] = det_feature
        except Exception as e: print(f"E: FE call failed: {e}")
        return features

    def _perform_reid_on_tracks(self,
                                features_per_track: Dict[Tuple[str, int], np.ndarray]
                               ) -> Dict[Tuple[str, int], Optional[int]]:
        """Compares features *only for triggered tracks* against the global gallery."""
        # (Identical to previous version)
        assigned_global_ids: Dict[Tuple[str, int], Optional[int]] = {}
        if not features_per_track: return assigned_global_ids
        valid_gallery_ids = [gid for gid, emb in self.reid_gallery.items() if emb is not None and np.isfinite(emb).all()]
        valid_gallery_embeddings = [self.reid_gallery[gid] for gid in valid_gallery_ids]
        match_counts, new_id_counts, update_counts = 0, 0, 0
        for track_key, new_embedding in features_per_track.items():
            assigned_global_ids[track_key] = None
            if new_embedding is None or not np.isfinite(new_embedding).all(): continue
            best_match_global_id, best_match_score = None, 0.0
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
                    updated_embedding = (GALLERY_EMA_ALPHA * current_gallery_emb + (1 - GALLERY_EMA_ALPHA) * new_embedding)
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
                        updated_embedding = (GALLERY_EMA_ALPHA * current_gallery_emb + (1 - GALLERY_EMA_ALPHA) * new_embedding)
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
        return assigned_global_ids


    def process_frame_batch(self, frames: Dict[str, Optional[np.ndarray]], frame_idx: int
                            ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
        """Detect (RTDETR) -> Track (ByteTrack) -> Conditionally Extract/Re-ID (OSNet)."""
        # (Logic is identical to previous version, only _detect_persons call changed internally)
        t_start_batch = time.time()
        timings = {'detection_tracking': 0.0, 'feature_ext': 0.0, 'reid': 0.0, 'total': 0.0}
        final_results_per_camera: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        all_current_track_features: Dict[Tuple[str, int], np.ndarray] = {}
        t_det_track_start = time.time()
        current_frame_tracker_outputs: Dict[str, np.ndarray] = {}
        current_frame_active_track_ids: Dict[str, Set[int]] = defaultdict(set)
        tracks_needing_reid_this_frame: Set[Tuple[str, int]] = set()
        tracks_to_extract_features_for: Dict[str, List[np.ndarray]] = defaultdict(list)

        for cam_id, frame_bgr in frames.items():
            tracker = self.trackers.get(cam_id)
            if not tracker: continue
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8) if frame_bgr is None else frame_bgr
            np_dets = np.empty((0, 6))
            if frame_bgr is not None and frame_bgr.size > 0:
                detections = self._detect_persons(frame_bgr) # Now uses RTDETR
                if detections:
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in detections])
            try:
                tracked_dets_np = tracker.update(np_dets, dummy_frame)
                current_frame_tracker_outputs[cam_id] = np.array(tracked_dets_np) if tracked_dets_np is not None and len(tracked_dets_np) > 0 else np.empty((0, 8))
            except Exception as e:
                print(f"E: Tracker update failed cam {cam_id}: {e}")
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8))
            if current_frame_tracker_outputs[cam_id].shape[0] > 0:
                previous_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in current_frame_tracker_outputs[cam_id]:
                    if len(track_data) >= 7:
                        track_id = int(track_data[4])
                        current_track_key = (cam_id, track_id)
                        current_frame_active_track_ids[cam_id].add(track_id)
                        is_newly_seen = track_id not in previous_cam_track_ids
                        last_reid = self.track_last_reid_frame.get(current_track_key, -REID_REFRESH_INTERVAL)
                        is_due_for_refresh = (frame_idx - last_reid) >= REID_REFRESH_INTERVAL
                        if is_newly_seen or is_due_for_refresh:
                            tracks_needing_reid_this_frame.add(current_track_key)
                            tracks_to_extract_features_for[cam_id].append(track_data)
                            self.track_last_reid_frame[current_track_key] = frame_idx
        timings['detection_tracking'] = time.time() - t_det_track_start

        t_feat_start = time.time()
        extracted_features_this_frame: Dict[Tuple[str, int], np.ndarray] = {}
        for cam_id, tracks_data_list in tracks_to_extract_features_for.items():
            if tracks_data_list:
                frame_bgr = frames.get(cam_id)
                if frame_bgr is not None and frame_bgr.size > 0:
                    tracks_data_np = np.array(tracks_data_list)
                    features_this_cam = self._extract_features_for_tracks(frame_bgr, tracks_data_np)
                    for track_id, feature in features_this_cam.items():
                        extracted_features_this_frame[(cam_id, track_id)] = feature
        timings['feature_ext'] = time.time() - t_feat_start

        t_reid_start = time.time()
        assigned_global_ids = self._perform_reid_on_tracks(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                 for track_data in tracked_dets_np:
                    if len(track_data) >= 7:
                        x1, y1, x2, y2, track_id_float, conf, cls = track_data[0:7]
                        track_id = int(track_id_float); cls = int(cls)
                        current_track_key = (cam_id, track_id)
                        if current_track_key in assigned_global_ids:
                            global_id = assigned_global_ids[current_track_key]
                        else:
                            global_id = self.track_to_global_id.get(current_track_key, None)
                            if global_id is not None: self.track_to_global_id[current_track_key] = global_id
                        final_results_per_camera[cam_id].append({'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32), 'track_id': track_id, 'global_id': global_id, 'conf': float(conf), 'class_id': cls})

        self.last_seen_track_ids = current_frame_active_track_ids
        all_active_keys_this_frame = set((cid, tid) for cid, tids in current_frame_active_track_ids.items() for tid in tids)
        stale_reid_keys = set(self.track_last_reid_frame.keys()) - all_active_keys_this_frame
        for key in stale_reid_keys: del self.track_last_reid_frame[key]
        stale_global_id_keys = set(self.track_to_global_id.keys()) - all_active_keys_this_frame
        for key in stale_global_id_keys: del self.track_to_global_id[key]

        timings['total'] = time.time() - t_start_batch
        return dict(final_results_per_camera), timings


    def draw_annotations(self,
                         frames: Dict[str, Optional[np.ndarray]],
                         processed_results: Dict[str, List[Dict[str, Any]]]
                         ) -> Dict[str, Optional[np.ndarray]]:
        """Draws bounding boxes with track_id and global_id on frames."""
        # (Identical to previous version)
        annotated_frames: Dict[str, Optional[np.ndarray]] = {}
        default_frame_h, default_frame_w = 1080, 1920
        first_valid_frame_dims_set = False
        for frame in frames.values():
            if frame is not None and frame.size > 0:
                 default_frame_h, default_frame_w = frame.shape[:2]; first_valid_frame_dims_set = True; break
        for cam_id, frame in frames.items():
             if frame is None or frame.size == 0:
                  placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                  cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                  annotated_frames[cam_id] = placeholder; continue
             current_h, current_w = frame.shape[:2]; annotated_frame = frame.copy()
             results_for_cam = processed_results.get(cam_id, [])
             for track_info in results_for_cam:
                  bbox_xyxy = track_info.get('bbox_xyxy'); track_id = track_info.get('track_id'); global_id = track_info.get('global_id'); conf = track_info.get('conf', 0.0)
                  if bbox_xyxy is None: continue
                  x1, y1, x2, y2 = map(int, bbox_xyxy); x1 = max(0, x1); y1 = max(0, y1); x2 = min(current_w, x2); y2 = min(current_h, y2)
                  if x1 >= x2 or y1 >= y2: continue
                  color = (255, 182, 193);
                  if global_id is not None: seed = int(global_id) * 3 + 5; color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)
                  if DRAW_BOUNDING_BOXES: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                  label_parts = [];
                  if SHOW_TRACK_ID and track_id is not None: label_parts.append(f"T:{track_id}")
                  if SHOW_GLOBAL_ID: label_parts.append(f"G:{global_id}" if global_id is not None else "G:?")
                  label = " ".join(label_parts)
                  if label:
                       (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2); ly = y1 - 10 if y1 > (lh + 10) else y1 + lh + 5; ly = max(lh + 5, ly); lx = x1
                       cv2.rectangle(annotated_frame, (lx, ly - lh - bl), (lx + lw, ly), color, cv2.FILLED)
                       cv2.putText(annotated_frame, label, (lx, ly - bl//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
             annotated_frames[cam_id] = annotated_frame
        return annotated_frames


# --- Main Execution ---
if __name__ == "__main__":
    # Determine device first
    selected_device = get_and_set_device()

    # --- Find BoxMOT Path and Config ---
    BYTETRACK_CONFIG_PATH = BOXMOT_PATH / "configs" / f"{TRACKER_TYPE}.yaml"
    # (Fallback logic kept)
    if not BYTETRACK_CONFIG_PATH.is_file():
         print(f"W: Could not find {TRACKER_TYPE}.yaml at {BYTETRACK_CONFIG_PATH}")
         script_dir = Path(__file__).parent.resolve()
         fallback_paths = [
             script_dir / "boxmot" / "configs" / f"{TRACKER_TYPE}.yaml",
             script_dir / "configs" / f"{TRACKER_TYPE}.yaml",
             script_dir / f"{TRACKER_TYPE}.yaml"]
         found = False
         for i, path in enumerate(fallback_paths):
             if path.is_file(): BYTETRACK_CONFIG_PATH = path; print(f"Found config at fallback {i+1}: {BYTETRACK_CONFIG_PATH}"); found = True; break
         if not found: print(f"FATAL ERROR: Could not find {TRACKER_TYPE}.yaml config."); sys.exit(1)

    # Validate ReID weights path
    if not REID_MODEL_WEIGHTS.is_file():
        print(f"W: ReID weights not found at: {REID_MODEL_WEIGHTS}")
        script_dir = Path(__file__).parent.resolve()
        fallback_reid_path = script_dir / REID_MODEL_WEIGHTS.name
        if fallback_reid_path.is_file(): REID_MODEL_WEIGHTS = fallback_reid_path; print(f"Found ReID weights at fallback: {REID_MODEL_WEIGHTS}")
        else: print(f"FATAL ERROR: ReID weights not found at {REID_MODEL_WEIGHTS} or {fallback_reid_path}."); sys.exit(1)

    # Validate Detector weights path
    detector_path = Path(DETECTOR_MODEL_PATH)
    if not detector_path.is_file():
        print(f"W: Detector weights not found at: {detector_path}")
        script_dir = Path(__file__).parent.resolve()
        fallback_detector_path = script_dir / detector_path.name
        if fallback_detector_path.is_file(): DETECTOR_MODEL_PATH = str(fallback_detector_path); print(f"Found detector weights at fallback: {DETECTOR_MODEL_PATH}")
        else: print(f"FATAL ERROR: Detector weights not found at {detector_path} or {fallback_detector_path}."); sys.exit(1)


    # --- Construct Camera Paths and Validate ---
    # (Identical to previous version)
    camera_base_dirs = {}
    valid_cameras = []
    base_scene_path = Path(DATASET_BASE_PATH) / "train" / "train" / SELECTED_SCENE
    if not base_scene_path.is_dir(): print(f"FATAL ERROR: Scene dir not found: {base_scene_path}"); sys.exit(1)
    print("\n--- Validating Camera Directories ---")
    for cam_id in SELECTED_CAMERAS:
        rgb_dir = base_scene_path / cam_id / "rgb"
        if rgb_dir.is_dir(): camera_base_dirs[cam_id] = str(rgb_dir); valid_cameras.append(cam_id); print(f"Found valid dir: {rgb_dir}")
        else: print(f"W: Dir not found for {cam_id}. Skipping.")
    if not valid_cameras: print(f"FATAL ERROR: No valid cameras found."); sys.exit(1)
    print(f"Processing cameras: {valid_cameras}")

    # --- Determine Frame Sequence ---
    # (Identical to previous version)
    image_filenames = []
    try:
        first_cam_id = valid_cameras[0]; first_cam_dir = camera_base_dirs[first_cam_id]
        image_filenames = sorted_alphanumeric([f for f in os.listdir(first_cam_dir) if f.lower().endswith(".jpg")])
        if not image_filenames: raise ValueError(f"No JPGs in {first_cam_dir}")
        print(f"\nFound {len(image_filenames)} frames based on {first_cam_id}.")
    except Exception as e: print(f"FATAL ERROR listing image files: {e}"); sys.exit(1)

    # --- Initialize Pipeline ---
    pipeline = MultiCameraPipeline(
        detector_weights_path=DETECTOR_MODEL_PATH, # Pass RTDETR path
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
        t_frame_start = time.time()

        # --- Load Frames ---
        # (Identical to previous version)
        current_frames_bgr: Dict[str, Optional[np.ndarray]] = {}
        valid_frame_loaded = False
        for cam_id in valid_cameras:
            image_path = os.path.join(camera_base_dirs[cam_id], current_filename)
            img = None
            if os.path.exists(image_path):
                try: img = cv2.imread(image_path)
                except Exception as e: print(f"E: reading {image_path}: {e}")
                if img is not None and img.size > 0: current_frames_bgr[cam_id] = img; valid_frame_loaded = True
                else: current_frames_bgr[cam_id] = None
            else: current_frames_bgr[cam_id] = None
        if not valid_frame_loaded: print(f"W: No valid frames index {frame_idx}. Processing empty.")
        actual_frames_processed += 1

        # --- Process Batch ---
        processed_results, timings = pipeline.process_frame_batch(current_frames_bgr, frame_idx)

        # --- Print Timings ---
        # (Identical to previous version)
        t_frame_end = time.time(); frame_proc_time = t_frame_end - t_frame_start
        if frame_idx % 10 == 0 or frame_idx < 5:
            print(f"\n--- Frame {frame_idx} ({current_filename}) --- Frame Proc Time: {frame_proc_time*1000:.1f} ms ---")
            print(f"  Pipeline Timings (ms): Total={timings['total']*1000:.1f} | Detect+Track={timings['detection_tracking']*1000:.1f} | FeatExtract={timings['feature_ext']*1000:.1f} | ReID={timings['reid']*1000:.1f}")
            track_count = sum(len(tracks) for tracks in processed_results.values()); print(f"  Active Tracks: {track_count}")

        # --- Draw Annotations ---
        annotated_frames = pipeline.draw_annotations(current_frames_bgr, processed_results)

        # --- Visualization ---
        # (Identical to previous version)
        valid_annotated = [f for f in annotated_frames.values() if f is not None and f.size > 0] # Simplified collection
        combined_display = None
        if valid_annotated:
            num_cams = len(valid_annotated)
            rows = int(np.ceil(np.sqrt(num_cams))); cols = int(np.ceil(num_cams / rows))
            try: target_h, target_w = valid_annotated[0].shape[:2]
            except IndexError: target_h, target_w = 480, 640
            combined_h, combined_w = rows * target_h, cols * target_w
            combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            frame_num = 0
            for r in range(rows):
                for c in range(cols):
                    if frame_num < num_cams:
                        f = valid_annotated[frame_num]
                        if f.shape[0]!= target_h or f.shape[1]!= target_w:
                             try: f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
                             except Exception: f = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                        combined_display[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = f
                        frame_num += 1
            disp_h, disp_w = combined_display.shape[:2]
            if disp_w > MAX_DISPLAY_WIDTH:
                scale = MAX_DISPLAY_WIDTH / disp_w; disp_h = int(disp_h * scale); disp_w = MAX_DISPLAY_WIDTH
                combined_display = cv2.resize(combined_display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
             combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
             cv2.putText(combined_display, "No Frames", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, combined_display)
        key = cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF
        if key == ord('q'): print("Quitting..."); break
        elif key == ord('p'): print("Paused."); cv2.waitKey(0)

    # --- Cleanup ---
    # (Identical to previous version)
    end_time = time.time(); total_time = end_time - start_time
    print(f"\n--- Pipeline Finished ---")
    print(f"Processed {actual_frames_processed} frame indices.")
    print(f"Total time: {total_time:.2f} seconds.")
    if actual_frames_processed > 0 and total_time > 0: print(f"Avg FPS: {actual_frames_processed / total_time:.2f}")
    else: print("Avg FPS: N/A")
    cv2.destroyAllWindows()
    print("Resources released.")