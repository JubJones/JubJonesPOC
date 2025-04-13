# -*- coding: utf-8 -*-
import os
import sys
import time
import re # Added for sorting
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.appearance.backends.base_backend import BaseModelBackend
from scipy.spatial.distance import cosine as cosine_distance
from ultralytics import RTDETR
from ultralytics.engine.results import Results # To help with type hinting

# --- Configuration ---
DETECTOR_MODEL_PATH = "rtdetr-x.pt"  # RTDETR model for detection
REID_MODEL_WEIGHTS = "osnet_x0_25_msmt17.pt" # OSNet model for Re-ID

# --- MTMMC Dataset Configuration ---
# !! MODIFY THESE PATHS !!
# Set the root path of the MTMMC dataset (Example for Windows)
DATASET_BASE_PATH = r"D:\MTMMC" # Use raw string or double backslashes
# DATASET_BASE_PATH = "/Volumes/HDD/MTMMC" # Example for macOS/Linux
SELECTED_SCENE = "s10"
SELECTED_CAMERAS = ["c09", "c12", "c13", "c16"] # e.g., ["c01", "c02", "c09", "c16"]

# --- General Configuration ---
PERSON_CLASS_ID = 0  # COCO class ID for 'person'
CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence for RTDETR

# --- Re-ID Configuration ---
REID_SIMILARITY_THRESHOLD = 0.65 # Threshold for matching embeddings
REID_TRIGGER_AGE = 5          # Min age for Condition 1 (New Track Confirmation)
PERIODIC_REID_INTERVAL = 30   # Frame interval for Condition 4 (Periodic Refresh)
MAX_TRACK_AGE_BEFORE_DELETE = 60 # Frames a track can be 'lost' before removal
GALLERY_EMA_ALPHA = 0.9       # Exponential Moving Average alpha for updating gallery embeddings

# --- Visualization ---
DRAW_BOUNDING_BOXES = True
SHOW_TRACK_ID = True
SHOW_GLOBAL_ID = True
WINDOW_NAME = "Multi-Camera Re-ID POC (MTMMC Images - Multi-Instance Detector)"
DISPLAY_WAIT_MS = 1 # Wait time for cv2.waitKey (1 for fast playback, 0 for pause on each frame)
MAX_DISPLAY_WIDTH = 1920 # Max width for the combined display window (optional resizing)

# --- Helper Functions ---

def sorted_alphanumeric(data):
    """Sorts a list of strings alphanumerically (handling numbers correctly)."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_device() -> torch.device:
    """Selects the best available compute device."""
    print("--- Determining Device ---")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps")
            print("Using Metal Performance Shaders (MPS) device.")
            return torch.device("mps")
        except Exception as e:
            print(f"MPS reported available, but test failed ({e}). Falling back...")

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
            _ = torch.tensor([1.0], device="cuda") + torch.tensor([1.0], device="cuda")
            return device
        except Exception as e:
             print(f"CUDA reported available, but test failed ({e}). Falling back...")
    else:
        print("CUDA not available.")

    print("Using CPU device.")
    return torch.device("cpu")


def calculate_cosine_similarity(feat1: Optional[np.ndarray], feat2: Optional[np.ndarray]) -> float:
    """Calculates cosine similarity between two feature vectors. Handles None inputs."""
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten(); feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if np.all(feat1 == 0) or np.all(feat2 == 0) or not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0
    try:
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        distance = max(0.0, float(distance)); distance = min(2.0, distance)
    except Exception as e: print(f"Error calculating cosine distance: {e}"); return 0.0
    similarity = 1.0 - distance
    return float(np.clip(similarity, 0.0, 1.0))


def crop_image(image: np.ndarray, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
    """Crops an image based on xyxy bounding box, handles boundary checks."""
    if image is None or bbox_xyxy is None or len(bbox_xyxy) != 4: return None
    h, w = image.shape[:2]; x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x1 >= x2 or y1 >= y2: return None
    return image[y1:y2, x1:x2].copy()

# --- Main Tracking and Re-ID Class ---

class MultiCameraReIDTracker:
    def __init__(self,
                 detector_path: str,
                 reid_weights_path: str,
                 camera_ids: List[str],
                 device: torch.device):
        self.device = device
        self.camera_ids = camera_ids
        print(f"\n--- Loading Models on Device: {self.device} ---")
        print(f"--- WARNING: Initializing SEPARATE detector instances for {len(camera_ids)} cameras. This will increase memory usage and load time. ---")

        # Load Multiple Detector Instances (RTDETR)
        self.detectors: Dict[str, RTDETR] = {}
        try:
            dummy_frame_det = np.zeros((640, 640, 3), dtype=np.uint8)
            for cam_id in self.camera_ids:
                print(f"Loading RTDETR detector for {cam_id} from: {detector_path}")
                detector_instance = RTDETR(detector_path)
                detector_instance.to(self.device)
                print(f"Warming up detector for {cam_id}...")
                _ = detector_instance.predict(dummy_frame_det, verbose=False, device=self.device)
                self.detectors[cam_id] = detector_instance
                print(f"RTDETR Detector for {cam_id} loaded successfully.")
        except Exception as e: print(f"FATAL ERROR loading detector instance: {e}"); sys.exit(1)

        # Load Single ReID Model Instance (OSNet)
        try:
            print(f"Loading SHARED OSNet ReID model from: {reid_weights_path}")
            weights_path_obj = Path(reid_weights_path); reid_device_specifier = 'cpu'
            if self.device.type == 'cuda':
                cuda_index = self.device.index if self.device.index is not None else 0
                reid_device_specifier = str(cuda_index)
                print(f"Using CUDA device index '{reid_device_specifier}' for ReID model.")
            elif self.device.type == 'mps': reid_device_specifier = 'mps'; print(f"Attempting to use MPS device '{reid_device_specifier}' for ReID model.")
            elif self.device.type == 'cpu': reid_device_specifier = 'cpu'; print(f"Using CPU device '{reid_device_specifier}' for ReID model.")
            self.reid_model_handler = ReidAutoBackend(weights=weights_path_obj, device=reid_device_specifier)
            self.reid_model: BaseModelBackend = self.reid_model_handler.model
            if hasattr(self.reid_model, "warmup"): print("Warming up OSNet ReID model..."); self.reid_model.warmup()
            print("OSNet ReID Model loaded successfully.")
        except Exception as e: print(f"FATAL ERROR loading ReID model with device specifier '{reid_device_specifier}': {e}"); sys.exit(1)

        # State Management
        self.track_states: Dict[int, Dict[str, Any]] = {}
        self.reid_gallery: Dict[int, np.ndarray] = {}
        self.next_global_id = 1

    # --- Internal methods (_update_track_states, etc.) remain the same as previous version ---
    # --- (omitted for brevity, they are unchanged logically) ---
    def _update_track_states(self, current_detections: Dict[str, Results], current_frame_idx: int):
        current_time = time.time(); seen_tracker_ids_this_frame = set()
        for camera_id, results in current_detections.items():
            if results is None or results.boxes is None or results.boxes.id is None: continue
            boxes_xyxy = results.boxes.xyxy.cpu().numpy(); tracker_ids = results.boxes.id.int().cpu().tolist()
            for i, tracker_id in enumerate(tracker_ids):
                seen_tracker_ids_this_frame.add(tracker_id); bbox_xyxy = boxes_xyxy[i]
                if tracker_id not in self.track_states:
                    self.track_states[tracker_id] = {'global_id': None, 'last_seen_frame': current_frame_idx,'last_update_time': current_time, 'age': 1, 'state': 'new','last_embedding': None, 'last_camera': camera_id,'last_bbox_xyxy': bbox_xyxy, 'needs_reid': True,'history': deque([(bbox_xyxy[0] + bbox_xyxy[2]) / 2, bbox_xyxy[3]], maxlen=50)}
                else:
                    track = self.track_states[tracker_id]; previous_state = track['state']
                    track['last_seen_frame'] = current_frame_idx; track['last_update_time'] = current_time
                    track['last_camera'] = camera_id; track['last_bbox_xyxy'] = bbox_xyxy
                    track['history'].append(((bbox_xyxy[0] + bbox_xyxy[2]) / 2, bbox_xyxy[3]))
                    if previous_state == 'lost': track['state'] = 'tracked'; track['age'] = 1; track['needs_reid'] = True
                    else: track['state'] = 'tracked'; track['age'] += 1
        lost_track_ids = []
        for tracker_id, track in self.track_states.items():
            if tracker_id not in seen_tracker_ids_this_frame and track['state'] != 'lost': track['state'] = 'lost'; track['age'] = 0; lost_track_ids.append(tracker_id)
        ids_to_delete = [ tid for tid, trk in self.track_states.items() if trk['state'] == 'lost' and (current_frame_idx - trk['last_seen_frame'] > MAX_TRACK_AGE_BEFORE_DELETE)]
        for tracker_id in ids_to_delete:
             if tracker_id in self.track_states: del self.track_states[tracker_id]

    def _determine_reid_needs(self, current_frame_idx: int):
        for tracker_id, track in self.track_states.items():
            if track['needs_reid'] or track['state'] == 'lost': continue
            if track['global_id'] is None and track['age'] >= REID_TRIGGER_AGE: track['needs_reid'] = True; continue
            if track['global_id'] is not None and track['age'] > 0 and (track['age'] % PERIODIC_REID_INTERVAL == 0): track['needs_reid'] = True; continue

    def _extract_features(self, frames: Dict[str, np.ndarray]) -> Dict[int, np.ndarray]:
        tracks_to_extract = []; features = {}
        for tid, trk in self.track_states.items():
            if trk is not None and trk.get('needs_reid') and trk.get('state') != 'lost' and trk.get('last_bbox_xyxy') is not None:
                 tracks_to_extract.append((tid, trk['last_bbox_xyxy'], trk['last_camera']))
        if not tracks_to_extract: return features
        tracks_by_camera = defaultdict(list); bboxes_by_camera = defaultdict(list); tracker_ids_by_camera = defaultdict(list)
        for tid, bbox, cam_id in tracks_to_extract:
            if frames.get(cam_id) is not None: bboxes_by_camera[cam_id].append(bbox.astype(np.float32)); tracker_ids_by_camera[cam_id].append(tid)
        # extraction_time = 0.0; total_features_extracted = 0 # Only used internally if printing timing here
        for cam_id, bboxes_list in bboxes_by_camera.items():
            # t_start = time.time(); # Only used internally if printing timing here
            frame_bgr = frames.get(cam_id)
            if frame_bgr is None or frame_bgr.size == 0: continue
            bboxes_np = np.array(bboxes_list)
            if bboxes_np.ndim != 2 or bboxes_np.shape[1] != 4: continue
            try:
                batch_features = self.reid_model.get_features(bboxes_np, frame_bgr)
                # t_end = time.time(); extraction_time += (t_end - t_start) # Only used internally if printing timing here
                if batch_features is not None and len(batch_features) == len(tracker_ids_by_camera[cam_id]):
                    # total_features_extracted += len(batch_features) # Only used internally if printing timing here
                    for i, tracker_id in enumerate(tracker_ids_by_camera[cam_id]):
                        if batch_features[i] is not None and np.isfinite(batch_features[i]).all(): features[tracker_id] = batch_features[i]
            except Exception as e: print(f"E: Feature extraction call for cam {cam_id}: {e}")
        return features

    def _perform_reid(self, new_features: Dict[int, np.ndarray], current_frame_idx: int):
        if not new_features: return
        gallery_ids = list(self.reid_gallery.keys())
        gallery_embeddings = [emb for emb in self.reid_gallery.values() if emb is not None]
        valid_gallery_ids = [gid for gid, emb in self.reid_gallery.items() if emb is not None]
        match_counts = 0; new_id_counts = 0
        for tracker_id, new_embedding in new_features.items():
            if tracker_id not in self.track_states: continue
            if new_embedding is None:
                 if tracker_id in self.track_states: self.track_states[tracker_id]['needs_reid'] = False
                 continue
            track = self.track_states[tracker_id]; best_match_global_id = None; best_match_score = 0.0
            if valid_gallery_ids:
                similarities = [calculate_cosine_similarity(new_embedding, gal_emb) for gal_emb in gallery_embeddings]
                if similarities:
                    max_similarity = max(similarities) if similarities else 0.0
                    if max_similarity >= REID_SIMILARITY_THRESHOLD: best_match_index = np.argmax(similarities); best_match_global_id = valid_gallery_ids[best_match_index]; best_match_score = max_similarity
            if best_match_global_id is not None:
                matched_global_id = best_match_global_id
                if track['global_id'] is not None and track['global_id'] != matched_global_id: print(f"[F:{current_frame_idx}] Potential ID Switch: T:{tracker_id} (was G:{track['global_id']}) -> G:{matched_global_id} (S:{best_match_score:.3f})")
                track['global_id'] = matched_global_id; match_counts += 1
                current_gallery_emb = self.reid_gallery.get(matched_global_id)
                if current_gallery_emb is not None:
                    updated_embedding = (GALLERY_EMA_ALPHA * current_gallery_emb + (1 - GALLERY_EMA_ALPHA) * new_embedding)
                    norm = np.linalg.norm(updated_embedding)
                    if norm > 1e-6: self.reid_gallery[matched_global_id] = updated_embedding / norm
                    else: self.reid_gallery[matched_global_id] = updated_embedding
            else:
                if track['global_id'] is None:
                    new_global_id = self.next_global_id; self.next_global_id += 1; track['global_id'] = new_global_id
                    norm = np.linalg.norm(new_embedding); new_id_counts += 1
                    if norm > 1e-6: self.reid_gallery[new_global_id] = new_embedding / norm
                    else: self.reid_gallery[new_global_id] = new_embedding
            track['last_embedding'] = new_embedding; track['needs_reid'] = False


    # Modified process_frame_batch to measure and return timings
    def process_frame_batch(self, frames: Dict[str, Optional[np.ndarray]], frame_idx: int) -> Tuple[Dict[str, Optional[Results]], Dict[str, float]]:
        """
        Main processing function. Uses dedicated detector per camera.
        Returns detection results and a dictionary of timings.
        """
        t_start_batch = time.time()
        timings = {'detection': 0.0, 'state': 0.0, 'feature_ext': 0.0, 'reid': 0.0, 'total': 0.0}

        if not frames:
            return {}, timings

        # --- 1. Detection and Tracking ---
        current_detections: Dict[str, Optional[Results]] = {}
        t_det_start = time.time()
        for cam_id, frame_bgr in frames.items():
            if frame_bgr is None: current_detections[cam_id] = None; continue
            if cam_id not in self.detectors: print(f"W: No detector for {cam_id}."); current_detections[cam_id] = None; continue
            detector_instance = self.detectors[cam_id]
            try:
                frame_contiguous = np.ascontiguousarray(frame_bgr)
                results = detector_instance.track(frame_contiguous, persist=True, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, device=self.device, verbose=False, tracker="bytetrack.yaml")
                current_detections[cam_id] = results[0] if results else None
            except Exception as e: print(f"E: tracking cam {cam_id}: {e}"); current_detections[cam_id] = None
        timings['detection'] = time.time() - t_det_start

        # --- 2. Update State & Determine ReID Needs ---
        t_state_start = time.time()
        self._update_track_states(current_detections, frame_idx)
        self._determine_reid_needs(frame_idx)
        timings['state'] = time.time() - t_state_start

        # --- 3. Extract Features ---
        t_feat_start = time.time()
        new_features = self._extract_features(frames)
        timings['feature_ext'] = time.time() - t_feat_start

        # --- 4. Perform Re-ID ---
        t_reid_start = time.time()
        self._perform_reid(new_features, frame_idx)
        timings['reid'] = time.time() - t_reid_start

        timings['total'] = time.time() - t_start_batch
        return current_detections, timings # Return results and timings


    # draw_annotations remains unchanged
    def draw_annotations(self, frames: Dict[str, Optional[np.ndarray]]) -> Dict[str, Optional[np.ndarray]]:
        """Draws bounding boxes with tracker_id and global_id on frames."""
        annotated_frames = {}; default_frame_h, default_frame_w = 1080, 1920 # Default to typical HD size
        first_valid_frame_dims_set = False

        for cam_id, frame in frames.items():
            if frame is None:
                 # Update default size if not set yet and some frames ARE valid (should have been done by now ideally)
                 if not first_valid_frame_dims_set:
                      for f in frames.values():
                           if f is not None:
                                default_frame_h, default_frame_w = f.shape[:2]
                                first_valid_frame_dims_set = True
                                break
                 placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                 cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                 annotated_frames[cam_id] = placeholder; continue

            current_h, current_w = frame.shape[:2]
            if not first_valid_frame_dims_set: # Set default on first valid frame
                default_frame_h, default_frame_w = current_h, current_w
                first_valid_frame_dims_set = True

            annotated_frame = frame.copy()
            for tracker_id, track in self.track_states.items():
                 if (track is not None and
                     track.get('last_camera') == cam_id and
                     track.get('state') == 'tracked' and
                     track.get('last_bbox_xyxy') is not None):
                    bbox = track['last_bbox_xyxy']; global_id = track.get('global_id'); age = track.get('age', 0)
                    x1, y1, x2, y2 = map(int, bbox); x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(current_w, x2), min(current_h, y2)
                    if x1 >= x2 or y1 >= y2: continue
                    color = (255, 182, 193); # Pink if no global ID
                    if global_id is not None: seed = global_id * 3 + 5; color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)
                    if DRAW_BOUNDING_BOXES: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label_parts = []
                    if SHOW_TRACK_ID: label_parts.append(f"T:{tracker_id}")
                    if SHOW_GLOBAL_ID: gid_str = f"G:{global_id}" if global_id is not None else "G:None"; label_parts.append(gid_str)
                    label = " ".join(label_parts)
                    if label:
                        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        ly = y1 - 10 if y1 > (lh + 10) else y1 + lh + 5; ly = max(lh + 5, ly); lx = x1
                        cv2.rectangle(annotated_frame, (lx, ly - lh - bl), (lx + lw, ly), color, cv2.FILLED)
                        cv2.putText(annotated_frame, label, (lx, ly - bl//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
            annotated_frames[cam_id] = annotated_frame
        return annotated_frames

# --- Main Execution ---
if __name__ == "__main__":
    selected_device = get_device()

    # --- Construct Camera Paths and Validate ---
    camera_base_dirs = {}; valid_cameras = []
    base_scene_path = os.path.join(DATASET_BASE_PATH, "train", "train", SELECTED_SCENE)
    if not os.path.isdir(base_scene_path): print(f"FATAL ERROR: Scene directory not found: {base_scene_path}"); sys.exit(1)
    for cam_id in SELECTED_CAMERAS:
        cam_dir = os.path.join(base_scene_path, cam_id); rgb_dir = os.path.join(cam_dir, "rgb")
        if os.path.isdir(rgb_dir): camera_base_dirs[cam_id] = rgb_dir; valid_cameras.append(cam_id); print(f"Found valid camera directory: {rgb_dir}")
        else: print(f"W: RGB directory not found for {cam_id} in scene {SELECTED_SCENE}. Skipping.")
    if not valid_cameras: print(f"FATAL ERROR: No valid cameras found for {SELECTED_CAMERAS} in {SELECTED_SCENE}."); sys.exit(1)
    print(f"Processing cameras: {valid_cameras}")

    # --- Determine Frame Sequence and Count ---
    image_filenames = []
    try:
        first_cam_id = valid_cameras[0]; first_cam_dir = camera_base_dirs[first_cam_id]
        image_filenames = sorted_alphanumeric([f for f in os.listdir(first_cam_dir) if f.lower().endswith(".jpg")])
        if not image_filenames: raise ValueError(f"No JPG files found in {first_cam_dir}")
        max_proc_frames = len(image_filenames); print(f"Found {max_proc_frames} frames based on {first_cam_id}.")
    except Exception as e: print(f"FATAL ERROR: Could not list image files: {e}"); sys.exit(1)

    # --- Initialize the Tracker (passing valid_cameras) ---
    tracker = MultiCameraReIDTracker(detector_path=DETECTOR_MODEL_PATH, reid_weights_path=REID_MODEL_WEIGHTS, camera_ids=valid_cameras, device=selected_device)

    # --- Initialize Display Window ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Create resizable window

    # --- Processing Loop ---
    start_time = time.time(); actual_frames_processed = 0
    for frame_idx, current_filename in enumerate(image_filenames):
        current_frames_bgr = {}
        valid_frame_loaded = False
        for cam_id in valid_cameras:
            image_path = os.path.join(camera_base_dirs[cam_id], current_filename)
            if os.path.exists(image_path):
                try: img = cv2.imread(image_path);
                except Exception as e: print(f"E: reading {image_path}: {e}"); img = None
                if img is not None: current_frames_bgr[cam_id] = img; valid_frame_loaded = True
                else: current_frames_bgr[cam_id] = None
            else: current_frames_bgr[cam_id] = None
        if not valid_frame_loaded: print(f"W: No valid frames loaded for index {frame_idx} ({current_filename}). Stopping."); break
        actual_frames_processed += 1

        # Process the batch and get timings
        _, timings = tracker.process_frame_batch(current_frames_bgr, frame_idx)

        # Print detailed timings periodically
        if frame_idx % 10 == 0 or frame_idx < 5:
            print(f"\n--- Frame {frame_idx} ({current_filename}) ---")
            print(f"  Timings (ms): Total={timings['total']*1000:.1f} | "
                  f"Detect/Track={timings['detection']*1000:.1f} | "
                  f"StateUpdate={timings['state']*1000:.1f} | "
                  f"FeatExtract={timings['feature_ext']*1000:.1f} | "
                  f"ReID={timings['reid']*1000:.1f}")

        # Draw annotations
        annotated_frames = tracker.draw_annotations(current_frames_bgr)

        # --- Visualization ---
        valid_annotated = [f for cam_id in valid_cameras for f in [annotated_frames.get(cam_id)] if f is not None]
        combined_display = None
        if valid_annotated:
            num_cams = len(valid_annotated); rows = int(np.ceil(np.sqrt(num_cams))); cols = int(np.ceil(num_cams / rows))
            target_h, target_w = valid_annotated[0].shape[:2]
            combined_h, combined_w = rows * target_h, cols * target_w
            combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            frame_num_in_batch = 0
            for r in range(rows):
                for c in range(cols):
                    if frame_num_in_batch < num_cams:
                         frame_to_place = valid_annotated[frame_num_in_batch]
                         if frame_to_place.shape[0]!= target_h or frame_to_place.shape[1]!= target_w:
                              try: frame_to_place = cv2.resize(frame_to_place, (target_w, target_h))
                              except Exception: frame_to_place = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                         combined_display[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame_to_place
                         frame_num_in_batch += 1

            # --- Optional Resizing for Display ---
            display_h, display_w = combined_display.shape[:2]
            if display_w > MAX_DISPLAY_WIDTH:
                scale = MAX_DISPLAY_WIDTH / display_w
                display_h = int(display_h * scale)
                display_w = MAX_DISPLAY_WIDTH
                combined_display = cv2.resize(combined_display, (display_w, display_h), interpolation=cv2.INTER_AREA)
            # ------------------------------------

        else:
             combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
             cv2.putText(combined_display, "No Frames Processed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, combined_display)
        if cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF == ord('q'): print("Quitting..."); break

    # --- Cleanup ---
    end_time = time.time(); total_time = end_time - start_time
    print(f"\n--- POC Finished ---"); print(f"Processed {actual_frames_processed} frame indices.")
    print(f"Total execution time: {total_time:.2f} seconds.")
    if actual_frames_processed > 0 and total_time > 0: print(f"Average FPS: {actual_frames_processed / total_time:.2f}")
    cv2.destroyAllWindows()