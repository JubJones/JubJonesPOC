# -*- coding: utf-8 -*-
import os
import sys
import time
import re # Added for sorting
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union # Added Union

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image # Needed for FasterRCNN preprocessing

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.appearance.backends.base_backend import BaseModelBackend
from scipy.spatial.distance import cosine as cosine_distance
# Removed ultralytics imports related to RTDETR and Results

# --- Configuration ---
# FASTER_RCNN_MODEL_PATH = "fasterrcnn_resnet50_fpn" # Path ignored for default torchvision weights
REID_MODEL_WEIGHTS = "osnet_x0_25_msmt17.pt" # OSNet model for Re-ID

# --- MTMMC Dataset Configuration ---
# !! MODIFY THESE PATHS !!
# Set the root path of the MTMMC dataset (Example for Windows)
DATASET_BASE_PATH = r"D:\MTMMC" # Use raw string or double backslashes
# DATASET_BASE_PATH = "/Volumes/HDD/MTMMC" # Example for macOS/Linux
SELECTED_SCENE = "s10"
SELECTED_CAMERAS = ["c09", "c12", "c13", "c16"] # e.g., ["c01", "c02", "c09", "c16"]

# --- General Configuration ---
PERSON_CLASS_ID = 1 # COCO class ID for 'person' in TorchVision models is 1
CONFIDENCE_THRESHOLD = 0.5 # Minimum detection confidence for Faster R-CNN

# --- Re-ID Configuration ---
REID_SIMILARITY_THRESHOLD = 0.65 # Threshold for matching embeddings
GALLERY_EMA_ALPHA = 0.9 # Exponential Moving Average alpha for updating gallery embeddings
# Removed tracking-specific Re-ID configs (REID_TRIGGER_AGE, PERIODIC_REID_INTERVAL, MAX_TRACK_AGE_BEFORE_DELETE)

# --- Visualization ---
DRAW_BOUNDING_BOXES = True
# SHOW_TRACK_ID = False # Removed, no tracker IDs anymore
SHOW_GLOBAL_ID = True
WINDOW_NAME = "Multi-Camera Re-ID POC (MTMMC Images - FasterRCNN Detector)"
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
    # NOTE: FasterRCNN from TorchVision might have issues with MPS.
    # Consider forcing CPU ('cpu') or CUDA ('cuda:0') if needed.
    print("--- Determining Device ---")
    # Prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
            # Quick test
            _ = torch.tensor([1.0], device="cuda") + torch.tensor([1.0], device="cuda")
            return device
        except Exception as e:
             print(f"CUDA reported available, but test failed ({e}). Falling back...")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
         try:
             # Quick test
             _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps")
             print("Using Metal Performance Shaders (MPS) device.")
             return torch.device("mps")
         except Exception as e:
             print(f"MPS reported available, but test failed ({e}). Falling back...")
    else:
        print("CUDA / MPS not available.")

    print("Using CPU device.")
    return torch.device("cpu")


def calculate_cosine_similarity(feat1: Optional[np.ndarray], feat2: Optional[np.ndarray]) -> float:
    """Calculates cosine similarity between two feature vectors. Handles None inputs."""
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten(); feat2 = feat2.flatten()
    # Basic shape and zero checks
    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if np.all(feat1 == 0) or np.all(feat2 == 0): return 0.0
    # Check for NaN/inf
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0
    try:
        # Use float64 for precision, handle potential errors
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        # Clamp distance to valid range [0, 2] before calculating similarity
        distance = max(0.0, float(distance)); distance = min(2.0, distance)
    except Exception as e: print(f"Error calculating cosine distance: {e}"); return 0.0
    similarity = 1.0 - distance
    # Final clamp for similarity [0, 1]
    return float(np.clip(similarity, 0.0, 1.0))


def crop_image(image: np.ndarray, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
    """Crops an image based on xyxy bounding box, handles boundary checks."""
    if image is None or bbox_xyxy is None or len(bbox_xyxy) != 4: return None
    h, w = image.shape[:2]; x1, y1, x2, y2 = map(int, bbox_xyxy)
    # Ensure coordinates are within bounds and valid
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x1 >= x2 or y1 >= y2: return None # Invalid box dimensions
    return image[y1:y2, x1:x2].copy()

# --- Main Re-ID Class (Tracking Logic Removed) ---

class MultiCameraReIDProcessor: # Renamed from Tracker
    def __init__(self,
                 reid_weights_path: str,
                 camera_ids: List[str],
                 device: torch.device):
        self.device = device
        self.camera_ids = camera_ids
        print(f"\n--- Loading Models on Device: {self.device} ---")

        # Load Single Faster R-CNN Detector Instance (Shared)
        # NOTE: TorchVision FasterRCNN doesn't require a path for default weights.
        self.detector = None
        self.detector_transforms = None
        try:
            print("Loading SHARED Faster R-CNN (ResNet50 FPN) detector...")
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.detector.to(self.device)
            self.detector.eval()
            self.detector_transforms = weights.transforms()
            # Warmup is implicitly handled by the first inference
            print("Faster R-CNN Detector loaded successfully.")
        except Exception as e: print(f"FATAL ERROR loading Faster R-CNN detector: {e}"); sys.exit(1)

        # Load Single ReID Model Instance (OSNet - Shared)
        self.reid_model: Optional[BaseModelBackend] = None
        try:
            print(f"Loading SHARED OSNet ReID model from: {reid_weights_path}")
            weights_path_obj = Path(reid_weights_path)

            # Determine device specifier for ReidAutoBackend based on self.device
            reid_device_specifier = 'cpu' # Default
            if self.device.type == 'cuda':
                cuda_index = self.device.index if self.device.index is not None else 0
                reid_device_specifier = str(cuda_index)
                print(f"Using CUDA device index '{reid_device_specifier}' for ReID model.")
            elif self.device.type == 'mps':
                reid_device_specifier = 'mps'
                print(f"Attempting to use MPS device '{reid_device_specifier}' for ReID model.")
            elif self.device.type == 'cpu':
                reid_device_specifier = 'cpu'
                print(f"Using CPU device '{reid_device_specifier}' for ReID model.")

            # Initialize ReidAutoBackend
            # Explicitly disable half-precision (half=False) for broader compatibility maybe
            reid_model_handler = ReidAutoBackend(weights=weights_path_obj, device=reid_device_specifier, half=False)
            self.reid_model = reid_model_handler.model # Get the actual backend model

            # Optional warmup call if the backend supports it
            if hasattr(self.reid_model, "warmup"):
                print("Warming up OSNet ReID model...")
                self.reid_model.warmup()
            print("OSNet ReID Model loaded successfully.")

        except ImportError as e: print(f"FATAL ERROR loading ReID model: BoxMOT/ReID dependencies might be missing. {e}"); sys.exit(1)
        except FileNotFoundError as e: print(f"FATAL ERROR loading ReID model: Weights file not found at {reid_weights_path}. {e}"); sys.exit(1)
        except Exception as e: print(f"FATAL ERROR loading ReID model with device specifier '{reid_device_specifier}': {e}"); sys.exit(1)


        # State Management (Simplified - No Tracking State)
        self.reid_gallery: Dict[int, np.ndarray] = {} # {global_id: embedding}
        self.next_global_id = 1

    # Removed _update_track_states and _determine_reid_needs

    def _detect_persons(self, frames: Dict[str, np.ndarray]) -> Dict[str, List[Dict[str, Any]]]:
        """Detects persons in a batch of frames using Faster R-CNN."""
        detections_per_camera: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if self.detector is None or self.detector_transforms is None:
            print("E: Detector not initialized.")
            return detections_per_camera

        for camera_id, frame_bgr in frames.items():
            if frame_bgr is None or frame_bgr.size == 0:
                continue

            try:
                # 1. Preprocess Frame
                img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                input_tensor = self.detector_transforms(img_pil)
                input_batch = [input_tensor.to(self.device)]

                # 2. Perform Detection
                with torch.no_grad():
                    predictions = self.detector(input_batch)

                # 3. Process Results
                pred_boxes_xyxy = predictions[0]['boxes'].cpu().numpy()
                pred_labels = predictions[0]['labels'].cpu().numpy()
                pred_scores = predictions[0]['scores'].cpu().numpy()

                for box_xyxy, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                    if label == PERSON_CLASS_ID and score >= CONFIDENCE_THRESHOLD:
                        # Ensure box coordinates are valid
                        x1, y1, x2, y2 = box_xyxy
                        if x2 > x1 and y2 > y1:
                             detections_per_camera[camera_id].append({
                                 'bbox_xyxy': box_xyxy.astype(np.float32),
                                 'conf': float(score),
                                 'global_id': None # Initialize global ID as None
                             })

            except Exception as e:
                print(f"E: Detection error on camera {camera_id}: {e}")
                # Add traceback here if needed: import traceback; traceback.print_exc()

        return detections_per_camera


    def _extract_features(self,
                          detections_per_camera: Dict[str, List[Dict[str, Any]]],
                          frames: Dict[str, np.ndarray]) -> Dict[Tuple[str, int], np.ndarray]:
        """
        Extracts Re-ID features for detected persons.
        Returns a dictionary mapping (camera_id, detection_index) to feature embedding.
        """
        features: Dict[Tuple[str, int], np.ndarray] = {}
        if self.reid_model is None:
            print("E: ReID model not initialized.")
            return features

        for camera_id, detections in detections_per_camera.items():
            if not detections: continue # Skip if no detections for this camera
            frame_bgr = frames.get(camera_id)
            if frame_bgr is None or frame_bgr.size == 0: continue # Skip if frame is invalid

            # Prepare batch of bounding boxes for this camera
            bboxes_xyxy_list = [det['bbox_xyxy'] for det in detections]
            if not bboxes_xyxy_list: continue

            bboxes_np = np.array(bboxes_xyxy_list).astype(np.float32)
            if bboxes_np.ndim != 2 or bboxes_np.shape[1] != 4:
                 print(f"W: Invalid bbox shape for feature extraction on cam {camera_id}. Shape: {bboxes_np.shape}")
                 continue

            try:
                # Extract features for all detections in this camera's frame at once
                batch_features = self.reid_model.get_features(bboxes_np, frame_bgr)

                if batch_features is not None and len(batch_features) == len(detections):
                    for i, det_feature in enumerate(batch_features):
                        # Check if feature is valid before storing
                        if det_feature is not None and np.isfinite(det_feature).all():
                            detection_key = (camera_id, i) # Use (cam_id, index) as key
                            features[detection_key] = det_feature
                        # else:
                            # print(f"W: Invalid feature extracted for detection {i} on cam {camera_id}. Skipping.") # Optional warning
                # else:
                    # print(f"W: Feature extraction mismatch or failure for cam {camera_id}. Expected {len(detections)}, got {len(batch_features) if batch_features is not None else 'None'}.") # Optional warning

            except Exception as e:
                print(f"E: Feature extraction call failed for cam {camera_id}: {e}")
                # Optionally print traceback: import traceback; traceback.print_exc()

        return features

    def _perform_reid(self,
                      new_features: Dict[Tuple[str, int], np.ndarray]
                      ) -> Dict[Tuple[str, int], Optional[int]]:
        """
        Compares new features against the gallery and assigns global IDs.
        Returns a dictionary mapping (camera_id, detection_index) to assigned global_id (or None).
        """
        assigned_global_ids: Dict[Tuple[str, int], Optional[int]] = {}
        if not new_features:
            return assigned_global_ids

        # Prepare gallery data (filter out None embeddings if any crept in)
        valid_gallery_ids = [gid for gid, emb in self.reid_gallery.items() if emb is not None and np.isfinite(emb).all()]
        valid_gallery_embeddings = [self.reid_gallery[gid] for gid in valid_gallery_ids]

        match_counts = 0
        new_id_counts = 0

        for detection_key, new_embedding in new_features.items():
            assigned_global_ids[detection_key] = None # Default to None
            if new_embedding is None or not np.isfinite(new_embedding).all():
                continue # Skip invalid new embeddings

            best_match_global_id = None
            best_match_score = 0.0

            # Compare against the gallery only if gallery is not empty
            if valid_gallery_ids:
                similarities = [calculate_cosine_similarity(new_embedding, gal_emb) for gal_emb in valid_gallery_embeddings]

                if similarities: # Check if comparison happened
                    max_similarity = max(similarities)
                    if max_similarity >= REID_SIMILARITY_THRESHOLD:
                        best_match_index = np.argmax(similarities)
                        best_match_global_id = valid_gallery_ids[best_match_index]
                        best_match_score = max_similarity

            # --- Assign Global ID and Update Gallery ---
            if best_match_global_id is not None:
                # Matched existing ID
                matched_global_id = best_match_global_id
                assigned_global_ids[detection_key] = matched_global_id
                match_counts += 1

                # Update gallery embedding using EMA
                current_gallery_emb = self.reid_gallery.get(matched_global_id)
                if current_gallery_emb is not None: # Should exist if matched
                    # Apply EMA update
                    updated_embedding = (GALLERY_EMA_ALPHA * current_gallery_emb +
                                       (1 - GALLERY_EMA_ALPHA) * new_embedding)
                    # Normalize the updated embedding (important for cosine similarity)
                    norm = np.linalg.norm(updated_embedding)
                    if norm > 1e-6: # Avoid division by zero
                        self.reid_gallery[matched_global_id] = updated_embedding / norm
                    else:
                        self.reid_gallery[matched_global_id] = updated_embedding # Store as is if norm is near zero

            else:
                # No match found, assign a new global ID
                new_global_id = self.next_global_id
                self.next_global_id += 1
                assigned_global_ids[detection_key] = new_global_id
                new_id_counts += 1

                # Add the new embedding to the gallery (normalize it)
                norm = np.linalg.norm(new_embedding)
                if norm > 1e-6:
                    self.reid_gallery[new_global_id] = new_embedding / norm
                else:
                    self.reid_gallery[new_global_id] = new_embedding # Add as is if norm is near zero

        # Optional: Print summary for the batch
        # print(f"  Re-ID: Matched {match_counts} detections, Created {new_id_counts} new global IDs.")
        return assigned_global_ids


    def process_frame_batch(self, frames: Dict[str, Optional[np.ndarray]], frame_idx: int
                            ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
        """
        Main processing function using Faster R-CNN for detection and OSNet for Re-ID.
        Returns processed detections with global IDs and timings.
        """
        t_start_batch = time.time()
        timings = {'detection': 0.0, 'feature_ext': 0.0, 'reid': 0.0, 'total': 0.0}

        # --- Input Validation ---
        valid_frames = {cam_id: frame for cam_id, frame in frames.items() if frame is not None and frame.size > 0}
        if not valid_frames:
            # print(f"W: No valid frames in batch for index {frame_idx}. Skipping processing.")
            return {}, timings

        # --- 1. Detection ---
        t_det_start = time.time()
        # Uses the single shared Faster R-CNN detector instance
        current_detections = self._detect_persons(valid_frames)
        timings['detection'] = time.time() - t_det_start

        # --- 2. Extract Features ---
        t_feat_start = time.time()
        # Uses the single shared OSNet ReID model instance
        new_features = self._extract_features(current_detections, valid_frames)
        timings['feature_ext'] = time.time() - t_feat_start

        # --- 3. Perform Re-ID ---
        t_reid_start = time.time()
        assigned_global_ids = self._perform_reid(new_features)
        timings['reid'] = time.time() - t_reid_start

        # --- 4. Integrate Global IDs back into detections ---
        # We modify the current_detections dictionary in-place
        for (camera_id, det_index), global_id in assigned_global_ids.items():
             if camera_id in current_detections and len(current_detections[camera_id]) > det_index:
                  current_detections[camera_id][det_index]['global_id'] = global_id

        # --- 5. Final Timing ---
        timings['total'] = time.time() - t_start_batch

        # Return the detections dictionary which now includes global IDs
        return current_detections, timings


    def draw_annotations(self,
                         frames: Dict[str, Optional[np.ndarray]],
                         processed_detections: Dict[str, List[Dict[str, Any]]]
                         ) -> Dict[str, Optional[np.ndarray]]:
        """Draws bounding boxes with assigned global_id on frames."""
        annotated_frames: Dict[str, Optional[np.ndarray]] = {}
        default_frame_h, default_frame_w = 1080, 1920 # Default size if needed
        first_valid_frame_dims_set = False

        # Determine default dimensions from the first valid input frame
        for frame in frames.values():
            if frame is not None and frame.size > 0:
                 default_frame_h, default_frame_w = frame.shape[:2]
                 first_valid_frame_dims_set = True
                 break

        for cam_id, frame in frames.items():
             # Create placeholder if frame is missing or invalid
             if frame is None or frame.size == 0:
                  placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
                  cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                  annotated_frames[cam_id] = placeholder
                  continue

             # Use frame's actual dimensions
             current_h, current_w = frame.shape[:2]
             annotated_frame = frame.copy() # Work on a copy

             # Iterate through the processed detections for this camera
             detections_for_cam = processed_detections.get(cam_id, [])
             for detection_info in detections_for_cam:
                  bbox_xyxy = detection_info.get('bbox_xyxy')
                  global_id = detection_info.get('global_id') # This might be None
                  conf = detection_info.get('conf', 0.0)

                  if bbox_xyxy is None: continue

                  x1, y1, x2, y2 = map(int, bbox_xyxy)
                  # Clip box to image boundaries
                  x1 = max(0, x1); y1 = max(0, y1); x2 = min(current_w, x2); y2 = min(current_h, y2)
                  if x1 >= x2 or y1 >= y2: continue # Skip invalid boxes

                  # Determine color based on Global ID presence
                  color = (255, 182, 193) # Pink if no global ID (or None)
                  if global_id is not None:
                       # Generate color based on global_id (consistent color per ID)
                       seed = global_id * 3 + 5
                       color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55) # BGR

                  # Draw bounding box
                  if DRAW_BOUNDING_BOXES:
                       cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                  # Prepare label
                  label_parts = []
                  # Removed SHOW_TRACK_ID part
                  if SHOW_GLOBAL_ID:
                       gid_str = f"G:{global_id}" if global_id is not None else "G:None"
                       label_parts.append(gid_str)
                  label_parts.append(f"C:{conf:.2f}") # Add confidence score
                  label = " ".join(label_parts)

                  # Draw label background and text
                  if label:
                       (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                       # Position label above the box, adjusting if near the top edge
                       ly = y1 - 10 if y1 > (lh + 10) else y1 + lh + 5
                       ly = max(lh + 5, ly) # Ensure it's not cut off at the top
                       lx = x1

                       # Draw filled rectangle background for label
                       cv2.rectangle(annotated_frame, (lx, ly - lh - bl), (lx + lw, ly), color, cv2.FILLED)
                       # Draw label text (black color for contrast)
                       cv2.putText(annotated_frame, label, (lx, ly - bl//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

             annotated_frames[cam_id] = annotated_frame

        return annotated_frames

# --- Main Execution ---
if __name__ == "__main__":
    selected_device = get_device()

    # --- Construct Camera Paths and Validate ---
    camera_base_dirs = {}
    valid_cameras = []
    base_scene_path = os.path.join(DATASET_BASE_PATH, "train", "train", SELECTED_SCENE)
    if not os.path.isdir(base_scene_path): print(f"FATAL ERROR: Scene directory not found: {base_scene_path}"); sys.exit(1)

    print("--- Validating Camera Directories ---")
    for cam_id in SELECTED_CAMERAS:
        cam_dir = os.path.join(base_scene_path, cam_id)
        rgb_dir = os.path.join(cam_dir, "rgb")
        if os.path.isdir(rgb_dir):
            camera_base_dirs[cam_id] = rgb_dir
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
        # Filter for jpg files and sort alphanumerically
        image_filenames = sorted_alphanumeric([f for f in os.listdir(first_cam_dir) if f.lower().endswith(".jpg")])
        if not image_filenames: raise ValueError(f"No JPG files found in {first_cam_dir}")
        max_proc_frames = len(image_filenames)
        print(f"Found {max_proc_frames} frames based on {first_cam_id}.")
    except Exception as e: print(f"FATAL ERROR: Could not list image files: {e}"); sys.exit(1)

    # --- Initialize the Processor ---
    processor = MultiCameraReIDProcessor(reid_weights_path=REID_MODEL_WEIGHTS, camera_ids=valid_cameras, device=selected_device)

    # --- Initialize Display Window ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Create resizable window

    # --- Processing Loop ---
    start_time = time.time(); actual_frames_processed = 0
    print("\n--- Starting Frame Processing Loop ---")

    for frame_idx, current_filename in enumerate(image_filenames):
        # --- Load Frames for the Current Index ---
        current_frames_bgr = {}
        valid_frame_loaded_this_iter = False
        for cam_id in valid_cameras:
            image_path = os.path.join(camera_base_dirs[cam_id], current_filename)
            if os.path.exists(image_path):
                try: img = cv2.imread(image_path)
                except Exception as e: print(f"E: reading {image_path}: {e}"); img = None
                if img is not None and img.size > 0:
                    current_frames_bgr[cam_id] = img
                    valid_frame_loaded_this_iter = True
                else: current_frames_bgr[cam_id] = None # Mark as None if read failed or empty
            else: current_frames_bgr[cam_id] = None # Mark as None if file doesn't exist

        # If no valid frames could be loaded for this index across all cameras, stop.
        if not valid_frame_loaded_this_iter:
             print(f"W: No valid frames loaded for index {frame_idx} ({current_filename}) across any camera. Stopping.")
             break
        actual_frames_processed += 1

        # --- Process the Batch ---
        # Process the frames using the processor instance
        processed_detections, timings = processor.process_frame_batch(current_frames_bgr, frame_idx)

        # --- Print Timings ---
        if frame_idx % 10 == 0 or frame_idx < 5: # Print periodically or for first few frames
            print(f"\n--- Frame {frame_idx} ({current_filename}) ---")
            print(f"  Timings (ms): Total={timings['total']*1000:.1f} | "
                  f"Detect={timings['detection']*1000:.1f} | "
                  # Removed StateUpdate timing
                  f"FeatExtract={timings['feature_ext']*1000:.1f} | "
                  f"ReID={timings['reid']*1000:.1f}")
            # Optional: Print counts of detections/matches if needed
            # det_count = sum(len(dets) for dets in processed_detections.values())
            # print(f"  Detected: {det_count} persons across {len(processed_detections)} cameras.")


        # --- Draw Annotations ---
        # Pass the original frames AND the processed detections
        annotated_frames = processor.draw_annotations(current_frames_bgr, processed_detections)

        # --- Visualization ---
        # Combine annotated frames into a single display grid
        valid_annotated = [f for cam_id in valid_cameras for f in [annotated_frames.get(cam_id)] if f is not None and f.size > 0]
        combined_display = None
        if valid_annotated:
            num_cams = len(valid_annotated)
            # Simple grid calculation (adjust rows/cols as needed)
            rows = int(np.ceil(np.sqrt(num_cams))); cols = int(np.ceil(num_cams / rows))
            target_h, target_w = valid_annotated[0].shape[:2] # Use size of first valid frame
            combined_h, combined_w = rows * target_h, cols * target_w
            combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8) # Black background

            frame_num_in_batch = 0
            for r in range(rows):
                for c in range(cols):
                    if frame_num_in_batch < num_cams:
                        frame_to_place = valid_annotated[frame_num_in_batch]
                        # Ensure consistent size before placing
                        if frame_to_place.shape[0]!= target_h or frame_to_place.shape[1]!= target_w:
                             try: frame_to_place = cv2.resize(frame_to_place, (target_w, target_h), interpolation=cv2.INTER_AREA)
                             except Exception: frame_to_place = np.zeros((target_h, target_w, 3), dtype=np.uint8) # Error placeholder
                        # Place the frame
                        combined_display[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = frame_to_place
                        frame_num_in_batch += 1
                    # else: Optional: Fill remaining grid cells with black/placeholder if num_cams is not a perfect square

            # --- Optional Resizing for Display ---
            display_h, display_w = combined_display.shape[:2]
            if display_w > MAX_DISPLAY_WIDTH:
                scale = MAX_DISPLAY_WIDTH / display_w
                display_h = int(display_h * scale)
                display_w = MAX_DISPLAY_WIDTH
                combined_display = cv2.resize(combined_display, (display_w, display_h), interpolation=cv2.INTER_AREA)
            # ------------------------------------

        else: # Handle case where no frames could be annotated/displayed
             combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
             cv2.putText(combined_display, "No Frames to Display", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- Show Combined Display ---
        cv2.imshow(WINDOW_NAME, combined_display)

        # --- Handle User Input (Quit) ---
        if cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # --- Cleanup ---
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- POC Finished ---")
    print(f"Processed {actual_frames_processed} frame indices.")
    print(f"Total execution time: {total_time:.2f} seconds.")
    if actual_frames_processed > 0 and total_time > 0:
        print(f"Average FPS: {actual_frames_processed / total_time:.2f}")
    else:
         print("Average FPS: N/A (no frames processed or zero time elapsed)")

    cv2.destroyAllWindows()
    print("Resources released.")