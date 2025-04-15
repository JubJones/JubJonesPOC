"""Core processing class for the multi-camera detection, tracking, and Re-ID pipeline."""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN

# Ensure necessary torchvision transforms are available
try:
    from torchvision.transforms.v2 import Compose
except ImportError:
    from torchvision.transforms import Compose # Fallback

# BoxMOT imports needed for type hints and ReID model interaction
try:
    from boxmot.appearance.backends.base_backend import BaseModelBackend
    from boxmot.trackers.basetracker import BaseTracker
except ImportError as e:
    logging.critical(f"Failed to import boxmot components needed for Pipeline. Is boxmot installed? Error: {e}")
    # Define dummy types if import fails
    BaseModelBackend = type(None)
    BaseTracker = type(None)


from reid_poc.config import PipelineConfig # Use relative import
from reid_poc.alias_types import ( # Use relative import
    CameraID, TrackID, GlobalID, TrackKey, FeatureVector,
    Detection, TrackData, FrameData, Timings, ProcessedBatchResult, ScaleFactors
)
from reid_poc.utils import calculate_cosine_similarity, normalize_embedding # Use relative import


logger = logging.getLogger(__name__)

class MultiCameraPipeline:
    """Handles multi-camera detection, tracking, and Re-Identification."""

    def __init__(
        self,
        config: PipelineConfig,
        detector: FasterRCNN,
        detector_transforms: Compose,
        reid_model: Optional[BaseModelBackend],
        trackers: Dict[CameraID, BaseTracker]
    ):
        """Initializes the pipeline with pre-loaded models and trackers."""
        self.config = config
        self.device = config.device
        self.camera_ids = config.selected_cameras # Use the potentially updated list from config setup

        # Store pre-initialized components
        self.detector = detector
        self.detector_transforms = detector_transforms
        self.reid_model = reid_model
        self.trackers = trackers

        # --- Logging config details (INFO level) ---
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids} on device: {self.device.type}")
        if config.detection_input_width:
            logger.info(f"Detection resizing ENABLED to width: {config.detection_input_width}")
        else:
            logger.info("Detection resizing DISABLED.")

        if config.use_amp and self.device.type == 'cuda':
            logger.info("AMP (FP16) for detection ENABLED.")
        elif config.use_amp:
            logger.warning("AMP requested but device is not CUDA. AMP disabled.")
        else:
            logger.info("AMP (FP16) for detection DISABLED.")

        if self.config.frame_skip_rate > 1:
            logger.info(f"Frame skipping ENABLED (Processing 1/{self.config.frame_skip_rate} frames).")
        else:
            logger.info("Frame skipping DISABLED (Processing all frames).")

        # --- State Initialization ---
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {} # Stores representative feature for each GlobalID
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {} # Maps (cam_id, track_id) -> global_id
        self.next_global_id: GlobalID = 1 # Counter for assigning new global IDs
        # Stores the set of active track IDs seen in the *last processed* frame for each camera
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        # Stores the processed frame index when ReID was last attempted for a track
        self.track_last_reid_frame: Dict[TrackKey, int] = {}
        # Counts only the frames actively processed by the pipeline (respecting frame skipping)
        self.processed_frame_counter: int = 0

    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        """Extracts Re-ID features for the given tracks using the provided frame."""
        features: Dict[TrackID, FeatureVector] = {}
        if self.reid_model is None:
            logger.warning("ReID model not available, cannot extract features.")
            return features
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Cannot extract features: Invalid frame provided.")
            return features
        if tracked_dets_np.shape[0] == 0:
            return features # No tracks to extract features for

        # Ensure tracked_dets_np has the expected format (at least xyxy + track_id)
        if tracked_dets_np.shape[1] < 5:
             logger.warning(f"Track data has unexpected shape {tracked_dets_np.shape}, expected at least 5 columns (xyxy, id). Skipping feature extraction.")
             return features

        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids_float = tracked_dets_np[:, 4] # Keep as float initially for direct indexing

        # Basic check on bounding boxes
        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4:
            logger.warning(f"Invalid bbox shape {bboxes_xyxy.shape} received for feature extraction. Skipping.")
            return features

        try:
            # Use the ReID model's feature extraction method
            # The backend should handle batching internally if supported
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)

            if batch_features is not None and len(batch_features) == len(track_ids_float):
                for i, det_feature in enumerate(batch_features):
                    # Validate the extracted feature before storing
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        track_id = int(track_ids_float[i]) # Convert track ID to int for dictionary key
                        features[track_id] = det_feature # Store the raw, non-normalized feature here
                    # else: # Optional: Log if a feature extraction failed for a specific box
                    #    logger.debug(f"Feature extraction yielded invalid result for track {int(track_ids_float[i])}")
            # else: # Optional: Log mismatch
                # logger.warning(f"Feature extraction output count ({len(batch_features) if batch_features is not None else 'None'}) mismatch with input tracks ({len(track_ids_float)}).")

        except Exception as e:
            # Log error without traceback for less console spam during runtime issues
            logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features

    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[TrackKey, Optional[GlobalID]]:
        """Associates tracks with Global IDs based on feature similarity to the gallery."""
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {}
        if not features_per_track:
            return newly_assigned_global_ids # Nothing to associate

        # --- Prepare Gallery for Comparison ---
        # Filter gallery for valid, non-None embeddings before comparison
        valid_gallery_items = [
            (gid, emb) for gid, emb in self.reid_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        valid_gallery_ids: List[GlobalID] = []
        valid_gallery_embeddings: List[FeatureVector] = []
        if valid_gallery_items:
            valid_gallery_ids, valid_gallery_embeddings = zip(*valid_gallery_items)
            # Convert embeddings list to numpy array for potentially faster bulk operations if needed later,
            # though individual comparison is done here.
            # valid_gallery_embeddings_np = np.array(valid_gallery_embeddings)

        # --- Iterate Through New Features ---
        for track_key, new_embedding_raw in features_per_track.items():
            newly_assigned_global_ids[track_key] = None # Default to no assignment

            # Validate the new embedding before proceeding
            if new_embedding_raw is None or not np.isfinite(new_embedding_raw).all() or new_embedding_raw.size == 0:
                logger.warning(f"Skipping ReID for {track_key}: Received invalid embedding.")
                continue

            # Normalize the new embedding *before* comparison and gallery update
            normalized_new_embedding = normalize_embedding(new_embedding_raw)

            best_match_global_id: Optional[GlobalID] = None
            max_similarity = -1.0 # Initialize below threshold

            # --- Compare with Existing Gallery ---
            if valid_gallery_ids: # Only compare if the gallery is not empty
                try:
                    similarities = np.array([
                        calculate_cosine_similarity(normalized_new_embedding, gal_emb)
                        for gal_emb in valid_gallery_embeddings # Compare against existing valid gallery embeddings
                    ])

                    if similarities.size > 0:
                        max_similarity_idx = np.argmax(similarities)
                        max_similarity = similarities[max_similarity_idx]

                        # Check if the best match meets the threshold
                        if max_similarity >= self.config.reid_similarity_threshold:
                            best_match_global_id = valid_gallery_ids[max_similarity_idx]

                except Exception as sim_err:
                    logger.error(f"Similarity calculation error during ReID for {track_key}: {sim_err}", exc_info=False)
                    # Continue to next track if similarity calculation fails

            # --- Assign Global ID and Update Gallery ---
            assigned_global_id: Optional[GlobalID] = None

            if best_match_global_id is not None:
                # Case A: Found a match above threshold
                assigned_global_id = best_match_global_id
                current_gallery_emb = self.reid_gallery.get(assigned_global_id)

                if current_gallery_emb is not None:
                    # Update gallery embedding using EMA
                    updated_embedding = (
                        self.config.gallery_ema_alpha * current_gallery_emb +
                        (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding
                    )
                    # Re-normalize after EMA update
                    self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)
                else:
                    # This case should ideally not happen if gallery was prepared correctly, but handle defensively
                    logger.warning(f"Track {track_key}: Matched GID {assigned_global_id} was unexpectedly None in gallery? Overwriting with new embedding.")
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding # Use the already normalized one

            else:
                # Case B: No match found above threshold
                # Check if this track *previously* had a global ID assigned
                last_known_global_id = self.track_to_global_id.get(track_key)

                if last_known_global_id is not None and last_known_global_id in self.reid_gallery:
                    # Case B.1: Track had a previous ID, re-assign it and update gallery
                    assigned_global_id = last_known_global_id
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id) # Should exist based on check

                    if current_gallery_emb is not None:
                         # Update gallery embedding using EMA
                        updated_embedding = (
                            self.config.gallery_ema_alpha * current_gallery_emb +
                            (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding
                        )
                        # Re-normalize after EMA update
                        self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)
                    else:
                         # Defensive handling
                        logger.warning(f"Track {track_key}: Re-assigned GID {assigned_global_id} was unexpectedly None in gallery? Overwriting.")
                        self.reid_gallery[assigned_global_id] = normalized_new_embedding

                else:
                    # Case B.2: No match and no reliable previous ID, assign a new Global ID
                    assigned_global_id = self.next_global_id
                    self.next_global_id += 1
                    # Add the normalized embedding to the gallery for the new ID
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding
                    # logger.info(f"Assigned New GID {assigned_global_id} to {track_key}") # Optional: Log new ID assignments

            # --- Update State Mappings ---
            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                # Update the primary mapping for future lookups
                self.track_to_global_id[track_key] = assigned_global_id

        return newly_assigned_global_ids

    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """Updates the set of last seen tracks and cleans up state for disappeared tracks."""
        # 1. Update last_seen_track_ids based on currently active tracks
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys:
            new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        # 2. Clean up track_last_reid_frame for tracks that are no longer active
        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        if keys_to_delete_reid:
            # logger.debug(f"Cleaning up ReID timestamp for {len(keys_to_delete_reid)} inactive tracks.")
            for key in keys_to_delete_reid:
                del self.track_last_reid_frame[key]

        # 3. Clean up track_to_global_id mapping for tracks that are no longer active
        #    Note: We keep the global ID entry in self.reid_gallery itself, as it might reappear.
        keys_to_delete_global = set(self.track_to_global_id.keys()) - current_frame_active_track_keys
        if keys_to_delete_global:
            # logger.debug(f"Cleaning up Global ID mapping for {len(keys_to_delete_global)} inactive tracks.")
            for key in keys_to_delete_global:
                del self.track_to_global_id[key]


    def process_frame_batch_full(self, frames: Dict[CameraID, FrameData], frame_idx_global: int) -> ProcessedBatchResult:
        """Processes a batch of frames: Detect -> Track -> Re-ID -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)
        self.processed_frame_counter += 1 # Increment counter for frames actually processed
        proc_frame_id = self.processed_frame_counter # Use this for ReID interval checks

        # --- Stage 1a: Preprocess Frames for Detection ---
        t_prep_start = time.time()
        batch_input_tensors: List[torch.Tensor] = []
        batch_cam_ids: List[CameraID] = [] # Keep track of order
        batch_original_shapes: List[Tuple[int, int]] = [] # (height, width)
        batch_scale_factors: List[ScaleFactors] = [] # (scale_x, scale_y) for reverting boxes

        for cam_id, frame_bgr in frames.items():
            if frame_bgr is not None and frame_bgr.size > 0:
                original_h, original_w = frame_bgr.shape[:2]
                frame_for_det = frame_bgr
                scale_x, scale_y = 1.0, 1.0

                # --- Optional Resizing ---
                if self.config.detection_input_width and original_w > self.config.detection_input_width:
                    target_w = self.config.detection_input_width
                    scale = target_w / original_w
                    target_h = int(original_h * scale)
                    try:
                        # Use INTER_LINEAR for downscaling, generally faster and good enough
                        frame_for_det = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        scale_x = original_w / target_w
                        scale_y = original_h / target_h
                    except Exception as resize_err:
                        logger.warning(f"[{cam_id}] Resizing frame to width {target_w} failed: {resize_err}. Using original size for detection.")
                        frame_for_det = frame_bgr # Fallback to original
                        scale_x, scale_y = 1.0, 1.0
                # --- End Resizing ---

                try:
                    # Convert BGR (OpenCV) to RGB (PIL/Torchvision)
                    img_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    # Apply detector-specific transforms (usually includes ToTensor and normalization)
                    input_tensor = self.detector_transforms(img_pil)

                    batch_input_tensors.append(input_tensor.to(self.device)) # Move tensor to device
                    batch_cam_ids.append(cam_id)
                    batch_original_shapes.append((original_h, original_w))
                    batch_scale_factors.append((scale_x, scale_y))
                except Exception as transform_err:
                    logger.error(f"[{cam_id}] Preprocessing (cvtColor/PIL/Transform) failed: {transform_err}", exc_info=False)
            # else: # Optional log for missing frames in the input batch
                # logger.debug(f"[{cam_id}] Frame data is None or empty in input batch.")

        timings['preprocess'] = time.time() - t_prep_start

        # --- Stage 1b: Batched Detection ---
        t_detect_start = time.time()
        all_predictions: List[Dict[str, torch.Tensor]] = []
        if batch_input_tensors: # Only run detection if there are valid inputs
            try:
                with torch.no_grad():
                    # Check if AMP is enabled and device is CUDA
                    use_amp_runtime = self.config.use_amp and self.device.type == 'cuda'
                    with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                        # Detector expects a list of tensors
                        all_predictions = self.detector(batch_input_tensors)
            except Exception as e:
                logger.error(f"Object detection failed: {e}", exc_info=False) # Log error briefly
                all_predictions = [] # Ensure it's an empty list on failure
        timings['detection_batched'] = time.time() - t_detect_start

        # --- Stage 1c: Postprocess Detections (Filter, Scale) ---
        t_postproc_start = time.time()
        detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)
        # Check if the number of predictions matches the number of inputs processed
        if len(all_predictions) == len(batch_cam_ids):
            for i, prediction_dict in enumerate(all_predictions):
                cam_id = batch_cam_ids[i]
                original_h, original_w = batch_original_shapes[i]
                scale_x, scale_y = batch_scale_factors[i]

                try:
                    # Move predictions to CPU and convert to numpy for easier handling
                    pred_boxes = prediction_dict['boxes'].cpu().numpy()
                    pred_labels = prediction_dict['labels'].cpu().numpy()
                    pred_scores = prediction_dict['scores'].cpu().numpy()

                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        # Filter by person class ID and confidence threshold
                        if label == self.config.person_class_id and score >= self.config.detection_confidence_threshold:
                            # Scale box coordinates back to original frame dimensions
                            x1, y1, x2, y2 = box
                            orig_x1 = max(0.0, x1 * scale_x)
                            orig_y1 = max(0.0, y1 * scale_y)
                            orig_x2 = min(float(original_w - 1), x2 * scale_x) # Clamp to frame boundary
                            orig_y2 = min(float(original_h - 1), y2 * scale_y) # Clamp to frame boundary

                            # Create bounding box in xyxy format for the tracker
                            bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32)

                            # Basic check for valid box dimensions after scaling/clamping
                            if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1: # Require min width/height
                                detections_per_camera[cam_id].append({
                                    'bbox_xyxy': bbox_orig,
                                    'conf': float(score),
                                    'class_id': int(label) # Store class ID just in case
                                })
                except Exception as postproc_err:
                    logger.error(f"[{cam_id}] Error postprocessing detections: {postproc_err}", exc_info=False)
        elif batch_input_tensors: # Log error only if we had inputs but mismatching outputs
            logger.error(f"Detection output count ({len(all_predictions)}) does not match input batch size ({len(batch_cam_ids)}). Skipping detection postprocessing for this batch.")

        timings['postprocess_scale'] = time.time() - t_postproc_start

        # --- Stage 1d: Tracking per Camera ---
        t_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {} # Store raw tracker output
        current_frame_active_track_keys: Set[TrackKey] = set() # Track keys active in this frame
        tracks_to_extract_features_for: Dict[CameraID, List[np.ndarray]] = defaultdict(list) # Tracks triggering ReID

        for cam_id in self.camera_ids: # Iterate through all configured cameras
            tracker = self.trackers.get(cam_id)
            if not tracker:
                logger.warning(f"[{cam_id}] Tracker instance not found. Skipping tracking.")
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8)) # Placeholder empty array
                continue

            # Prepare detections in the format expected by the tracker (usually N x [x1, y1, x2, y2, conf, class_id])
            cam_detections = detections_per_camera.get(cam_id, [])
            np_dets = np.empty((0, 6)) # Default empty array
            if cam_detections:
                try:
                    # Create numpy array [N, 6] -> [x1, y1, x2, y2, conf, cls_id]
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                except Exception as format_err:
                    logger.error(f"[{cam_id}] Failed to format detections for tracker input: {format_err}", exc_info=False)
                    np_dets = np.empty((0, 6)) # Ensure it's empty on error

            # Get the original frame (or a dummy if missing) required by some trackers
            original_frame_bgr = frames.get(cam_id)
            # Try to get original shape if available, otherwise use a default
            frame_shape_for_tracker = next((shape for cid, shape in zip(batch_cam_ids, batch_original_shapes) if cid == cam_id), (1080, 1920)) # Default fallback shape

            # Provide a black frame if the original is missing, as some trackers require an image input
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*frame_shape_for_tracker, 3), dtype=np.uint8)

            # Update the tracker for this camera
            try:
                # Tracker update call
                tracked_dets_list = tracker.update(np_dets, dummy_frame)

                # Convert tracker output to numpy array [M, 8] -> [x1, y1, x2, y2, track_id, conf, cls_id, Optional<idx>]
                # Handle None or empty list output from tracker
                if tracked_dets_list is not None and len(tracked_dets_list) > 0:
                     tracked_dets_np = np.array(tracked_dets_list)
                     # Basic sanity check on output shape (expecting at least 7 columns usually)
                     if tracked_dets_np.ndim != 2 or tracked_dets_np.shape[1] < 7:
                          logger.warning(f"[{cam_id}] Tracker output has unexpected shape {tracked_dets_np.shape}. Expected [M, >=7]. Processing as empty.")
                          tracked_dets_np = np.empty((0, 8)) # Treat as empty if shape is wrong
                else:
                    tracked_dets_np = np.empty((0, 8)) # Ensure consistent empty array format

                current_frame_tracker_outputs[cam_id] = tracked_dets_np

            except Exception as e:
                logger.error(f"[{cam_id}] Tracker update failed: {e}", exc_info=True) # Log with traceback for tracker errors
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8)) # Ensure output is empty on failure

            # --- Identify Active Tracks and Determine Re-ID Triggers ---
            if current_frame_tracker_outputs[cam_id].shape[0] > 0:
                # Get track IDs seen in the *previous* processed frame for this camera
                previous_processed_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())

                for track_data in current_frame_tracker_outputs[cam_id]:
                    # Ensure track_data has enough elements (at least up to track_id)
                    if len(track_data) >= 5:
                        try:
                            track_id = int(track_data[4]) # Track ID is usually the 5th element (index 4)
                        except (ValueError, IndexError):
                            logger.warning(f"[{cam_id}] Could not parse track ID from tracker output: {track_data}")
                            continue # Skip this track if ID is invalid

                        current_track_key: TrackKey = (cam_id, track_id)
                        current_frame_active_track_keys.add(current_track_key) # Mark this track as active

                        # Check if ReID should be triggered for this track
                        if original_frame_bgr is not None and original_frame_bgr.size > 0: # Need frame for feature extraction
                            # Trigger 1: Is this track newly seen since the last frame *we processed*?
                            is_newly_seen_since_last_proc = track_id not in previous_processed_cam_track_ids

                            # Trigger 2: Is it time to refresh the ReID embedding for this track?
                            last_reid_attempt_proc_idx = self.track_last_reid_frame.get(
                                current_track_key,
                                -self.config.reid_refresh_interval_frames - 1 # Default to ensure first check passes
                            )
                            # Check against the current *processed* frame counter
                            is_due_for_refresh = (proc_frame_id - last_reid_attempt_proc_idx) >= self.config.reid_refresh_interval_frames

                            # Trigger ReID if either condition is met
                            trigger_reid = is_newly_seen_since_last_proc or is_due_for_refresh

                            if trigger_reid:
                                # Add the full track data (needed for bbox) to the list for feature extraction
                                tracks_to_extract_features_for[cam_id].append(track_data)
                                # Record that we are attempting ReID in this processed frame
                                self.track_last_reid_frame[current_track_key] = proc_frame_id

        timings['tracking'] = time.time() - t_track_start

        # --- Stage 2: Conditional Feature Extraction (Batched per Camera) ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        if tracks_to_extract_features_for: # Only run if any tracks triggered ReID
            for cam_id, tracks_data_list in tracks_to_extract_features_for.items():
                if tracks_data_list: # If there are tracks for this camera
                    frame_bgr = frames.get(cam_id)
                    if frame_bgr is not None and frame_bgr.size > 0:
                        try:
                            # Convert list of track data arrays to a single numpy array for batch extraction
                            tracks_data_np = np.array(tracks_data_list)
                            # Call the internal feature extraction method
                            features_this_cam = self._extract_features_for_tracks(frame_bgr, tracks_data_np)
                            # Map the extracted features back to their TrackKey
                            for track_id, feature in features_this_cam.items():
                                extracted_features_this_frame[(cam_id, track_id)] = feature
                        except Exception as fe_err:
                            logger.error(f"[{cam_id}] Error during batched feature extraction call: {fe_err}", exc_info=False)
                    # else: # Log if frame is missing for a camera needing feature extraction
                       # logger.warning(f"[{cam_id}] Frame missing, cannot extract features for {len(tracks_data_list)} tracks.")
        timings['feature_ext'] = time.time() - t_feat_start

        # --- Stage 3: Re-ID Association ---
        t_reid_start = time.time()
        # Perform association using the features extracted in this frame
        assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        # --- Stage 4: Combine Results and Finalize Track Data ---
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    # Expecting at least [x1, y1, x2, y2, track_id, conf, class_id] (7 elements)
                    if len(track_data) >= 7:
                        try:
                            # Parse data from the tracker output array
                            x1, y1, x2, y2 = map(float, track_data[0:4])
                            track_id = int(track_data[4])
                            conf = float(track_data[5]) # Tracker confidence
                            cls = int(track_data[6]) # Class ID from tracker

                            current_track_key: TrackKey = (cam_id, track_id)
                            global_id: Optional[GlobalID]

                            # Determine the Global ID for this track
                            # Priority: Use the ID assigned in *this* ReID cycle if available
                            if current_track_key in assigned_global_ids_this_cycle:
                                global_id = assigned_global_ids_this_cycle[current_track_key]
                            else:
                                # Otherwise, use the last known mapping (could be from a previous frame)
                                global_id = self.track_to_global_id.get(current_track_key) # Returns None if not found

                            # Append the final structured data for this track
                            final_results_per_camera[cam_id].append({
                                'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                                'track_id': track_id,
                                'global_id': global_id, # This can be None
                                'conf': conf,
                                'class_id': cls
                            })
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"[{cam_id}] Failed to parse final track data from {track_data}: {e}", exc_info=False)
                            continue # Skip this problematic track data entry
                    # else: # Log if tracker output row is too short
                        # logger.warning(f"[{cam_id}] Skipping tracker output row due to insufficient columns: {track_data}")

        # --- Stage 5: Update State and Cleanup ---
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch

        return ProcessedBatchResult(
            results_per_camera=dict(final_results_per_camera), # Convert defaultdict to dict
            timings=dict(timings), # Convert defaultdict to dict
            processed_this_frame=True # Mark that this frame contributed to processing
        )