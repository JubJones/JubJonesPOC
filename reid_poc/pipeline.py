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
    CameraID, TrackID, GlobalID, TrackKey, FeatureVector, BoundingBox, ExitRule,
    Detection, TrackData, FrameData, Timings, ProcessedBatchResult, ScaleFactors,
    HandoffTriggerInfo, QuadrantName, ExitDirection
)
from reid_poc.utils import calculate_cosine_similarity, normalize_embedding, normalize_overlap_set # Use relative import


logger = logging.getLogger(__name__)

class MultiCameraPipeline:
    """Handles multi-camera detection, tracking, Re-ID, and handoff triggering."""

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
        self.camera_ids = config.selected_cameras # Use the validated list from config setup

        # Store pre-initialized components
        self.detector = detector
        self.detector_transforms = detector_transforms
        self.reid_model = reid_model
        self.trackers = trackers

        # --- Logging config details (INFO level) ---
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids} on device: {self.device.type}")
        # ... (rest of logging remains similar) ...
        if config.min_bbox_overlap_ratio_in_quadrant > 0:
             logger.info(f"Handoff Triggering ENABLED (Quadrant Overlap Ratio: {config.min_bbox_overlap_ratio_in_quadrant:.2f})")
        else:
             logger.info("Handoff Triggering DISABLED (Overlap Ratio <= 0)")


        # --- State Initialization ---
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {} # Stores representative feature for each GlobalID
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {} # Maps (cam_id, track_id) -> global_id
        self.next_global_id: GlobalID = 1 # Counter for assigning new global IDs
        # Stores the set of active track IDs seen in the *last processed* frame for each camera
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        # Stores the processed frame index when ReID was last attempted for a track
        self.track_last_reid_frame: Dict[TrackKey, int] = {}
        # Stores the last camera where a Global ID was seen/updated
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        # Stores handoff triggers detected in the current frame processing cycle
        self.handoff_triggers_this_frame: List[HandoffTriggerInfo] = []
        # Counts only the frames actively processed by the pipeline (respecting frame skipping)
        self.processed_frame_counter: int = 0

        # Pre-normalize overlap sets for faster lookup later
        self.possible_overlaps_normalized = normalize_overlap_set(config.possible_overlaps)


    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        """Extracts Re-ID features for the given tracks using the provided frame."""
        # --- Function remains the same as original ---
        features: Dict[TrackID, FeatureVector] = {}
        if self.reid_model is None:
            logger.warning("ReID model not available, cannot extract features.")
            return features
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Cannot extract features: Invalid frame provided.")
            return features
        if tracked_dets_np.shape[0] == 0:
            return features # No tracks to extract features for

        if tracked_dets_np.shape[1] < 5:
             logger.warning(f"Track data has unexpected shape {tracked_dets_np.shape}, expected at least 5 columns (xyxy, id). Skipping feature extraction.")
             return features

        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids_float = tracked_dets_np[:, 4]

        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4:
            logger.warning(f"Invalid bbox shape {bboxes_xyxy.shape} received for feature extraction. Skipping.")
            return features

        try:
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)

            if batch_features is not None and len(batch_features) == len(track_ids_float):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        track_id = int(track_ids_float[i])
                        features[track_id] = det_feature
            # else:
                # logger.warning(f"Feature extraction output count mismatch or None.")

        except Exception as e:
            logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features


    def _check_handoff_triggers(
        self,
        cam_id: CameraID,
        tracked_dets_np: np.ndarray,
        frame_shape: Optional[Tuple[int, int]]
    ):
        """
        Checks if any tracks in the current camera trigger predefined handoff rules
        based on quadrant overlap. Appends triggers to self.handoff_triggers_this_frame.
        """
        cam_handoff_cfg = self.config.cameras_handoff_config.get(cam_id)
        # Check prerequisites for handoff calculation
        if (not cam_handoff_cfg or
            not cam_handoff_cfg.exit_rules or
            not frame_shape or frame_shape[0] <= 0 or frame_shape[1] <= 0 or
            tracked_dets_np.shape[0] == 0 or
            self.config.min_bbox_overlap_ratio_in_quadrant <= 0):
            return # Cannot perform handoff check

        H, W = frame_shape
        mid_x, mid_y = W // 2, H // 2

        # Define quadrant regions (x1, y1, x2, y2)
        quadrant_regions: Dict[QuadrantName, Tuple[int, int, int, int]] = {
            'upper_left': (0, 0, mid_x, mid_y), 'upper_right': (mid_x, 0, W, mid_y),
            'lower_left': (0, mid_y, mid_x, H), 'lower_right': (mid_x, mid_y, W, H),
        }

        # Map exit directions to relevant quadrants
        direction_to_quadrants: Dict[ExitDirection, List[QuadrantName]] = {
            'up': ['upper_left', 'upper_right'],
            'down': ['lower_left', 'lower_right'],
            'left': ['upper_left', 'lower_left'],
            'right': ['upper_right', 'lower_right'],
        }

        processed_track_ids = set() # Avoid triggering multiple rules for the same track

        for rule in cam_handoff_cfg.exit_rules:
            relevant_quadrant_names = direction_to_quadrants.get(rule.direction, [])
            if not relevant_quadrant_names:
                logger.warning(f"[{cam_id}] Skipping rule with invalid direction: {rule.direction}")
                continue

            # Get the region(s) associated with this rule's exit direction
            exit_regions_coords = [quadrant_regions[name] for name in relevant_quadrant_names if name in quadrant_regions]
            if not exit_regions_coords:
                continue # Should not happen with valid directions

            # Check each track against this rule
            for track_data in tracked_dets_np:
                 # Expecting at least [x1, y1, x2, y2, track_id, ...]
                if len(track_data) < 5: continue
                try:
                    track_id = int(track_data[4])
                except (ValueError, IndexError):
                    continue # Invalid track ID format

                if track_id in processed_track_ids:
                    continue # Already triggered a rule for this track

                bbox = track_data[0:4].astype(np.float32)
                x1, y1, x2, y2 = map(int, bbox)
                bbox_w, bbox_h = x2 - x1, y2 - y1
                if bbox_w <= 0 or bbox_h <= 0: continue
                bbox_area = float(bbox_w * bbox_h)

                total_intersection_area = 0.0
                for qx1, qy1, qx2, qy2 in exit_regions_coords:
                    inter_x1, inter_y1 = max(x1, qx1), max(y1, qy1)
                    inter_x2, inter_y2 = min(x2, qx2), min(y2, qy2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    total_intersection_area += float(inter_w * inter_h)

                # Check overlap ratio against threshold
                if bbox_area > 1e-5 and (total_intersection_area / bbox_area) >= self.config.min_bbox_overlap_ratio_in_quadrant:
                    source_track_key: TrackKey = (cam_id, track_id)
                    trigger_info = HandoffTriggerInfo(
                        source_track_key=source_track_key,
                        rule=rule,
                        source_bbox=bbox
                    )
                    self.handoff_triggers_this_frame.append(trigger_info)
                    processed_track_ids.add(track_id) # Mark as processed for this frame

                    # Log the trigger event
                    logger.info(
                        f"HANDOFF TRIGGER: Track {source_track_key} at {bbox.astype(int)} "
                        f"matched rule '{rule.direction}' -> Cam [{rule.target_cam_id}] "
                        f"Area [{rule.target_entry_area}]."
                    )
                    # Break inner loop (track loop) once a track matches a rule
                    break


    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        """Gets the target camera and any possibly overlapping cameras for handoff."""
        relevant_cams = {target_cam_id}
        # Check normalized overlap set
        for c1, c2 in self.possible_overlaps_normalized:
            if c1 == target_cam_id:
                relevant_cams.add(c2)
            elif c2 == target_cam_id:
                relevant_cams.add(c1)
        return relevant_cams


    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[TrackKey, Optional[GlobalID]]:
        """Associates tracks with Global IDs, prioritizing handoff targets."""
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {}
        if not features_per_track:
            return newly_assigned_global_ids

        # --- Prepare Full Gallery for Fallback Comparison ---
        valid_gallery_items = [
            (gid, emb) for gid, emb in self.reid_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        full_gallery_ids: List[GlobalID] = []
        full_gallery_embeddings: List[FeatureVector] = []
        if valid_gallery_items:
            full_gallery_ids, full_gallery_embeddings = zip(*valid_gallery_items)


        # --- Find active handoff triggers for tracks needing ReID ---
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo] = {
             trigger.source_track_key: trigger
             for trigger in self.handoff_triggers_this_frame
             if trigger.source_track_key in features_per_track # Only consider tracks we have features for
        }


        # --- Iterate Through New Features ---
        for track_key, new_embedding_raw in features_per_track.items():
            cam_id, track_id = track_key
            newly_assigned_global_ids[track_key] = None # Default

            if new_embedding_raw is None or not np.isfinite(new_embedding_raw).all() or new_embedding_raw.size == 0:
                logger.warning(f"Skipping ReID for {track_key}: Received invalid embedding.")
                continue

            normalized_new_embedding = normalize_embedding(new_embedding_raw)

            best_match_global_id: Optional[GlobalID] = None
            max_similarity = -1.0

            # --- Check for Handoff Trigger ---
            triggered_handoff = active_triggers_map.get(track_key)
            gallery_ids_to_check = full_gallery_ids
            gallery_embeddings_to_check = full_gallery_embeddings

            if triggered_handoff:
                target_cam_id = triggered_handoff.rule.target_cam_id
                relevant_cams = self._get_relevant_handoff_cams(target_cam_id)
                logger.debug(f"Track {track_key} triggered handoff. Prioritizing check against cams: {relevant_cams}")

                # Filter gallery based on relevant cameras
                filtered_gallery_indices = [
                    idx for idx, gid in enumerate(full_gallery_ids)
                    if self.global_id_last_seen_cam.get(gid) in relevant_cams
                ]

                if filtered_gallery_indices:
                    gallery_ids_to_check = [full_gallery_ids[i] for i in filtered_gallery_indices]
                    gallery_embeddings_to_check = [full_gallery_embeddings[i] for i in filtered_gallery_indices]
                    logger.debug(f"  -> Checking against {len(gallery_ids_to_check)} filtered gallery entries.")
                else:
                    # If no relevant entries, fallback to full gallery check is needed later
                    logger.debug(f"  -> No relevant gallery entries found for cams {relevant_cams}. Will check full gallery.")
                    gallery_ids_to_check = [] # Prevent comparison in the next block for now

            # --- Compare with Selected Gallery (Filtered or Full) ---
            if gallery_ids_to_check: # Only compare if we have candidates
                try:
                    similarities = np.array([
                        calculate_cosine_similarity(normalized_new_embedding, gal_emb)
                        for gal_emb in gallery_embeddings_to_check
                    ])

                    if similarities.size > 0:
                        max_similarity_idx = np.argmax(similarities)
                        current_max_similarity = similarities[max_similarity_idx]

                        if current_max_similarity >= self.config.reid_similarity_threshold:
                             # Found a match within the (potentially filtered) set
                            best_match_global_id = gallery_ids_to_check[max_similarity_idx]
                            max_similarity = current_max_similarity
                            logger.debug(f"  -> Match found in {'filtered' if triggered_handoff else 'full'} gallery: GID {best_match_global_id} (Sim: {max_similarity:.3f})")

                except Exception as sim_err:
                    logger.error(f"Similarity calculation error during ReID for {track_key}: {sim_err}", exc_info=False)


            # --- Fallback: If handoff was triggered but no match found in filtered set, check full gallery ---
            if triggered_handoff and best_match_global_id is None and full_gallery_ids:
                 logger.debug(f"  -> Handoff triggered, but no match in filtered set. Checking full gallery...")
                 try:
                    similarities_full = np.array([
                        calculate_cosine_similarity(normalized_new_embedding, gal_emb)
                        for gal_emb in full_gallery_embeddings # Compare against ALL valid embeddings
                    ])
                    if similarities_full.size > 0:
                        max_similarity_idx_full = np.argmax(similarities_full)
                        current_max_similarity_full = similarities_full[max_similarity_idx_full]

                        if current_max_similarity_full >= self.config.reid_similarity_threshold:
                            best_match_global_id = full_gallery_ids[max_similarity_idx_full]
                            max_similarity = current_max_similarity_full
                            logger.debug(f"  -> Match found in FULL gallery fallback: GID {best_match_global_id} (Sim: {max_similarity:.3f})")

                 except Exception as sim_err_full:
                    logger.error(f"Similarity calculation error during FULL ReID fallback for {track_key}: {sim_err_full}", exc_info=False)


            # --- Assign Global ID and Update Gallery ---
            assigned_global_id: Optional[GlobalID] = None

            if best_match_global_id is not None:
                # Case A: Found a match above threshold (either filtered or full)
                assigned_global_id = best_match_global_id
                current_gallery_emb = self.reid_gallery.get(assigned_global_id)

                if current_gallery_emb is not None:
                    updated_embedding = (
                        self.config.gallery_ema_alpha * current_gallery_emb +
                        (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding # Use the normalized one
                    )
                    self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)
                else:
                    logger.warning(f"Track {track_key}: Matched GID {assigned_global_id} was None in gallery? Overwriting.")
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding

            else:
                # Case B: No match found above threshold even after potential fallback
                last_known_global_id = self.track_to_global_id.get(track_key)

                if last_known_global_id is not None and last_known_global_id in self.reid_gallery:
                    # Case B.1: Re-assign previous ID and update gallery
                    assigned_global_id = last_known_global_id
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (
                            self.config.gallery_ema_alpha * current_gallery_emb +
                            (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding
                        )
                        self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)
                    else:
                        self.reid_gallery[assigned_global_id] = normalized_new_embedding # Overwrite if missing
                else:
                    # Case B.2: Assign a new Global ID
                    assigned_global_id = self.next_global_id
                    self.next_global_id += 1
                    self.reid_gallery[assigned_global_id] = normalized_new_embedding
                    logger.info(f"Assigned NEW Global ID {assigned_global_id} to {track_key}")


            # --- Update State Mappings ---
            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id
                # Update the last seen camera for this global ID
                self.global_id_last_seen_cam[assigned_global_id] = cam_id

        return newly_assigned_global_ids


    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """Updates the set of last seen tracks and cleans up state for disappeared tracks."""
        # 1. Update last_seen_track_ids
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys:
            new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        # 2. Clean up track_last_reid_frame
        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_reid:
            del self.track_last_reid_frame[key]

        # 3. Clean up track_to_global_id mapping
        keys_to_delete_global = set(self.track_to_global_id.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_global:
            del self.track_to_global_id[key]

        # 4. Optional: Cleanup global_id_last_seen_cam for GIDs not seen recently?
        #    For simplicity, we don't do this now. It might remove relevant history for handoffs.

        # 5. Cleanup handoff triggers from the *previous* frame (done at start of next process call)


    def process_frame_batch_full(self, frames: Dict[CameraID, FrameData], frame_idx_global: int) -> ProcessedBatchResult:
        """Processes a batch of frames: Detect -> Track -> Handoff Check -> Re-ID -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)

        # Clear triggers from the previous frame processing cycle
        self.handoff_triggers_this_frame.clear()

        # Increment counter only for frames actually processed (respecting skipping)
        self.processed_frame_counter += 1
        proc_frame_id = self.processed_frame_counter

        # --- Stage 1a: Preprocess Frames for Detection ---
        t_prep_start = time.time()
        batch_input_tensors: List[torch.Tensor] = []
        batch_cam_ids: List[CameraID] = [] # Keep track of order
        batch_original_shapes: List[Tuple[int, int]] = [] # (height, width)
        batch_scale_factors: List[ScaleFactors] = [] # (scale_x, scale_y)

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
                        frame_for_det = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        scale_x = original_w / target_w
                        scale_y = original_h / target_h
                    except Exception as resize_err:
                        logger.warning(f"[{cam_id}] Resizing failed: {resize_err}. Using original.")
                        frame_for_det = frame_bgr # Fallback
                        scale_x, scale_y = 1.0, 1.0

                try:
                    img_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    input_tensor = self.detector_transforms(img_pil)

                    batch_input_tensors.append(input_tensor.to(self.device))
                    batch_cam_ids.append(cam_id)
                    # Store the *original* shape, important for handoff quadrant calc
                    batch_original_shapes.append((original_h, original_w))
                    batch_scale_factors.append((scale_x, scale_y))
                except Exception as transform_err:
                    logger.error(f"[{cam_id}] Preprocessing failed: {transform_err}", exc_info=False)

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
                logger.error(f"Object detection failed: {e}", exc_info=False)
                all_predictions = []
        timings['detection_batched'] = time.time() - t_detect_start

        # --- Stage 1c: Postprocess Detections (Filter, Scale) ---
        # --- Function remains the same as original ---
        t_postproc_start = time.time()
        detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)
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
                        if label == self.config.person_class_id and score >= self.config.detection_confidence_threshold:
                            x1, y1, x2, y2 = box
                            orig_x1 = max(0.0, x1 * scale_x)
                            orig_y1 = max(0.0, y1 * scale_y)
                            orig_x2 = min(float(original_w - 1), x2 * scale_x)
                            orig_y2 = min(float(original_h - 1), y2 * scale_y)

                            if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1:
                                bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32)
                                detections_per_camera[cam_id].append({
                                    'bbox_xyxy': bbox_orig, 'conf': float(score), 'class_id': int(label)
                                })
                except Exception as postproc_err:
                    logger.error(f"[{cam_id}] Error postprocessing detections: {postproc_err}", exc_info=False)
        elif batch_input_tensors:
            logger.error(f"Detection output count mismatch: {len(all_predictions)} vs {len(batch_cam_ids)}")
        timings['postprocess_scale'] = time.time() - t_postproc_start


        # --- Stage 1d: Tracking per Camera ---
        t_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        current_frame_active_track_keys: Set[TrackKey] = set()
        tracks_to_extract_features_for: Dict[CameraID, List[np.ndarray]] = defaultdict(list)
        # Store original shapes mapped by cam_id for handoff check
        original_shapes_map: Dict[CameraID, Tuple[int, int]] = {
            cam_id: shape for cam_id, shape in zip(batch_cam_ids, batch_original_shapes)
        }

        for cam_id in self.camera_ids: # Iterate all configured cameras
            tracker = self.trackers.get(cam_id)
            if not tracker:
                logger.warning(f"[{cam_id}] Tracker instance missing.")
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8))
                continue

            cam_detections = detections_per_camera.get(cam_id, [])
            np_dets = np.empty((0, 6))
            if cam_detections:
                try:
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                except Exception as format_err:
                    logger.error(f"[{cam_id}] Failed to format detections for tracker: {format_err}", exc_info=False)

            original_frame_bgr = frames.get(cam_id)
            # Get the original shape for this camera, needed for tracker AND handoff
            frame_shape_orig = original_shapes_map.get(cam_id)
            # Provide a black frame if original is missing (some trackers need it)
            dummy_frame_shape = frame_shape_orig if frame_shape_orig else (1080, 1920) # Fallback shape
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*dummy_frame_shape, 3), dtype=np.uint8)

            tracked_dets_np = np.empty((0, 8)) # Default empty
            try:
                tracked_dets_list = tracker.update(np_dets, dummy_frame)
                if tracked_dets_list is not None and len(tracked_dets_list) > 0:
                     tracked_dets_np_maybe = np.array(tracked_dets_list)
                     if tracked_dets_np_maybe.ndim == 2 and tracked_dets_np_maybe.shape[1] >= 7:
                          tracked_dets_np = tracked_dets_np_maybe
                     else:
                          logger.warning(f"[{cam_id}] Tracker output has unexpected shape {tracked_dets_np_maybe.shape}. Treating as empty.")
            except Exception as e:
                logger.error(f"[{cam_id}] Tracker update failed: {e}", exc_info=True)

            current_frame_tracker_outputs[cam_id] = tracked_dets_np

            # --- Identify Active Tracks & Re-ID Triggers (based on processed frame counter) ---
            if tracked_dets_np.shape[0] > 0:
                previous_processed_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in tracked_dets_np:
                    if len(track_data) >= 5:
                        try: track_id = int(track_data[4])
                        except (ValueError, IndexError): continue

                        current_track_key: TrackKey = (cam_id, track_id)
                        current_frame_active_track_keys.add(current_track_key)

                        if original_frame_bgr is not None and original_frame_bgr.size > 0:
                            is_newly_seen_since_last_proc = track_id not in previous_processed_cam_track_ids
                            last_reid_attempt_proc_idx = self.track_last_reid_frame.get(
                                current_track_key, -self.config.reid_refresh_interval_frames - 1
                            )
                            is_due_for_refresh = (proc_frame_id - last_reid_attempt_proc_idx) >= self.config.reid_refresh_interval_frames
                            trigger_reid = is_newly_seen_since_last_proc or is_due_for_refresh

                            if trigger_reid:
                                tracks_to_extract_features_for[cam_id].append(track_data)
                                self.track_last_reid_frame[current_track_key] = proc_frame_id

        timings['tracking'] = time.time() - t_track_start

        # --- Stage 1e: Handoff Trigger Check (After Tracking) ---
        t_handoff_start = time.time()
        if self.config.min_bbox_overlap_ratio_in_quadrant > 0:
            for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
                 frame_shape_orig = original_shapes_map.get(cam_id)
                 self._check_handoff_triggers(cam_id, tracked_dets_np, frame_shape_orig)
        timings['handoff_check'] = time.time() - t_handoff_start


        # --- Stage 2: Conditional Feature Extraction ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        if tracks_to_extract_features_for:
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
                            logger.error(f"[{cam_id}] Error during batched feature extraction call: {fe_err}", exc_info=False)
        timings['feature_ext'] = time.time() - t_feat_start

        # --- Stage 3: Re-ID Association (Handoff-Aware) ---
        t_reid_start = time.time()
        assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start

        # --- Stage 4: Combine Results and Finalize Track Data ---
        # --- Function remains the same as original ---
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    if len(track_data) >= 7: # Need x1,y1,x2,y2,tid,conf,cls
                        try:
                            x1, y1, x2, y2 = map(float, track_data[0:4])
                            track_id = int(track_data[4])
                            conf = float(track_data[5])
                            cls = int(track_data[6])
                            current_track_key: TrackKey = (cam_id, track_id)
                            global_id: Optional[GlobalID]
                            # Priority: Use ID assigned in this ReID cycle
                            if current_track_key in assigned_global_ids_this_cycle:
                                global_id = assigned_global_ids_this_cycle[current_track_key]
                            else:
                                # Fallback: Use last known mapping
                                global_id = self.track_to_global_id.get(current_track_key)

                            final_results_per_camera[cam_id].append({
                                'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                                'track_id': track_id, 'global_id': global_id,
                                'conf': conf, 'class_id': cls
                            })
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"[{cam_id}] Failed to parse final track data from {track_data}: {e}", exc_info=False)

        # --- Stage 5: Update State and Cleanup ---
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch

        return ProcessedBatchResult(
            results_per_camera=dict(final_results_per_camera),
            timings=dict(timings),
            processed_this_frame=True,
            # Pass the triggers detected in this frame for visualization
            handoff_triggers=list(self.handoff_triggers_this_frame)
        )