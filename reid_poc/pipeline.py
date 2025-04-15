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
        logger.info(f" - Scene: {config.selected_scene}")
        logger.info(f" - Detector Confidence: {config.detection_confidence_threshold}")
        logger.info(f" - ReID Similarity Threshold: {config.reid_similarity_threshold}")
        logger.info(f" - ReID Refresh Interval (Processed Frames): {config.reid_refresh_interval_frames}")
        logger.info(f" - Frame Skip Rate: {config.frame_skip_rate}")
        logger.info(f" - Tracker Type: {config.tracker_type}")
        if config.min_bbox_overlap_ratio_in_quadrant > 0:
            logger.info(f"Handoff Triggering ENABLED (Quadrant Overlap Ratio: {config.min_bbox_overlap_ratio_in_quadrant:.2f})")
        else:
            logger.info("Handoff Triggering DISABLED (Overlap Ratio <= 0)")
        logger.info(f"Possible Overlaps: {config.possible_overlaps}") # Log original for clarity
        logger.info(f"No Overlaps: {config.no_overlaps}") # Log original for clarity
        logger.info(f"Lost Track Buffer: {config.lost_track_buffer_frames} frames") # Log new config

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
        # --- NEW: Store info about recently lost tracks ---
        # Maps GlobalID -> (last_feature_vector, disappearance_frame_number)
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {}

        # Pre-normalize overlap sets for faster lookup later
        self.possible_overlaps_normalized = normalize_overlap_set(config.possible_overlaps)


    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        """Extracts Re-ID features for the given tracks using the provided frame."""
        # --- Function remains the same ---
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
            #     logger.warning(f"Feature extraction output count mismatch or None.")

        except Exception as e:
            logger.error(f"Feature extraction call failed: {e}", exc_info=False) # Reduce noise
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
        # --- Function remains the same ---
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

        processed_track_ids = set() # Avoid triggering multiple rules for the same track in this cycle

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
                    # Don't break inner loop here, a track might match multiple rules if configured oddly,
                    # but we only add it once per frame due to processed_track_ids set.


    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        """Gets the target camera and any possibly overlapping cameras for handoff."""
        # --- Function remains the same ---
        relevant_cams = {target_cam_id}
        # Check normalized overlap set
        for c1, c2 in self.possible_overlaps_normalized:
            if c1 == target_cam_id:
                relevant_cams.add(c2)
            elif c2 == target_cam_id:
                relevant_cams.add(c1)
        return relevant_cams


    def _perform_reid_association(self, features_per_track: Dict[TrackKey, FeatureVector]) -> Dict[TrackKey, Optional[GlobalID]]:
        """
        Associates tracks with Global IDs, prioritizing checks against recently lost tracks,
        then handoff targets, then the full gallery.
        """
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {}
        if not features_per_track:
            return newly_assigned_global_ids

        # --- Prepare Full Main Gallery for Fallback Comparison ---
        valid_main_gallery_items = [
            (gid, emb) for gid, emb in self.reid_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        full_main_gallery_ids: List[GlobalID] = []
        full_main_gallery_embeddings: List[FeatureVector] = []
        if valid_main_gallery_items:
            full_main_gallery_ids, full_main_gallery_embeddings = zip(*valid_main_gallery_items)

        # --- Prepare Lost Track Gallery ---
        valid_lost_gallery_items = [
            (gid, (emb, frame_num)) for gid, (emb, frame_num) in self.lost_track_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        lost_gallery_ids: List[GlobalID] = []
        lost_gallery_embeddings: List[FeatureVector] = []
        if valid_lost_gallery_items:
            # Sort by disappearance frame number (most recent first) - might help slightly? Optional.
            # valid_lost_gallery_items.sort(key=lambda item: item[1][1], reverse=True)
            lost_gallery_ids, lost_gallery_data = zip(*valid_lost_gallery_items)
            lost_gallery_embeddings = [data[0] for data in lost_gallery_data]


        # --- Find active handoff triggers for tracks needing ReID ---
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo] = {
            trigger.source_track_key: trigger
            for trigger in self.handoff_triggers_this_frame
            if trigger.source_track_key in features_per_track
        }

        # --- Iterate Through New Features ---
        for track_key, new_embedding_raw in features_per_track.items():
            cam_id, track_id = track_key
            newly_assigned_global_ids[track_key] = None # Default

            if new_embedding_raw is None or not np.isfinite(new_embedding_raw).all() or new_embedding_raw.size == 0:
                logger.warning(f"Skipping ReID for {track_key}: Received invalid embedding.")
                continue

            normalized_new_embedding = normalize_embedding(new_embedding_raw)
            assigned_global_id: Optional[GlobalID] = None # Track the final assignment for this track_key

            # --- Step 1: Check Lost Track Gallery First ---
            best_lost_match_gid: Optional[GlobalID] = None
            max_lost_similarity = -1.0

            if lost_gallery_ids: # Only check if the lost gallery is not empty
                try:
                    lost_similarities = np.array([
                        calculate_cosine_similarity(normalized_new_embedding, lost_emb)
                        for lost_emb in lost_gallery_embeddings
                    ])
                    if lost_similarities.size > 0:
                        max_lost_similarity_idx = np.argmax(lost_similarities)
                        current_max_lost_similarity = lost_similarities[max_lost_similarity_idx]

                        if current_max_lost_similarity >= self.config.reid_similarity_threshold:
                            best_lost_match_gid = lost_gallery_ids[max_lost_similarity_idx]
                            max_lost_similarity = current_max_lost_similarity
                            logger.info(f"Re-associating {track_key} with LOST GID {best_lost_match_gid} (Sim: {max_lost_similarity:.3f}).")

                            # Assign the matched Global ID
                            assigned_global_id = best_lost_match_gid

                            # Restore to main gallery (and remove from lost)
                            lost_embedding, _ = self.lost_track_gallery.pop(assigned_global_id) # Remove and get data
                            # Update main gallery using EMA with the new embedding
                            updated_embedding = (
                                self.config.gallery_ema_alpha * lost_embedding + # Use the stored lost embedding
                                (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding
                            )
                            self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)

                except Exception as lost_sim_err:
                    logger.error(f"Similarity calculation error during Lost Gallery check for {track_key}: {lost_sim_err}", exc_info=False)


            # --- Step 2: Check Main Gallery (if not re-associated from lost tracks) ---
            if assigned_global_id is None:
                best_main_match_global_id: Optional[GlobalID] = None
                max_main_similarity = -1.0
                triggered_handoff = active_triggers_map.get(track_key)

                # Determine which part of the main gallery to check (potentially filtered by handoff)
                gallery_ids_to_check = full_main_gallery_ids
                gallery_embeddings_to_check = full_main_gallery_embeddings
                is_checking_filtered_gallery = False

                if triggered_handoff:
                    target_cam_id = triggered_handoff.rule.target_cam_id
                    relevant_cams = self._get_relevant_handoff_cams(target_cam_id)
                    logger.debug(f"Track {track_key} triggered handoff. Prioritizing check against main gallery entries from cams: {relevant_cams}")

                    filtered_gallery_indices = [
                        idx for idx, gid in enumerate(full_main_gallery_ids)
                        if self.global_id_last_seen_cam.get(gid) in relevant_cams
                    ]

                    if filtered_gallery_indices:
                        gallery_ids_to_check = [full_main_gallery_ids[i] for i in filtered_gallery_indices]
                        gallery_embeddings_to_check = [full_main_gallery_embeddings[i] for i in filtered_gallery_indices]
                        logger.debug(f"  -> Checking against {len(gallery_ids_to_check)} filtered main gallery entries.")
                        is_checking_filtered_gallery = True
                    else:
                        logger.debug(f"  -> No relevant main gallery entries found for cams {relevant_cams}. Will check full main gallery.")
                        gallery_ids_to_check = [] # Prevent comparison in the next block for now

                # Perform comparison with selected main gallery (filtered or full)
                if gallery_ids_to_check:
                    try:
                        main_similarities = np.array([
                            calculate_cosine_similarity(normalized_new_embedding, gal_emb)
                            for gal_emb in gallery_embeddings_to_check
                        ])
                        if main_similarities.size > 0:
                            max_main_similarity_idx = np.argmax(main_similarities)
                            current_max_main_similarity = main_similarities[max_main_similarity_idx]

                            if current_max_main_similarity >= self.config.reid_similarity_threshold:
                                best_main_match_global_id = gallery_ids_to_check[max_main_similarity_idx]
                                max_main_similarity = current_max_main_similarity
                                logger.debug(f"  -> Match found in {'filtered' if is_checking_filtered_gallery else 'full'} main gallery: GID {best_main_match_global_id} (Sim: {max_main_similarity:.3f})")
                    except Exception as sim_err:
                        logger.error(f"Similarity calculation error during Main Gallery check for {track_key}: {sim_err}", exc_info=False)


                # Fallback: If handoff was triggered but no match found in filtered set, check full main gallery
                if triggered_handoff and is_checking_filtered_gallery and best_main_match_global_id is None and full_main_gallery_ids:
                    logger.debug(f"  -> Handoff triggered, but no match in filtered main set. Checking full main gallery...")
                    try:
                        similarities_full = np.array([
                            calculate_cosine_similarity(normalized_new_embedding, gal_emb)
                            for gal_emb in full_main_gallery_embeddings
                        ])
                        if similarities_full.size > 0:
                            max_similarity_idx_full = np.argmax(similarities_full)
                            current_max_similarity_full = similarities_full[max_similarity_idx_full]

                            if current_max_similarity_full >= self.config.reid_similarity_threshold:
                                best_main_match_global_id = full_main_gallery_ids[max_similarity_idx_full]
                                max_main_similarity = current_max_similarity_full
                                logger.debug(f"  -> Match found in FULL main gallery fallback: GID {best_main_match_global_id} (Sim: {max_main_similarity:.3f})")
                    except Exception as sim_err_full:
                         logger.error(f"Similarity calculation error during FULL main gallery fallback for {track_key}: {sim_err_full}", exc_info=False)


                # Assign ID if a match was found in the main gallery
                if best_main_match_global_id is not None:
                    assigned_global_id = best_main_match_global_id
                    # Update the main gallery embedding using EMA
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (
                            self.config.gallery_ema_alpha * current_gallery_emb +
                            (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding
                        )
                        self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)
                    else: # Should not happen if gallery lists were built correctly, but safeguard
                         logger.warning(f"Track {track_key}: Matched main GID {assigned_global_id} was None in gallery? Overwriting.")
                         self.reid_gallery[assigned_global_id] = normalized_new_embedding


            # --- Step 3: Assign New Global ID (if not matched in Lost or Main gallery) ---
            if assigned_global_id is None:
                assigned_global_id = self.next_global_id
                self.next_global_id += 1
                self.reid_gallery[assigned_global_id] = normalized_new_embedding # Add the new embedding
                logger.info(f"Assigned NEW Global ID {assigned_global_id} to {track_key}")


            # --- Step 4: Update State Mappings ---
            if assigned_global_id is not None:
                newly_assigned_global_ids[track_key] = assigned_global_id
                self.track_to_global_id[track_key] = assigned_global_id
                # Update the last seen camera for this global ID
                self.global_id_last_seen_cam[assigned_global_id] = cam_id

        return newly_assigned_global_ids


    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """
        Updates the set of last seen tracks. Moves disappeared tracks' info to a temporary
        'lost_track_gallery' and purges expired entries from it. Cleans up other state.
        """
        # --- 1. Determine Disappeared Tracks ---
        all_previous_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previous_track_keys - current_frame_active_track_keys

        # --- 2. Move Disappeared Tracks to Lost Gallery ---
        for track_key in disappeared_track_keys:
            global_id = self.track_to_global_id.pop(track_key, None) # Remove and get GID
            if global_id is not None:
                last_feature = self.reid_gallery.get(global_id)
                if last_feature is not None:
                    # Store the last known feature and the frame number it disappeared
                    self.lost_track_gallery[global_id] = (last_feature, self.processed_frame_counter)
                    logger.info(f"Track {track_key} (GID {global_id}) disappeared. Moved info to lost gallery (frame {self.processed_frame_counter}).")
                    # Optional: Remove from main gallery immediately? Or keep for a while?
                    # Let's keep it in the main gallery for now, in case it reappears instantly elsewhere
                    # before the lost check happens (e.g., handoff). The lost check will prioritize.
                else:
                     logger.warning(f"Track {track_key} (GID {global_id}) disappeared but feature not found in main gallery.")
            # Clean up last reid frame entry as well
            if track_key in self.track_last_reid_frame:
                 del self.track_last_reid_frame[track_key]


        # --- 3. Purge Expired Entries from Lost Gallery ---
        expired_lost_gids = [
            gid for gid, (feat, frame_num) in self.lost_track_gallery.items()
            if (self.processed_frame_counter - frame_num) > self.config.lost_track_buffer_frames
        ]
        for gid in expired_lost_gids:
            del self.lost_track_gallery[gid]
            # Optionally remove from main gallery now if it hasn't been updated
            # This requires tracking last *update* time for main gallery entries, adding complexity.
            # For simplicity, let's just remove from lost gallery for now. It might linger in main gallery.
            logger.info(f"Purged expired GID {gid} from lost track gallery.")


        # --- 4. Update last_seen_track_ids for the next cycle ---
        # (Based on tracks active *this* processed frame)
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys:
            new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        # --- 5. Clean up track_last_reid_frame for tracks that *are* active but maybe weren't in the map before ---
        # This ensures only currently active tracks persist in this map.
        # (The disappeared ones were handled in step 2)
        current_reid_keys = set(self.track_last_reid_frame.keys())
        keys_to_delete_reid = current_reid_keys - current_frame_active_track_keys
        for key in keys_to_delete_reid:
             # Check again in case it was already deleted in step 2 (shouldn't hurt)
            if key in self.track_last_reid_frame:
                del self.track_last_reid_frame[key]
                # logger.debug(f"Cleaned up lingering track_last_reid_frame for {key}")


        # 6. Cleanup handoff triggers from the *previous* frame (done at start of next process call)
        # - This remains unchanged, handled by self.handoff_triggers_this_frame.clear()


    def process_frame_batch_full(self, frames: Dict[CameraID, FrameData], frame_idx_global: int) -> ProcessedBatchResult:
        """Processes a batch of frames: Detect -> Track -> Handoff Check -> Re-ID (Lost Check -> Main Check -> New ID) -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)

        # Increment counter only for frames actually processed (respecting skipping)
        self.processed_frame_counter += 1
        proc_frame_id = self.processed_frame_counter # Local alias for clarity

        # --- Stage 1a: Preprocess Frames for Detection ---
        # --- Remains the same ---
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
        # --- Remains the same ---
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
        # --- Remains the same ---
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
        # --- Remains the same ---
        t_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        original_shapes_map: Dict[CameraID, Tuple[int, int]] = {
            cam_id: shape for cam_id, shape in zip(batch_cam_ids, batch_original_shapes)
        }

        for cam_id in self.camera_ids: # Iterate all configured cameras
            tracker = self.trackers.get(cam_id)
            if not tracker:
                logger.warning(f"[{cam_id}] Tracker instance missing.")
                current_frame_tracker_outputs[cam_id] = np.empty((0, 8)) # Assume 8 columns: xyxy, id, conf, cls, ?
                continue

            cam_detections = detections_per_camera.get(cam_id, [])
            np_dets = np.empty((0, 6))
            if cam_detections:
                try:
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                except Exception as format_err:
                    logger.error(f"[{cam_id}] Failed to format detections for tracker: {format_err}", exc_info=False)

            original_frame_bgr = frames.get(cam_id)
            frame_shape_orig = original_shapes_map.get(cam_id)
            dummy_frame_shape = frame_shape_orig if frame_shape_orig else (1080, 1920) # Fallback shape
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*dummy_frame_shape, 3), dtype=np.uint8)

            tracked_dets_np = np.empty((0, 8)) # Default empty
            try:
                tracked_dets_list = tracker.update(np_dets, dummy_frame)
                if tracked_dets_list is not None and len(tracked_dets_list) > 0:
                    tracked_dets_np_maybe = np.array(tracked_dets_list)
                    if tracked_dets_np_maybe.ndim == 2 and tracked_dets_np_maybe.shape[1] >= 7: # BoxMOT usually gives xyxy, id, conf, cls, [index]
                        tracked_dets_np = tracked_dets_np_maybe[:, :8] # Take first 8 columns if more exist
                    else:
                        logger.warning(f"[{cam_id}] Tracker output has unexpected shape {tracked_dets_np_maybe.shape}. Treating as empty.")
            except Exception as e:
                logger.error(f"[{cam_id}] Tracker update failed: {e}", exc_info=True) # Log full trace for tracker errors

            current_frame_tracker_outputs[cam_id] = tracked_dets_np

        timings['tracking'] = time.time() - t_track_start


        # --- Stage 1e: Handoff Trigger Check ---
        # --- Remains the same ---
        t_handoff_start = time.time()
        self.handoff_triggers_this_frame.clear() # Clear triggers from previous batch first
        if self.config.min_bbox_overlap_ratio_in_quadrant > 0:
            for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
                frame_shape_orig = original_shapes_map.get(cam_id)
                self._check_handoff_triggers(cam_id, tracked_dets_np, frame_shape_orig)

        track_keys_triggering_handoff: Set[TrackKey] = {
            trigger.source_track_key for trigger in self.handoff_triggers_this_frame
        }
        timings['handoff_check'] = time.time() - t_handoff_start


        # --- Stage: Decide which tracks need Re-ID Features ---
        # --- Logic remains the same (still based on newly seen, refresh interval, or handoff trigger) ---
        t_reid_decision_start = time.time()
        tracks_to_extract_features_for: Dict[TrackKey, np.ndarray] = {}
        current_frame_active_track_keys: Set[TrackKey] = set()

        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                original_frame_bgr = frames.get(cam_id)
                previous_processed_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())

                for track_data in tracked_dets_np:
                    if len(track_data) < 5: continue
                    try:
                        track_id = int(track_data[4])
                    except (ValueError, IndexError):
                        logger.warning(f"[{cam_id}] Invalid track ID format in track_data: {track_data}")
                        continue

                    current_track_key: TrackKey = (cam_id, track_id)
                    current_frame_active_track_keys.add(current_track_key) # Collect ALL active keys

                    if original_frame_bgr is not None and original_frame_bgr.size > 0:
                        is_newly_seen = track_id not in previous_processed_cam_track_ids
                        last_reid_attempt_proc_idx = self.track_last_reid_frame.get(
                            current_track_key, -self.config.reid_refresh_interval_frames - 1
                        )
                        is_due_for_refresh = (proc_frame_id - last_reid_attempt_proc_idx) >= self.config.reid_refresh_interval_frames
                        is_triggering_handoff_now = current_track_key in track_keys_triggering_handoff

                        # Trigger ReID if new OR refresh OR triggering handoff
                        trigger_reid = is_newly_seen or is_due_for_refresh or is_triggering_handoff_now

                        if trigger_reid:
                            tracks_to_extract_features_for[current_track_key] = track_data
                            self.track_last_reid_frame[current_track_key] = proc_frame_id # Record *attempt*
                            # Logging remains the same
                            if is_triggering_handoff_now: logger.debug(f"ReID triggered for {current_track_key} due to Handoff Rule.")
                            elif is_newly_seen: logger.debug(f"ReID triggered for {current_track_key} as Newly Seen.")
                            elif is_due_for_refresh: logger.debug(f"ReID triggered for {current_track_key} due to Refresh Interval.")

        timings['reid_decision'] = time.time() - t_reid_decision_start


        # --- Stage 2: Conditional Feature Extraction ---
        # --- Remains the same ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        if tracks_to_extract_features_for:
            tracks_grouped_by_cam: Dict[CameraID, List[np.ndarray]] = defaultdict(list)
            track_keys_per_cam: Dict[CameraID, List[TrackKey]] = defaultdict(list)

            for track_key, track_data in tracks_to_extract_features_for.items():
                cam_id, _ = track_key
                tracks_grouped_by_cam[cam_id].append(track_data)
                track_keys_per_cam[cam_id].append(track_key)

            for cam_id, tracks_data_list in tracks_grouped_by_cam.items():
                frame_bgr = frames.get(cam_id)
                if frame_bgr is not None and frame_bgr.size > 0 and tracks_data_list:
                    try:
                        tracks_data_np = np.array(tracks_data_list)
                        features_this_cam: Dict[TrackID, FeatureVector] = self._extract_features_for_tracks(frame_bgr, tracks_data_np)

                        original_keys_for_this_cam = track_keys_per_cam[cam_id]
                        if len(features_this_cam) > 0 and len(original_keys_for_this_cam) == tracks_data_np.shape[0]:
                            track_ids_in_batch = tracks_data_np[:, 4].astype(int)
                            for i, track_id_in_batch in enumerate(track_ids_in_batch):
                                feature = features_this_cam.get(track_id_in_batch)
                                if feature is not None:
                                    original_key = original_keys_for_this_cam[i]
                                    if original_key[1] == track_id_in_batch: # Sanity check
                                        extracted_features_this_frame[original_key] = feature
                                    else:
                                        logger.warning(f"[{cam_id}] Mismatch mapping features back to TrackKey. Expected {original_key}, got T:{track_id_in_batch}")
                        elif len(features_this_cam) > 0:
                             logger.warning(f"[{cam_id}] Feature count ({len(features_this_cam)}) or key count ({len(original_keys_for_this_cam)}) mismatch with track data batch size ({tracks_data_np.shape[0]}). Feature mapping might be incomplete.")

                    except Exception as fe_err:
                        logger.error(f"[{cam_id}] Error during batched feature extraction call: {fe_err}", exc_info=False)

        timings['feature_ext'] = time.time() - t_feat_start


        # --- Stage 3: Re-ID Association (Uses MODIFIED _perform_reid_association) ---
        t_reid_start = time.time()
        assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start


        # --- Stage 4: Combine Results and Finalize Track Data ---
        # --- Logic remains the same (uses output from modified association) ---
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

                            # Priority: Use ID assigned or updated in this ReID cycle
                            if current_track_key in assigned_global_ids_this_cycle:
                                global_id = assigned_global_ids_this_cycle[current_track_key]
                            else:
                                # Fallback: Use last known mapping if ReID wasn't run/successful this cycle
                                global_id = self.track_to_global_id.get(current_track_key) # Will be None if it just appeared and ReID failed

                            final_results_per_camera[cam_id].append({
                                'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                                'track_id': track_id, 'global_id': global_id,
                                'conf': conf, 'class_id': cls
                            })
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"[{cam_id}] Failed to parse final track data from {track_data}: {e}", exc_info=False)


        # --- Stage 5: Update State and Cleanup (Uses MODIFIED _update_and_cleanup_state) ---
        # Uses current_frame_active_track_keys collected earlier
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch

        return ProcessedBatchResult(
            results_per_camera=dict(final_results_per_camera),
            timings=dict(timings),
            processed_this_frame=True,
            # Pass the triggers detected in this frame (based on quadrant overlap) for visualization
            handoff_triggers=list(self.handoff_triggers_this_frame)
        )