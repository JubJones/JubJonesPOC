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
    HandoffTriggerInfo, QuadrantName, ExitDirection, MapCoordinates
)
# Import utils relatively
from reid_poc.utils import calculate_cosine_similarity, normalize_embedding, normalize_overlap_set, project_point_to_map


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
        """Initializes the pipeline with pre-loaded models, trackers, and homography."""
        self.config = config
        self.device = config.device
        self.camera_ids = config.selected_cameras # Use the validated list from config setup

        # Store pre-initialized components
        self.detector = detector
        self.detector_transforms = detector_transforms
        self.reid_model = reid_model
        self.trackers = trackers
        # Store loaded homography matrices
        self.homography_matrices = config.homography_matrices

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
        logger.info(f"BEV Map Plotting: {'Enabled' if config.enable_bev_map else 'Disabled'}")
        logger.info(f"Homography matrices loaded for cameras: {list(self.homography_matrices.keys())}")


        # --- State Initialization ---
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {} # Stores representative feature for each GlobalID
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {} # Maps (cam_id, track_id) -> global_id
        self.next_global_id: GlobalID = 1 # Counter for assigning new global IDs
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        self.track_last_reid_frame: Dict[TrackKey, int] = {}
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        self.handoff_triggers_this_frame: List[HandoffTriggerInfo] = []
        self.processed_frame_counter: int = 0
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {}
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
        then handoff targets, then the full gallery. Includes conflict resolution with
        second-pass matching for reverted tracks.
        """
        # --- Prepare Galleries ---
        if not features_per_track:
            return {}

        # Prepare main gallery (filter invalid entries)
        valid_main_gallery_items = [
            (gid, emb) for gid, emb in self.reid_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        full_main_gallery_ids: List[GlobalID] = []
        full_main_gallery_embeddings: List[FeatureVector] = []
        if valid_main_gallery_items:
            full_main_gallery_ids, full_main_gallery_embeddings = zip(*valid_main_gallery_items)

        # Prepare lost gallery (filter invalid entries)
        valid_lost_gallery_items = [
            (gid, (emb, frame_num)) for gid, (emb, frame_num) in self.lost_track_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        lost_gallery_ids: List[GlobalID] = []
        lost_gallery_embeddings: List[FeatureVector] = []
        if valid_lost_gallery_items:
            lost_gallery_ids, lost_gallery_data = zip(*valid_lost_gallery_items)
            lost_gallery_embeddings = [data[0] for data in lost_gallery_data] # Extract just embeddings

        # Map active handoff triggers for quick lookup
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo] = {
            trigger.source_track_key: trigger
            for trigger in self.handoff_triggers_this_frame
            if trigger.source_track_key in features_per_track
        }

        # --- Perform Initial Matching ---
        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float]] = {} # Store (gid, score)

        for track_key, new_embedding_raw in features_per_track.items():
            cam_id, track_id = track_key

            if new_embedding_raw is None or not np.isfinite(new_embedding_raw).all() or new_embedding_raw.size == 0:
                logger.warning(f"Skipping ReID for {track_key}: Received invalid embedding.")
                tentative_assignments[track_key] = (None, -1.0)
                continue

            normalized_new_embedding = normalize_embedding(new_embedding_raw)
            assigned_global_id: Optional[GlobalID] = None
            max_similarity_score: float = -1.0 # Initialize score

            # --- Step 1: Check Lost Track Gallery First ---
            if lost_gallery_ids:
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
                            assigned_global_id = best_lost_match_gid
                            max_similarity_score = current_max_lost_similarity # Store score
                            logger.info(f"Re-associating {track_key} with LOST GID {best_lost_match_gid} (Sim: {max_similarity_score:.3f}).")
                            # Move from lost gallery to main, update embedding
                            lost_embedding, _ = self.lost_track_gallery.pop(assigned_global_id)
                            updated_embedding = (
                                self.config.gallery_ema_alpha * lost_embedding +
                                (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding
                            )
                            self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)
                except Exception as lost_sim_err:
                    logger.error(f"Similarity calculation error during Lost Gallery check for {track_key}: {lost_sim_err}", exc_info=False)

            # --- Step 2: Check Main Gallery (if not re-associated from lost tracks) ---
            if assigned_global_id is None:
                best_main_match_global_id: Optional[GlobalID] = None
                triggered_handoff = active_triggers_map.get(track_key)
                gallery_ids_to_check = full_main_gallery_ids
                gallery_embeddings_to_check = full_main_gallery_embeddings
                is_checking_filtered_gallery = False

                # Apply handoff filtering if applicable
                if triggered_handoff:
                    target_cam_id = triggered_handoff.rule.target_cam_id
                    relevant_cams = self._get_relevant_handoff_cams(target_cam_id)
                    filtered_gallery_indices = [idx for idx, gid in enumerate(full_main_gallery_ids) if self.global_id_last_seen_cam.get(gid) in relevant_cams]
                    if filtered_gallery_indices:
                        gallery_ids_to_check = [full_main_gallery_ids[i] for i in filtered_gallery_indices]
                        gallery_embeddings_to_check = [full_main_gallery_embeddings[i] for i in filtered_gallery_indices]
                        is_checking_filtered_gallery = True
                    else:
                        gallery_ids_to_check = [] # No relevant IDs found

                # Perform similarity check (potentially filtered)
                if gallery_ids_to_check:
                    try:
                        main_similarities = np.array([calculate_cosine_similarity(normalized_new_embedding, gal_emb) for gal_emb in gallery_embeddings_to_check])
                        if main_similarities.size > 0:
                            max_main_similarity_idx = np.argmax(main_similarities)
                            current_max_main_similarity = main_similarities[max_main_similarity_idx]
                            if current_max_main_similarity >= self.config.reid_similarity_threshold:
                                best_main_match_global_id = gallery_ids_to_check[max_main_similarity_idx]
                                # Store potential assignment and score
                                assigned_global_id = best_main_match_global_id
                                max_similarity_score = current_max_main_similarity
                    except Exception as sim_err:
                        logger.error(f"Similarity calculation error during Main Gallery check for {track_key}: {sim_err}", exc_info=False)
                        assigned_global_id = None # Reset on error
                        max_similarity_score = -1.0

                # Fallback to full gallery check if filtered check failed during handoff
                if triggered_handoff and is_checking_filtered_gallery and assigned_global_id is None and full_main_gallery_ids:
                    try:
                        similarities_full = np.array([calculate_cosine_similarity(normalized_new_embedding, gal_emb) for gal_emb in full_main_gallery_embeddings])
                        if similarities_full.size > 0:
                            max_similarity_idx_full = np.argmax(similarities_full)
                            current_max_similarity_full = similarities_full[max_similarity_idx_full]
                            if current_max_similarity_full >= self.config.reid_similarity_threshold:
                                best_main_match_global_id = full_main_gallery_ids[max_similarity_idx_full]
                                # Store potential assignment and score
                                assigned_global_id = best_main_match_global_id
                                max_similarity_score = current_max_similarity_full
                    except Exception as sim_err_full:
                        logger.error(f"Similarity calculation error during FULL main gallery fallback for {track_key}: {sim_err_full}", exc_info=False)
                        assigned_global_id = None # Reset on error
                        max_similarity_score = -1.0

                # Update gallery if a main gallery match was found
                if assigned_global_id is not None:
                    current_gallery_emb = self.reid_gallery.get(assigned_global_id)
                    if current_gallery_emb is not None:
                        updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb + (1.0 - self.config.gallery_ema_alpha) * normalized_new_embedding)
                        self.reid_gallery[assigned_global_id] = normalize_embedding(updated_embedding)
                    else:
                        self.reid_gallery[assigned_global_id] = normalized_new_embedding # Should only happen if GID somehow vanished


            # --- Step 3: Assign New Global ID ---
            if assigned_global_id is None:
                assigned_global_id = self.next_global_id
                self.next_global_id += 1
                self.reid_gallery[assigned_global_id] = normalized_new_embedding
                max_similarity_score = -1.0 # No match score for new IDs
                logger.info(f"Assigned NEW Global ID {assigned_global_id} to {track_key}")

            # Store tentative assignment and score
            tentative_assignments[track_key] = (assigned_global_id, max_similarity_score)

        # --- CONFLICT RESOLUTION ---
        final_assignments: Dict[TrackKey, Optional[GlobalID]] = {} # Stores final decision AFTER resolution
        newly_assigned_global_ids: Dict[TrackKey, Optional[GlobalID]] = {} # Dict to be returned by function
        assignments_by_cam: Dict[CameraID, Dict[GlobalID, List[Tuple[TrackID, float]]]] = defaultdict(lambda: defaultdict(list))

        # Group tentative assignments by camera and global ID
        for track_key, (gid, score) in tentative_assignments.items():
            if gid is not None:
                cam_id, track_id = track_key
                assignments_by_cam[cam_id][gid].append((track_id, score))

        # Resolve conflicts within each camera
        reverted_keys: Set[TrackKey] = set()
        for cam_id, gid_map in assignments_by_cam.items():
            for gid, track_score_list in gid_map.items():
                if len(track_score_list) > 1: # Conflict detected!
                    # Sort by score descending, keep the best one
                    track_score_list.sort(key=lambda x: x[1], reverse=True)
                    best_track_id, best_score = track_score_list[0]
                    logger.warning(f"[{cam_id}] Conflict for GID {gid}. Tracks: {track_score_list}. Keeping T:{best_track_id} (Score: {best_score:.3f}).")

                    # Mark others for reversion
                    for i in range(1, len(track_score_list)):
                        reverted_track_id, reverted_score = track_score_list[i]
                        reverted_keys.add((cam_id, reverted_track_id))
                        logger.warning(f"[{cam_id}]   -> Reverting T:{reverted_track_id} (Score: {reverted_score:.3f}).")


        # Create initial final assignments, excluding reverted ones
        # Also update state mappings for non-reverted tracks
        for track_key, (gid, score) in tentative_assignments.items():
            if track_key not in reverted_keys:
                final_assignments[track_key] = gid
                newly_assigned_global_ids[track_key] = gid # Add to the dict that gets returned
                # Update state mappings for non-reverted assignments
                if gid is not None:
                    self.track_to_global_id[track_key] = gid
                    self.global_id_last_seen_cam[gid] = track_key[0] # cam_id


        # --- Handle Reverted Tracks (Option 2: Second Pass Matching) ---
        for reverted_key in reverted_keys:
            cam_id, reverted_track_id = reverted_key
            logger.info(f"Attempting second pass matching for reverted track {reverted_key}...")

            # Find the GID that caused the conflict for THIS track key
            conflicting_gid: Optional[GlobalID] = None
            original_tentative_assignment = tentative_assignments.get(reverted_key)
            if original_tentative_assignment:
                conflicting_gid = original_tentative_assignment[0]

            # Get the embedding for the reverted track
            reverted_embedding = features_per_track.get(reverted_key)

            # Immediate fallback to new ID if data is missing
            if reverted_embedding is None or conflicting_gid is None:
                logger.warning(f"Cannot perform second pass for {reverted_key}: Missing embedding or original conflicting GID. Assigning new ID.")
                new_gid = self.next_global_id
                self.next_global_id += 1
                if reverted_embedding is not None: # Still try to add to gallery if embedding exists
                    normalized_reverted_embedding = normalize_embedding(reverted_embedding)
                    self.reid_gallery[new_gid] = normalized_reverted_embedding
                else: # No embedding, cannot add to gallery
                    logger.error(f"No embedding for reverted track {reverted_key}, cannot add new GID {new_gid} to gallery.")
                # Update state even if gallery add fails
                final_assignments[reverted_key] = new_gid
                newly_assigned_global_ids[reverted_key] = new_gid
                self.track_to_global_id[reverted_key] = new_gid
                self.global_id_last_seen_cam[new_gid] = cam_id
                logger.info(f"Assigned NEW Global ID {new_gid} to reverted track {reverted_key} (fallback).")
                continue # Move to the next reverted key

            # Proceed with second pass matching
            normalized_reverted_embedding = normalize_embedding(reverted_embedding)
            second_pass_assigned_gid: Optional[GlobalID] = None
            second_pass_score: float = -1.0

            # --- 2a: Check Lost Gallery (excluding conflicting GID) ---
            valid_lost_items_filtered = [
                (gid, (emb, fn)) for gid, (emb, fn) in self.lost_track_gallery.items()
                if gid != conflicting_gid and emb is not None and np.isfinite(emb).all() and emb.size > 0
            ]
            if valid_lost_items_filtered:
                lost_gids_f, lost_data_f = zip(*valid_lost_items_filtered)
                lost_embs_f = [data[0] for data in lost_data_f]
                try:
                    lost_sims_f = np.array([calculate_cosine_similarity(normalized_reverted_embedding, emb) for emb in lost_embs_f])
                    if lost_sims_f.size > 0:
                        max_lost_idx_f = np.argmax(lost_sims_f)
                        max_lost_sim_f = lost_sims_f[max_lost_idx_f]
                        if max_lost_sim_f >= self.config.reid_similarity_threshold:
                            second_pass_assigned_gid = lost_gids_f[max_lost_idx_f]
                            second_pass_score = max_lost_sim_f
                            logger.info(f"Second Pass: Re-associating {reverted_key} with LOST GID {second_pass_assigned_gid} (Sim: {second_pass_score:.3f}).")
                            # Move from lost to main gallery and update embedding
                            lost_embedding, _ = self.lost_track_gallery.pop(second_pass_assigned_gid)
                            updated_embedding = (self.config.gallery_ema_alpha * lost_embedding +
                                                 (1.0 - self.config.gallery_ema_alpha) * normalized_reverted_embedding)
                            self.reid_gallery[second_pass_assigned_gid] = normalize_embedding(updated_embedding)
                except Exception as e_lost_2p:
                    logger.error(f"Error during second pass lost gallery check for {reverted_key}: {e_lost_2p}")

            # --- 2b: Check Main Gallery (if no lost match, excluding conflicting GID) ---
            if second_pass_assigned_gid is None:
                valid_main_items_filtered = [
                    (gid, emb) for gid, emb in self.reid_gallery.items()
                    # Ensure the ID isn't the conflicting one AND isn't one just assigned from the lost gallery in the step above
                    if gid != conflicting_gid and gid != second_pass_assigned_gid and emb is not None and np.isfinite(emb).all() and emb.size > 0
                ]
                if valid_main_items_filtered:
                    main_gids_f, main_embs_f = zip(*valid_main_items_filtered)
                    try:
                        main_sims_f = np.array([calculate_cosine_similarity(normalized_reverted_embedding, emb) for emb in main_embs_f])
                        if main_sims_f.size > 0:
                            max_main_idx_f = np.argmax(main_sims_f)
                            max_main_sim_f = main_sims_f[max_main_idx_f]
                            if max_main_sim_f >= self.config.reid_similarity_threshold:
                                second_pass_assigned_gid = main_gids_f[max_main_idx_f]
                                second_pass_score = max_main_sim_f
                                logger.info(f"Second Pass: Re-associating {reverted_key} with MAIN GID {second_pass_assigned_gid} (Sim: {second_pass_score:.3f}).")
                                # Update gallery embedding
                                current_gallery_emb = self.reid_gallery.get(second_pass_assigned_gid)
                                if current_gallery_emb is not None:
                                    updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb +
                                                        (1.0 - self.config.gallery_ema_alpha) * normalized_reverted_embedding)
                                    self.reid_gallery[second_pass_assigned_gid] = normalize_embedding(updated_embedding)
                                else: # Safety check
                                    self.reid_gallery[second_pass_assigned_gid] = normalized_reverted_embedding
                    except Exception as e_main_2p:
                        logger.error(f"Error during second pass main gallery check for {reverted_key}: {e_main_2p}")

            # --- 2c: Fallback to New ID if Second Pass Failed ---
            if second_pass_assigned_gid is None:
                logger.info(f"Second pass failed for {reverted_key}. Assigning new ID.")
                new_gid = self.next_global_id
                self.next_global_id += 1
                self.reid_gallery[new_gid] = normalized_reverted_embedding # Use the embedding we already have
                second_pass_assigned_gid = new_gid # Assign the new GID

            # --- Update Final State for Reverted Key ---
            final_assignments[reverted_key] = second_pass_assigned_gid # Track final decision (not really used after this)
            newly_assigned_global_ids[reverted_key] = second_pass_assigned_gid # IMPORTANT: Update the dict to be returned
            self.track_to_global_id[reverted_key] = second_pass_assigned_gid
            self.global_id_last_seen_cam[second_pass_assigned_gid] = cam_id

        # Return the dictionary reflecting the FINAL assignments after conflict resolution
        return newly_assigned_global_ids


    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """
        Updates the set of last seen tracks. Moves disappeared tracks' info to a temporary
        'lost_track_gallery' and purges expired entries from it. Cleans up other state.
        """
        # --- Function remains the same ---
        all_previous_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previous_track_keys - current_frame_active_track_keys

        for track_key in disappeared_track_keys:
            global_id = self.track_to_global_id.pop(track_key, None)
            if global_id is not None:
                last_feature = self.reid_gallery.get(global_id)
                if last_feature is not None:
                    # Only add to lost gallery if it's not already there (can happen with rapid disappear/reappear)
                    if global_id not in self.lost_track_gallery:
                         self.lost_track_gallery[global_id] = (last_feature, self.processed_frame_counter)
                         logger.info(f"Track {track_key} (GID {global_id}) disappeared. Moved info to lost gallery (frame {self.processed_frame_counter}).")
                    else:
                         # Update the frame counter if already in lost gallery
                         existing_feat, _ = self.lost_track_gallery[global_id]
                         self.lost_track_gallery[global_id] = (existing_feat, self.processed_frame_counter)
                         logger.debug(f"Track {track_key} (GID {global_id}) disappeared again. Updated lost timestamp.")
                else:
                    logger.warning(f"Track {track_key} (GID {global_id}) disappeared but feature not found in main gallery.")
            # Clean up other state associated with the disappeared track key
            if track_key in self.track_last_reid_frame:
                del self.track_last_reid_frame[track_key]

        # Purge expired entries from the lost gallery
        expired_lost_gids = [gid for gid, (feat, frame_num) in self.lost_track_gallery.items()
                             if (self.processed_frame_counter - frame_num) > self.config.lost_track_buffer_frames]
        for gid in expired_lost_gids:
            if gid in self.lost_track_gallery:
                del self.lost_track_gallery[gid]
                logger.info(f"Purged expired GID {gid} from lost track gallery.")
            # Optionally remove from main gallery too if it hasn't been seen actively
            # This is debatable - might want to keep longer term IDs even if lost temporarily
            # if gid in self.reid_gallery and self.global_id_last_seen_cam.get(gid) is None: # Check if GID still active elsewhere
                 # del self.reid_gallery[gid]
                 # logger.info(f"Also removed expired GID {gid} from main gallery.")


        # Update the set of track IDs seen in the *last* processed frame for each camera
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys:
            new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        # Clean up track_last_reid_frame for tracks that no longer exist
        current_reid_keys = set(self.track_last_reid_frame.keys())
        keys_to_delete_reid = current_reid_keys - current_frame_active_track_keys
        for key in keys_to_delete_reid:
            if key in self.track_last_reid_frame:
                del self.track_last_reid_frame[key]


    def process_frame_batch_full(self, frames: Dict[CameraID, FrameData], frame_idx_global: int) -> ProcessedBatchResult:
        """Processes a batch of frames: Detect -> Track -> Handoff Check -> Re-ID -> Project -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)

        self.processed_frame_counter += 1
        proc_frame_id = self.processed_frame_counter

        # --- Stages 1a, 1b, 1c, 1d, 1e --- (Preprocessing, Detection, Postprocessing, Tracking, Handoff Check)
        # --- Stage 1a: Preprocess ---
        t_prep_start = time.time()
        batch_input_tensors: List[torch.Tensor] = []
        batch_cam_ids: List[CameraID] = [] # Keep track of order
        batch_original_shapes: List[Tuple[int, int]] = [] # (height, width)
        batch_scale_factors: List[ScaleFactors] = [] # (scale_x, scale_y)
        for cam_id, frame_bgr in frames.items():
             if frame_bgr is not None and frame_bgr.size > 0:
                 original_h, original_w = frame_bgr.shape[:2]
                 frame_for_det = frame_bgr; scale_x, scale_y = 1.0, 1.0
                 if self.config.detection_input_width and original_w > self.config.detection_input_width:
                     target_w = self.config.detection_input_width; scale = target_w / original_w; target_h = int(original_h * scale)
                     try: frame_for_det = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR); scale_x = original_w / target_w; scale_y = original_h / target_h
                     except Exception as resize_err: logger.warning(f"[{cam_id}] Resizing failed: {resize_err}. Using original."); frame_for_det = frame_bgr; scale_x, scale_y = 1.0, 1.0
                 try:
                     img_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB); img_pil = Image.fromarray(img_rgb); input_tensor = self.detector_transforms(img_pil)
                     batch_input_tensors.append(input_tensor.to(self.device)); batch_cam_ids.append(cam_id); batch_original_shapes.append((original_h, original_w)); batch_scale_factors.append((scale_x, scale_y))
                 except Exception as transform_err: logger.error(f"[{cam_id}] Preprocessing failed: {transform_err}", exc_info=False)
        timings['preprocess'] = time.time() - t_prep_start

        # --- Stage 1b: Detection ---
        t_detect_start = time.time()
        all_predictions: List[Dict[str, torch.Tensor]] = []
        if batch_input_tensors:
             try:
                 with torch.no_grad():
                     use_amp_runtime = self.config.use_amp and self.device.type == 'cuda'
                     with torch.cuda.amp.autocast(enabled=use_amp_runtime): all_predictions = self.detector(batch_input_tensors)
             except Exception as e: logger.error(f"Object detection failed: {e}", exc_info=False); all_predictions = []
        timings['detection_batched'] = time.time() - t_detect_start

        # --- Stage 1c: Postprocess ---
        t_postproc_start = time.time()
        detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)
        if len(all_predictions) == len(batch_cam_ids):
             for i, prediction_dict in enumerate(all_predictions):
                 cam_id = batch_cam_ids[i]; original_h, original_w = batch_original_shapes[i]; scale_x, scale_y = batch_scale_factors[i]
                 try:
                     pred_boxes = prediction_dict['boxes'].cpu().numpy(); pred_labels = prediction_dict['labels'].cpu().numpy(); pred_scores = prediction_dict['scores'].cpu().numpy()
                     for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                         if label == self.config.person_class_id and score >= self.config.detection_confidence_threshold:
                             x1, y1, x2, y2 = box; orig_x1 = max(0.0, x1 * scale_x); orig_y1 = max(0.0, y1 * scale_y); orig_x2 = min(float(original_w - 1), x2 * scale_x); orig_y2 = min(float(original_h - 1), y2 * scale_y)
                             if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1: bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32); detections_per_camera[cam_id].append({'bbox_xyxy': bbox_orig, 'conf': float(score), 'class_id': int(label)})
                 except Exception as postproc_err: logger.error(f"[{cam_id}] Error postprocessing detections: {postproc_err}", exc_info=False)
        elif batch_input_tensors: logger.error(f"Detection output count mismatch: {len(all_predictions)} vs {len(batch_cam_ids)}")
        timings['postprocess_scale'] = time.time() - t_postproc_start

        # --- Stage 1d: Tracking ---
        t_track_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        original_shapes_map: Dict[CameraID, Tuple[int, int]] = {cam_id: shape for cam_id, shape in zip(batch_cam_ids, batch_original_shapes)}
        for cam_id in self.camera_ids:
             tracker = self.trackers.get(cam_id)
             if not tracker: logger.warning(f"[{cam_id}] Tracker instance missing."); current_frame_tracker_outputs[cam_id] = np.empty((0, 8)); continue
             cam_detections = detections_per_camera.get(cam_id, []); np_dets = np.empty((0, 6))
             if cam_detections:
                 try: np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                 except Exception as format_err: logger.error(f"[{cam_id}] Failed to format detections for tracker: {format_err}", exc_info=False)
             original_frame_bgr = frames.get(cam_id); frame_shape_orig = original_shapes_map.get(cam_id); dummy_frame_shape = frame_shape_orig if frame_shape_orig else (1080, 1920); dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*dummy_frame_shape, 3), dtype=np.uint8)
             tracked_dets_np = np.empty((0, 8))
             try:
                 tracked_dets_list = tracker.update(np_dets, dummy_frame)
                 if tracked_dets_list is not None and len(tracked_dets_list) > 0:
                     tracked_dets_np_maybe = np.array(tracked_dets_list)
                     if tracked_dets_np_maybe.ndim == 2 and tracked_dets_np_maybe.shape[1] >= 7: tracked_dets_np = tracked_dets_np_maybe[:, :8]
                     else: logger.warning(f"[{cam_id}] Tracker output has unexpected shape {tracked_dets_np_maybe.shape}. Treating as empty.")
             except Exception as e: logger.error(f"[{cam_id}] Tracker update failed: {e}", exc_info=True)
             current_frame_tracker_outputs[cam_id] = tracked_dets_np
        timings['tracking'] = time.time() - t_track_start

        # --- Stage 1e: Handoff Check ---
        t_handoff_start = time.time()
        self.handoff_triggers_this_frame.clear()
        if self.config.min_bbox_overlap_ratio_in_quadrant > 0:
             for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
                 frame_shape_orig = original_shapes_map.get(cam_id); self._check_handoff_triggers(cam_id, tracked_dets_np, frame_shape_orig)
        track_keys_triggering_handoff: Set[TrackKey] = {trigger.source_track_key for trigger in self.handoff_triggers_this_frame}
        timings['handoff_check'] = time.time() - t_handoff_start


        # --- Stage: Decide ReID ---
        t_reid_decision_start = time.time()
        tracks_to_extract_features_for: Dict[TrackKey, np.ndarray] = {}
        current_frame_active_track_keys: Set[TrackKey] = set()
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            if tracked_dets_np.shape[0] > 0:
                original_frame_bgr = frames.get(cam_id); previous_processed_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                for track_data in tracked_dets_np:
                    if len(track_data) < 5: continue
                    try: track_id = int(track_data[4])
                    except (ValueError, IndexError): logger.warning(f"[{cam_id}] Invalid track ID format in track_data: {track_data}"); continue
                    current_track_key: TrackKey = (cam_id, track_id); current_frame_active_track_keys.add(current_track_key)
                    if original_frame_bgr is not None and original_frame_bgr.size > 0:
                        is_newly_seen = track_id not in previous_processed_cam_track_ids; last_reid_attempt_proc_idx = self.track_last_reid_frame.get(current_track_key, -self.config.reid_refresh_interval_frames - 1); is_due_for_refresh = (proc_frame_id - last_reid_attempt_proc_idx) >= self.config.reid_refresh_interval_frames; is_triggering_handoff_now = current_track_key in track_keys_triggering_handoff
                        trigger_reid = is_newly_seen or is_due_for_refresh or is_triggering_handoff_now
                        if trigger_reid: tracks_to_extract_features_for[current_track_key] = track_data; self.track_last_reid_frame[current_track_key] = proc_frame_id; # Logging debug moved outside loop for less noise
        timings['reid_decision'] = time.time() - t_reid_decision_start
        logger.debug(f"Frame {proc_frame_id}: Decided to extract ReID features for {len(tracks_to_extract_features_for)} tracks.")

        # --- Stage 2: Feature Extraction ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        if tracks_to_extract_features_for:
            tracks_grouped_by_cam: Dict[CameraID, List[np.ndarray]] = defaultdict(list); track_keys_per_cam: Dict[CameraID, List[TrackKey]] = defaultdict(list)
            for track_key, track_data in tracks_to_extract_features_for.items(): cam_id, _ = track_key; tracks_grouped_by_cam[cam_id].append(track_data); track_keys_per_cam[cam_id].append(track_key)
            for cam_id, tracks_data_list in tracks_grouped_by_cam.items():
                frame_bgr = frames.get(cam_id)
                if frame_bgr is not None and frame_bgr.size > 0 and tracks_data_list:
                    try:
                        tracks_data_np = np.array(tracks_data_list); features_this_cam: Dict[TrackID, FeatureVector] = self._extract_features_for_tracks(frame_bgr, tracks_data_np); original_keys_for_this_cam = track_keys_per_cam[cam_id]
                        if len(features_this_cam) > 0 and len(original_keys_for_this_cam) == tracks_data_np.shape[0]:
                            track_ids_in_batch = tracks_data_np[:, 4].astype(int)
                            for i, track_id_in_batch in enumerate(track_ids_in_batch):
                                feature = features_this_cam.get(track_id_in_batch)
                                if feature is not None:
                                    original_key = original_keys_for_this_cam[i]
                                    if original_key[1] == track_id_in_batch: extracted_features_this_frame[original_key] = feature
                                    else: logger.warning(f"[{cam_id}] Mismatch mapping features back to TrackKey. Expected {original_key}, got T:{track_id_in_batch}")
                        elif len(features_this_cam) > 0: logger.warning(f"[{cam_id}] Feature count ({len(features_this_cam)}) or key count ({len(original_keys_for_this_cam)}) mismatch with track data batch size ({tracks_data_np.shape[0]}). Feature mapping might be incomplete.")
                    except Exception as fe_err: logger.error(f"[{cam_id}] Error during batched feature extraction call: {fe_err}", exc_info=False)
        timings['feature_ext'] = time.time() - t_feat_start
        logger.debug(f"Frame {proc_frame_id}: Extracted {len(extracted_features_this_frame)} features.")

        # --- Stage 3: Re-ID Association (with Conflict Resolution) ---
        t_reid_start = time.time()
        final_assigned_global_ids_this_cycle = self._perform_reid_association(extracted_features_this_frame)
        timings['reid'] = time.time() - t_reid_start
        logger.debug(f"Frame {proc_frame_id}: ReID association completed. Final assignments this cycle: {len(final_assigned_global_ids_this_cycle)}")

        # --- Stage 4: Combine Results, Project to Map, and Finalize Track Data ---
        t_project_start = time.time()
        projection_success_count = 0
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
            homography_matrix = self.homography_matrices.get(cam_id) # Get H for this camera
            logger.debug(f"[{cam_id}] Projection Check: Homography matrix is {'AVAILABLE' if homography_matrix is not None else 'MISSING'}")
            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    if len(track_data) >= 7: # Need x1,y1,x2,y2,tid,conf,cls
                        map_coords: MapCoordinates = None # Default to None
                        track_info_dict = {} # Build the dict progressively
                        try:
                            x1, y1, x2, y2 = map(float, track_data[0:4])
                            track_id = int(track_data[4])
                            conf = float(track_data[5])
                            cls = int(track_data[6])
                            current_track_key: TrackKey = (cam_id, track_id)

                            # Get Global ID using the *final* assignments from ReID
                            # It could come from the current cycle's resolution OR persistent state
                            global_id: Optional[GlobalID] = final_assigned_global_ids_this_cycle.get(
                                current_track_key, # Check if assigned/resolved in this cycle
                                self.track_to_global_id.get(current_track_key) # Fallback to existing state if not processed this cycle
                            )

                            # Store basic info first
                            track_info_dict = {
                                'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                                'track_id': track_id,
                                'global_id': global_id,
                                'conf': conf,
                                'class_id': cls,
                                'map_coords': None # Initialize map_coords
                            }
                            logger.debug(f"[{cam_id}] Processing T:{track_id} G:{global_id}. BBox: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

                            # --- Project to Map Coordinates ---
                            if homography_matrix is not None:
                                # Use bottom-center of the bounding box
                                image_point_x = (x1 + x2) / 2.0
                                image_point_y = y2 # Bottom y-coordinate
                                logger.debug(f"[{cam_id}] T:{track_id} - Attempting projection for image point: ({image_point_x:.1f}, {image_point_y:.1f})")
                                map_coords = project_point_to_map(
                                    (image_point_x, image_point_y),
                                    homography_matrix
                                )
                                if map_coords is not None:
                                    track_info_dict['map_coords'] = map_coords # Update dict
                                    projection_success_count += 1
                                    logger.debug(f"[{cam_id}] T:{track_id} - Projection successful: Map coords = {map_coords}")
                                else:
                                    logger.debug(f"[{cam_id}] T:{track_id} - Projection returned None.")
                            else:
                                logger.debug(f"[{cam_id}] T:{track_id} - Skipping projection (no homography matrix).")

                            final_results_per_camera[cam_id].append(track_info_dict)
                            logger.debug(f"[{cam_id}] T:{track_id} - Final track data appended: {track_info_dict}")

                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"[{cam_id}] Failed to parse/project final track data from {track_data}: {e}", exc_info=False)
            else:
                logger.debug(f"[{cam_id}] No tracks found in tracker output for this frame.")

        timings['projection'] = time.time() - t_project_start
        logger.debug(f"Frame {proc_frame_id}: Projection stage finished. Successful projections: {projection_success_count}")


        # --- Stage 5: Update State and Cleanup ---
        # This uses the active track keys from *this frame* to manage disappearance
        self._update_and_cleanup_state(current_frame_active_track_keys)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch

        return ProcessedBatchResult(
            results_per_camera=dict(final_results_per_camera),
            timings=dict(timings),
            processed_this_frame=True,
            handoff_triggers=list(self.handoff_triggers_this_frame)
        )