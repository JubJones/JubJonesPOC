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
from scipy.spatial.distance import cdist # Added for batched distance calculation

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
    # ... (error handling as before) ...
    logging.critical(f"Failed to import boxmot components needed for Pipeline. Is boxmot installed? Error: {e}")
    BaseModelBackend = type(None)
    BaseTracker = type(None)

# Use relative imports
from reid_poc.config import PipelineConfig, CameraConfig
from reid_poc.alias_types import (
    CameraID, TrackID, GlobalID, TrackKey, FeatureVector, BoundingBox, ExitRule,
    Detection, TrackData, FrameData, Timings, ProcessedBatchResult, ScaleFactors,
    HandoffTriggerInfo, QuadrantName, ExitDirection, MapCoordinates
)
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
        """Initializes the pipeline with pre-loaded models, trackers, and configurations."""
        self.config = config
        self.device = config.device
        self.camera_ids = config.selected_cameras # Use the validated list

        # Store pre-initialized components
        self.detector = detector
        self.detector_transforms = detector_transforms
        self.reid_model = reid_model
        self.trackers = trackers
        # Homography matrices are now within config.cameras_config[cam_id].homography_matrix

        # --- Logging config details (INFO level) ---
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids} on device: {self.device.type}")
        logger.info(f" - Scene: {config.selected_scene}")
        logger.info(f" - Detector Confidence: {config.detection_confidence_threshold}")
        logger.info(f" - ReID Similarity Threshold: {config.reid_similarity_threshold}")
        logger.info(f" - ReID Refresh Interval (Processed Frames): {config.reid_refresh_interval_frames}")
        logger.info(f" - Frame Skip Rate: {config.frame_skip_rate}")
        logger.info(f" - Tracker Type: {config.tracker_type}")
        logger.info(f" - Handoff Overlap Ratio Threshold: {config.min_bbox_overlap_ratio_in_quadrant:.2f}")
        logger.info(f" - Possible Overlaps: {config.possible_overlaps}")
        logger.info(f" - Lost Track Buffer: {config.lost_track_buffer_frames} frames")
        logger.info(f" - BEV Map Plotting: {'Enabled' if config.enable_bev_map else 'Disabled'}")
        logger.info(f" - Require Homography for BEV: {config.require_homography_for_bev}")
        homography_count = sum(1 for cfg in config.cameras_config.values() if cfg.homography_matrix is not None)
        logger.info(f" - Homography matrices loaded for {homography_count}/{len(self.camera_ids)} cameras.")


        # --- State Initialization ---
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.next_global_id: GlobalID = 1
        self.last_seen_track_ids: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        self.track_last_reid_frame: Dict[TrackKey, int] = {}
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        # self.handoff_triggers_this_frame: List[HandoffTriggerInfo] = [] # Managed within process_frame_batch
        self.processed_frame_counter: int = 0
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {}
        self.possible_overlaps_normalized = config.possible_overlaps # Already normalized in config setup

    def _extract_features_for_tracks(self, frame_bgr: FrameData, tracked_dets_np: np.ndarray) -> Dict[TrackID, FeatureVector]:
        """Extracts Re-ID features for the given tracks using the provided frame."""
        # --- Function remains largely the same, ensure tracked_dets_np format is correct ---
        features: Dict[TrackID, FeatureVector] = {}
        if self.reid_model is None: return features
        if frame_bgr is None or frame_bgr.size == 0: return features
        if tracked_dets_np.shape[0] == 0: return features

        # Expecting [x1, y1, x2, y2, track_id, conf, class_id, ...] from BoxMOT trackers
        if tracked_dets_np.shape[1] < 5:
             logger.warning(f"Track data has unexpected shape {tracked_dets_np.shape}, expected at least 5 columns (xyxy, id). Skipping feature extraction.")
             return features

        bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
        track_ids_float = tracked_dets_np[:, 4]

        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4: return features

        try:
            batch_features = self.reid_model.get_features(bboxes_xyxy, frame_bgr)
            if batch_features is not None and len(batch_features) == len(track_ids_float):
                for i, det_feature in enumerate(batch_features):
                    if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                        track_id = int(track_ids_float[i])
                        features[track_id] = det_feature
            # else: # Debug logging if mismatch
        except Exception as e:
            logger.error(f"Feature extraction call failed: {e}", exc_info=False)
        return features

    def _check_handoff_triggers(
        self,
        cam_id: CameraID,
        tracked_dets_np: np.ndarray,
        frame_shape: Optional[Tuple[int, int]],
        cam_exit_rules: List[ExitRule] # Pass rules explicitly
    ) -> List[HandoffTriggerInfo]:
        """
        Checks if tracks trigger handoff rules based on quadrant overlap.
        Returns a list of triggers detected for this camera in this frame.
        """
        triggers_found: List[HandoffTriggerInfo] = [] # Local list for this call
        # Check prerequisites
        if (not cam_exit_rules or
            not frame_shape or frame_shape[0] <= 0 or frame_shape[1] <= 0 or
            tracked_dets_np.shape[0] == 0 or
            self.config.min_bbox_overlap_ratio_in_quadrant <= 0):
            return triggers_found # Cannot perform handoff check

        H, W = frame_shape
        mid_x, mid_y = W // 2, H // 2

        quadrant_regions: Dict[QuadrantName, Tuple[int, int, int, int]] = {
            'upper_left': (0, 0, mid_x, mid_y), 'upper_right': (mid_x, 0, W, mid_y),
            'lower_left': (0, mid_y, mid_x, H), 'lower_right': (mid_x, mid_y, W, H),
        }
        direction_to_quadrants: Dict[ExitDirection, List[QuadrantName]] = {
            'up': ['upper_left', 'upper_right'], 'down': ['lower_left', 'lower_right'],
            'left': ['upper_left', 'lower_left'], 'right': ['upper_right', 'lower_right'],
        }

        processed_track_ids_this_cam = set()

        for rule in cam_exit_rules:
            relevant_quadrant_names = direction_to_quadrants.get(rule.direction, [])
            if not relevant_quadrant_names: continue
            exit_regions_coords = [quadrant_regions[name] for name in relevant_quadrant_names if name in quadrant_regions]
            if not exit_regions_coords: continue

            for track_data in tracked_dets_np:
                if len(track_data) < 5: continue
                try: track_id = int(track_data[4])
                except (ValueError, IndexError): continue
                if track_id in processed_track_ids_this_cam: continue

                bbox = track_data[0:4].astype(np.float32)
                x1, y1, x2, y2 = map(int, bbox)
                bbox_w, bbox_h = x2 - x1, y2 - y1
                if bbox_w <= 0 or bbox_h <= 0: continue
                bbox_area = float(bbox_w * bbox_h)

                total_intersection_area = 0.0
                for qx1, qy1, qx2, qy2 in exit_regions_coords:
                    inter_x1, inter_y1 = max(x1, qx1), max(y1, qy1)
                    inter_x2, inter_y2 = min(x2, qx2), min(y2, qy2)
                    inter_w = max(0, inter_x2 - inter_x1); inter_h = max(0, inter_y2 - inter_y1)
                    total_intersection_area += float(inter_w * inter_h)

                if bbox_area > 1e-5 and (total_intersection_area / bbox_area) >= self.config.min_bbox_overlap_ratio_in_quadrant:
                    source_track_key: TrackKey = (cam_id, track_id)
                    trigger_info = HandoffTriggerInfo(
                        source_track_key=source_track_key, rule=rule, source_bbox=bbox
                    )
                    triggers_found.append(trigger_info)
                    processed_track_ids_this_cam.add(track_id) # Mark processed for this cam/frame

                    logger.info(
                        f"HANDOFF TRIGGER: Track {source_track_key} matched rule '{rule.direction}' -> Cam [{rule.target_cam_id}] Area [{rule.target_entry_area}]."
                    )
                    # Don't break inner track loop, let one track potentially match multiple rules if needed (though unlikely good)
                    # But processed_track_ids_this_cam prevents multiple appends for the same track_id here
                    break # Optimization: If a track triggers *a* rule, move to the next track for this camera

        return triggers_found

    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        """Gets the target camera and any possibly overlapping cameras for handoff."""
        # --- Function remains the same ---
        relevant_cams = {target_cam_id}
        for c1, c2 in self.possible_overlaps_normalized:
            if c1 == target_cam_id: relevant_cams.add(c2)
            elif c2 == target_cam_id: relevant_cams.add(c1)
        return relevant_cams

    def _perform_reid_association_batched(
        self,
        features_per_track: Dict[TrackKey, FeatureVector],
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> Dict[TrackKey, Optional[GlobalID]]:
        """
        Associates tracks with Global IDs using batched similarity calculation,
        prioritizing lost tracks, handling handoff context, and resolving conflicts.
        """
        if not features_per_track:
            return {}

        # --- 1. Prepare Query and Gallery Data ---
        query_track_keys: List[TrackKey] = []
        query_embeddings_list: List[FeatureVector] = []
        for tk, feat in features_per_track.items():
            if feat is not None and np.isfinite(feat).all() and feat.size > 0:
                normalized_feat = normalize_embedding(feat)
                query_track_keys.append(tk)
                query_embeddings_list.append(normalized_feat)
            else:
                 logger.warning(f"Skipping ReID for {tk}: Invalid embedding provided.")

        if not query_track_keys:
            return {} # No valid features to process

        query_embeddings_np = np.array(query_embeddings_list, dtype=np.float32)

        gallery_gids: List[GlobalID] = []
        gallery_embeddings_list: List[FeatureVector] = []
        gallery_types: List[str] = [] # 'lost' or 'main'

        # Add lost gallery items first (higher priority)
        valid_lost_items = [
            (gid, data) for gid, data in self.lost_track_gallery.items()
            if data[0] is not None and np.isfinite(data[0]).all() and data[0].size > 0
        ]
        for gid, (emb, frame_num) in valid_lost_items:
            gallery_gids.append(gid)
            gallery_embeddings_list.append(normalize_embedding(emb)) # Ensure gallery is normalized
            gallery_types.append('lost')
        num_lost_gallery = len(gallery_gids)

        # Add main gallery items
        valid_main_items = [
            (gid, emb) for gid, emb in self.reid_gallery.items()
            if emb is not None and np.isfinite(emb).all() and emb.size > 0
        ]
        for gid, emb in valid_main_items:
            gallery_gids.append(gid)
            gallery_embeddings_list.append(normalize_embedding(emb)) # Ensure gallery is normalized
            gallery_types.append('main')

        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float]] = {} # Store (gid, score)
        final_assignments: Dict[TrackKey, Optional[GlobalID]] = {} # Final result after conflict resolution
        reverted_keys_for_second_pass: Set[TrackKey] = set() # Keys needing second pass

        # --- 2. Calculate Batched Similarity ---
        similarity_matrix = None
        if gallery_embeddings_list:
            gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32)
            try:
                # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
                distance_matrix = cdist(query_embeddings_np, gallery_embeddings_np, metric='cosine')
                similarity_matrix = 1.0 - distance_matrix
                # Clip similarity: max 1.0, min can be -1.0 if vectors are opposite
                similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
                logger.debug(f"Calculated similarity matrix shape: {similarity_matrix.shape}")
            except Exception as e:
                logger.error(f"Batched similarity calculation failed: {e}", exc_info=True)
                similarity_matrix = None # Fallback needed

        # --- 3. Initial Association Pass (using matrix if available) ---
        if similarity_matrix is not None:
            best_match_indices = np.argmax(similarity_matrix, axis=1) # Index of best match in gallery for each query
            max_similarity_scores = similarity_matrix[np.arange(len(query_track_keys)), best_match_indices]

            for i, track_key in enumerate(query_track_keys):
                best_gallery_idx = best_match_indices[i]
                max_sim = max_similarity_scores[i]
                assigned_gid: Optional[GlobalID] = None

                if max_sim >= self.config.reid_similarity_threshold:
                    matched_gid = gallery_gids[best_gallery_idx]
                    matched_type = gallery_types[best_gallery_idx]

                    # Handoff Filtering Check (applied *after* finding best match)
                    perform_assignment = True
                    trigger_info = active_triggers_map.get(track_key)
                    if trigger_info:
                        target_cam_id = trigger_info.rule.target_cam_id
                        relevant_cams = self._get_relevant_handoff_cams(target_cam_id)
                        last_seen_cam = self.global_id_last_seen_cam.get(matched_gid)
                        # If last seen cam is known AND it's NOT in the relevant set, filter out match
                        if last_seen_cam is not None and last_seen_cam not in relevant_cams:
                            perform_assignment = False
                            logger.debug(f"Handoff Filter: Ignoring match GID {matched_gid} (last seen {last_seen_cam}) for {track_key} targeting {target_cam_id} (relevant: {relevant_cams}). Sim: {max_sim:.3f}")


                    if perform_assignment:
                        assigned_gid = matched_gid
                        logger.info(f"Tentative Match: {track_key} -> GID {assigned_gid} ({matched_type}) (Sim: {max_sim:.3f})")
                        # Handle gallery updates later during conflict resolution if assignment holds

                # Store tentative result (might be None if below threshold or filtered)
                tentative_assignments[track_key] = (assigned_gid, max_sim if assigned_gid is not None else -1.0)

        # Handle cases where similarity matrix calculation failed or no gallery existed
        # Or assign new IDs if no match was found above threshold / filtered out
        for i, track_key in enumerate(query_track_keys):
             if track_key not in tentative_assignments or tentative_assignments[track_key][0] is None:
                  # Assign New Global ID
                  new_gid = self.next_global_id; self.next_global_id += 1
                  self.reid_gallery[new_gid] = query_embeddings_np[i] # Add normalized embedding
                  tentative_assignments[track_key] = (new_gid, -1.0) # No match score
                  logger.info(f"Assigned NEW Global ID {new_gid} to {track_key}")


        # --- 4. Conflict Resolution ---
        assignments_by_cam: Dict[CameraID, Dict[GlobalID, List[Tuple[TrackID, float]]]] = defaultdict(lambda: defaultdict(list))
        for track_key, (gid, score) in tentative_assignments.items():
            if gid is not None:
                cam_id, track_id = track_key
                assignments_by_cam[cam_id][gid].append((track_id, score))

        reverted_keys: Set[TrackKey] = set()
        for cam_id, gid_map in assignments_by_cam.items():
            for gid, track_score_list in gid_map.items():
                if len(track_score_list) > 1: # Conflict
                    track_score_list.sort(key=lambda x: x[1], reverse=True)
                    best_track_id, best_score = track_score_list[0]
                    logger.warning(f"[{cam_id}] Conflict for GID {gid}. Tracks: {track_score_list}. Keeping T:{best_track_id} (Score: {best_score:.3f}).")
                    for i in range(1, len(track_score_list)):
                         reverted_track_id, reverted_score = track_score_list[i]
                         reverted_keys.add((cam_id, reverted_track_id))
                         logger.warning(f"[{cam_id}]   -> Reverting T:{reverted_track_id} (Score: {reverted_score:.3f}).")

        # --- 5. Finalize Non-Reverted Assignments & Update State ---
        for track_key, (gid, score) in tentative_assignments.items():
             if track_key not in reverted_keys and gid is not None:
                 final_assignments[track_key] = gid
                 # Update state mappings
                 self.track_to_global_id[track_key] = gid
                 self.global_id_last_seen_cam[gid] = track_key[0] # cam_id

                 # Update gallery (EMA) if it was a match (not a new ID)
                 if score > 0: # Check if it was a match (score > -1.0)
                     query_idx = query_track_keys.index(track_key)
                     new_embedding = query_embeddings_np[query_idx] # Already normalized

                     # Check if it came from the lost gallery originally
                     original_gallery_idx = -1
                     try: original_gallery_idx = gallery_gids.index(gid)
                     except ValueError: pass # GID not found in original list (shouldn't happen for matched IDs)

                     if original_gallery_idx != -1 and gallery_types[original_gallery_idx] == 'lost':
                         # Move from lost gallery to main, update embedding
                         lost_embedding, _ = self.lost_track_gallery.pop(gid)
                         updated_embedding = (self.config.gallery_ema_alpha * normalize_embedding(lost_embedding) +
                                              (1.0 - self.config.gallery_ema_alpha) * new_embedding)
                         self.reid_gallery[gid] = normalize_embedding(updated_embedding)
                         logger.debug(f"Moved GID {gid} from lost to main gallery, updated EMA.")
                     elif gid in self.reid_gallery: # If it was a match to the main gallery
                         current_gallery_emb = self.reid_gallery[gid] # Already normalized
                         updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb +
                                              (1.0 - self.config.gallery_ema_alpha) * new_embedding)
                         self.reid_gallery[gid] = normalize_embedding(updated_embedding)
                         logger.debug(f"Updated main gallery EMA for GID {gid}.")
                    # Else: it was a newly assigned ID, already added to gallery earlier.

        # --- 6. Handle Reverted Tracks (Second Pass - Simplified: Recalculate vs Filtered Gallery) ---
        # This part still benefits from the initial filtering done by the first pass winner
        for reverted_key in reverted_keys:
            cam_id, reverted_track_id = reverted_key
            logger.info(f"Attempting second pass matching for reverted track {reverted_key}...")

            original_tentative_assignment = tentative_assignments.get(reverted_key)
            conflicting_gid: Optional[GlobalID] = original_tentative_assignment[0] if original_tentative_assignment else None

            query_idx = query_track_keys.index(reverted_key)
            reverted_embedding_normalized = query_embeddings_np[query_idx] # Already normalized

            second_pass_assigned_gid: Optional[GlobalID] = None
            max_second_pass_score: float = -1.0

            # Prepare filtered gallery (excluding conflicting GID)
            filtered_gallery_gids: List[GlobalID] = []
            filtered_gallery_embeddings: List[FeatureVector] = []
            filtered_gallery_types: List[str] = []

            # Add lost (excluding conflict)
            for gid, (emb, _) in self.lost_track_gallery.items():
                 if gid != conflicting_gid and emb is not None and np.isfinite(emb).all() and emb.size > 0:
                     filtered_gallery_gids.append(gid)
                     filtered_gallery_embeddings.append(normalize_embedding(emb))
                     filtered_gallery_types.append('lost')
            num_lost_filtered = len(filtered_gallery_gids)

            # Add main (excluding conflict)
            for gid, emb in self.reid_gallery.items():
                 if gid != conflicting_gid and emb is not None and np.isfinite(emb).all() and emb.size > 0:
                      filtered_gallery_gids.append(gid)
                      filtered_gallery_embeddings.append(normalize_embedding(emb)) # Assumes already normalized, but redo for safety
                      filtered_gallery_types.append('main')

            # Perform similarity check against filtered gallery
            if filtered_gallery_embeddings:
                 filtered_gallery_np = np.array(filtered_gallery_embeddings, dtype=np.float32)
                 try:
                     # Calculate similarity vs this single reverted embedding
                     sims_filtered = 1.0 - cdist(reverted_embedding_normalized.reshape(1, -1), filtered_gallery_np, metric='cosine')
                     sims_filtered = np.clip(sims_filtered.flatten(), -1.0, 1.0) # Flatten to 1D array

                     if sims_filtered.size > 0:
                         best_match_idx_f = np.argmax(sims_filtered)
                         max_sim_f = sims_filtered[best_match_idx_f]

                         if max_sim_f >= self.config.reid_similarity_threshold:
                             second_pass_assigned_gid = filtered_gallery_gids[best_match_idx_f]
                             max_second_pass_score = max_sim_f
                             matched_type_f = filtered_gallery_types[best_match_idx_f]
                             logger.info(f"Second Pass: Re-associating {reverted_key} with {matched_type_f.upper()} GID {second_pass_assigned_gid} (Sim: {max_second_pass_score:.3f}).")

                             # Update gallery (EMA)
                             if matched_type_f == 'lost':
                                 lost_embedding, _ = self.lost_track_gallery.pop(second_pass_assigned_gid)
                                 updated_embedding = (self.config.gallery_ema_alpha * normalize_embedding(lost_embedding) +
                                                      (1.0 - self.config.gallery_ema_alpha) * reverted_embedding_normalized)
                                 self.reid_gallery[second_pass_assigned_gid] = normalize_embedding(updated_embedding)
                             elif second_pass_assigned_gid in self.reid_gallery: # Matched main
                                 current_gallery_emb = self.reid_gallery[second_pass_assigned_gid]
                                 updated_embedding = (self.config.gallery_ema_alpha * current_gallery_emb +
                                                      (1.0 - self.config.gallery_ema_alpha) * reverted_embedding_normalized)
                                 self.reid_gallery[second_pass_assigned_gid] = normalize_embedding(updated_embedding)

                 except Exception as e_2pass:
                      logger.error(f"Error during second pass similarity check for {reverted_key}: {e_2pass}")

            # Fallback to New ID if Second Pass Failed
            if second_pass_assigned_gid is None:
                logger.info(f"Second pass failed for {reverted_key}. Assigning new ID.")
                new_gid = self.next_global_id; self.next_global_id += 1
                self.reid_gallery[new_gid] = reverted_embedding_normalized # Use the normalized embedding
                second_pass_assigned_gid = new_gid

            # Update Final State for Reverted Key
            final_assignments[reverted_key] = second_pass_assigned_gid
            self.track_to_global_id[reverted_key] = second_pass_assigned_gid
            self.global_id_last_seen_cam[second_pass_assigned_gid] = cam_id


        # Return the dictionary reflecting the FINAL assignments after conflict resolution
        return final_assignments


    def _update_and_cleanup_state(self, current_frame_active_track_keys: Set[TrackKey]):
        """
        Updates the set of last seen tracks. Moves disappeared tracks' info to a temporary
        'lost_track_gallery' and purges expired entries from it. Cleans up other state.
        """
        # --- Function remains the same conceptually ---
        all_previous_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previous_track_keys - current_frame_active_track_keys

        for track_key in disappeared_track_keys:
            global_id = self.track_to_global_id.pop(track_key, None)
            if global_id is not None:
                last_feature = self.reid_gallery.get(global_id)
                if last_feature is not None:
                    if global_id not in self.lost_track_gallery:
                        self.lost_track_gallery[global_id] = (last_feature, self.processed_frame_counter)
                        logger.info(f"Track {track_key} (GID {global_id}) disappeared. Moved to lost gallery (frame {self.processed_frame_counter}).")
                    else: # Update timestamp if already lost
                        existing_feat, _ = self.lost_track_gallery[global_id]
                        self.lost_track_gallery[global_id] = (existing_feat, self.processed_frame_counter)
                        logger.debug(f"Track {track_key} (GID {global_id}) disappeared again. Updated lost timestamp.")
                # else: # Log if feature missing
            # Clean up other state
            self.track_last_reid_frame.pop(track_key, None)
            # Don't remove from global_id_last_seen_cam here, needed for handoff filtering

        # Purge expired lost tracks
        expired_lost_gids = [gid for gid, (_, frame_num) in self.lost_track_gallery.items()
                             if (self.processed_frame_counter - frame_num) > self.config.lost_track_buffer_frames]
        for gid in expired_lost_gids:
            if gid in self.lost_track_gallery: del self.lost_track_gallery[gid]
            if gid in self.global_id_last_seen_cam: del self.global_id_last_seen_cam[gid] # Clean up last seen if expired
            # Optionally remove from main gallery too if desired (currently not)
            # if gid in self.reid_gallery: del self.reid_gallery[gid]
            logger.info(f"Purged expired GID {gid} from lost gallery and associated state.")


        # Update the set of track IDs seen *in this frame* for each camera
        new_last_seen: Dict[CameraID, Set[TrackID]] = defaultdict(set)
        for cam_id, track_id in current_frame_active_track_keys:
            new_last_seen[cam_id].add(track_id)
        self.last_seen_track_ids = new_last_seen

        # Clean up track_last_reid_frame for tracks that no longer exist (redundant due to pop above, but safe)
        keys_to_delete_reid = set(self.track_last_reid_frame.keys()) - current_frame_active_track_keys
        for key in keys_to_delete_reid: self.track_last_reid_frame.pop(key, None)


    def process_frame_batch_full(self, frames: Dict[CameraID, FrameData], frame_idx_global: int) -> ProcessedBatchResult:
        """Processes a batch of frames: Detect -> Track -> Handoff Check -> Re-ID -> Project -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)
        all_handoff_triggers_this_batch: List[HandoffTriggerInfo] = [] # Collect triggers across all cams

        self.processed_frame_counter += 1
        proc_frame_id = self.processed_frame_counter

        # --- Stage 1a: Preprocess ---
        t_prep_start = time.time()
        batch_input_tensors: List[torch.Tensor] = []
        batch_cam_ids: List[CameraID] = []
        batch_original_shapes: List[Tuple[int, int]] = []
        batch_scale_factors: List[ScaleFactors] = []
        for cam_id, frame_bgr in frames.items():
            if frame_bgr is not None and frame_bgr.size > 0:
                # ... (resizing and transform logic as before) ...
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
                     # ... (postprocessing logic as before) ...
                     pred_boxes = prediction_dict['boxes'].cpu().numpy(); pred_labels = prediction_dict['labels'].cpu().numpy(); pred_scores = prediction_dict['scores'].cpu().numpy()
                     for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                          if label == self.config.person_class_id and score >= self.config.detection_confidence_threshold:
                              x1, y1, x2, y2 = box; orig_x1 = max(0.0, x1 * scale_x); orig_y1 = max(0.0, y1 * scale_y); orig_x2 = min(float(original_w - 1), x2 * scale_x); orig_y2 = min(float(original_h - 1), y2 * scale_y)
                              if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1: bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32); detections_per_camera[cam_id].append({'bbox_xyxy': bbox_orig, 'conf': float(score), 'class_id': int(label)})
                 except Exception as postproc_err: logger.error(f"[{cam_id}] Error postprocessing detections: {postproc_err}", exc_info=False)
        # ... (mismatch error log as before) ...
        timings['postprocess_scale'] = time.time() - t_postproc_start

        # --- Stage 1d & 1e: Tracking & Handoff Check (Interleaved per camera) ---
        t_track_handoff_start = time.time()
        current_frame_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        active_track_keys_this_frame: Set[TrackKey] = set() # Collect all active keys
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo] = {} # Map for ReID

        for cam_id in self.camera_ids: # Iterate through all configured cameras
            cam_config = self.config.cameras_config.get(cam_id)
            if not cam_config: continue # Skip if config missing (shouldn't happen after setup)

            tracker = self.trackers.get(cam_id)
            if not tracker: logger.warning(f"[{cam_id}] Tracker instance missing."); current_frame_tracker_outputs[cam_id] = np.empty((0, 8)); continue

            # Tracking
            cam_detections = detections_per_camera.get(cam_id, []); np_dets = np.empty((0, 6))
            if cam_detections:
                try: np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections])
                except Exception as format_err: logger.error(f"[{cam_id}] Failed to format detections for tracker: {format_err}", exc_info=False)

            original_frame_bgr = frames.get(cam_id)
            frame_shape_orig = cam_config.frame_shape
            # Use dummy frame if original is missing, needed by some trackers
            dummy_frame_shape = frame_shape_orig if frame_shape_orig else (1080, 1920);
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*dummy_frame_shape, 3), dtype=np.uint8)

            tracked_dets_np = np.empty((0, 8)) # Expecting [x1, y1, x2, y2, tid, conf, cls, idx]
            try:
                tracked_dets_list = tracker.update(np_dets, dummy_frame)
                if tracked_dets_list is not None and len(tracked_dets_list) > 0:
                     tracked_dets_np_maybe = np.array(tracked_dets_list)
                     # BoxMOT format often [x1,y1,x2,y2, tid, conf, cls, idx] (8 cols)
                     if tracked_dets_np_maybe.ndim == 2 and tracked_dets_np_maybe.shape[1] >= 7:
                          tracked_dets_np = tracked_dets_np_maybe[:, :8] # Take first 8 cols if available
                          # Collect active track keys for this frame
                          for track_data_row in tracked_dets_np:
                              try: active_track_keys_this_frame.add( (cam_id, int(track_data_row[4])) )
                              except (IndexError, ValueError): pass # Ignore malformed track data
                     else: logger.warning(f"[{cam_id}] Tracker output has unexpected shape {tracked_dets_np_maybe.shape}. Treating as empty.")
            except Exception as e: logger.error(f"[{cam_id}] Tracker update failed: {e}", exc_info=True)
            current_frame_tracker_outputs[cam_id] = tracked_dets_np

            # Handoff Check for this camera's tracks
            if self.config.min_bbox_overlap_ratio_in_quadrant > 0:
                triggers_this_cam = self._check_handoff_triggers(
                    cam_id, tracked_dets_np, frame_shape_orig, cam_config.exit_rules
                )
                all_handoff_triggers_this_batch.extend(triggers_this_cam)

        timings['tracking_and_handoff_check'] = time.time() - t_track_handoff_start

        # Create map for faster lookup during ReID
        for trigger in all_handoff_triggers_this_batch:
            active_triggers_map[trigger.source_track_key] = trigger
        track_keys_triggering_handoff: Set[TrackKey] = set(active_triggers_map.keys())


        # --- Stage: Decide ReID ---
        t_reid_decision_start = time.time()
        tracks_to_extract_features_for: Dict[TrackKey, np.ndarray] = {} # track_key -> full track_data row
        # Iterate using the collected active_track_keys_this_frame
        for track_key in active_track_keys_this_frame:
             cam_id, track_id = track_key
             # Find the corresponding track data row
             track_data_row = None
             cam_output = current_frame_tracker_outputs.get(cam_id)
             if cam_output is not None:
                  match = cam_output[cam_output[:, 4] == track_id]
                  if match.shape[0] > 0:
                      track_data_row = match[0] # Get the first row if multiple matches (shouldn't happen)

             if track_data_row is None:
                  logger.warning(f"Could not find track data row for active key {track_key}. Skipping ReID decision.")
                  continue

             original_frame_bgr = frames.get(cam_id)
             if original_frame_bgr is not None and original_frame_bgr.size > 0:
                  previous_processed_cam_track_ids = self.last_seen_track_ids.get(cam_id, set())
                  is_newly_seen = track_id not in previous_processed_cam_track_ids
                  last_reid_attempt_proc_idx = self.track_last_reid_frame.get(track_key, -self.config.reid_refresh_interval_frames - 1)
                  is_due_for_refresh = (proc_frame_id - last_reid_attempt_proc_idx) >= self.config.reid_refresh_interval_frames
                  is_triggering_handoff_now = track_key in track_keys_triggering_handoff
                  trigger_reid = is_newly_seen or is_due_for_refresh or is_triggering_handoff_now

                  if trigger_reid:
                      tracks_to_extract_features_for[track_key] = track_data_row
                      self.track_last_reid_frame[track_key] = proc_frame_id # Mark attempt frame

        timings['reid_decision'] = time.time() - t_reid_decision_start
        logger.debug(f"Frame {proc_frame_id}: Decided to extract ReID features for {len(tracks_to_extract_features_for)} tracks.")

        # --- Stage 2: Feature Extraction ---
        t_feat_start = time.time()
        extracted_features_this_frame: Dict[TrackKey, FeatureVector] = {}
        if tracks_to_extract_features_for:
            tracks_grouped_by_cam: Dict[CameraID, List[np.ndarray]] = defaultdict(list)
            track_keys_per_cam: Dict[CameraID, List[TrackKey]] = defaultdict(list)

            for track_key, track_data in tracks_to_extract_features_for.items():
                 cam_id, _ = track_key; tracks_grouped_by_cam[cam_id].append(track_data); track_keys_per_cam[cam_id].append(track_key)

            for cam_id, tracks_data_list in tracks_grouped_by_cam.items():
                 frame_bgr = frames.get(cam_id)
                 if frame_bgr is not None and frame_bgr.size > 0 and tracks_data_list:
                     try:
                         tracks_data_np = np.array(tracks_data_list)
                         features_this_cam: Dict[TrackID, FeatureVector] = self._extract_features_for_tracks(frame_bgr, tracks_data_np)
                         # Map features back using track_id
                         for track_id, feature in features_this_cam.items():
                             # Find the original TrackKey(s) for this track_id in this camera's batch
                             original_keys_for_tid = [tk for tk in track_keys_per_cam[cam_id] if tk[1] == track_id]
                             if original_keys_for_tid:
                                 extracted_features_this_frame[original_keys_for_tid[0]] = feature # Assume only one key per track_id in batch
                             else:
                                 logger.warning(f"[{cam_id}] Feature extracted for T:{track_id}, but couldn't map back to an original TrackKey in this batch.")
                     except Exception as fe_err: logger.error(f"[{cam_id}] Error during batched feature extraction call: {fe_err}", exc_info=False)
        timings['feature_ext'] = time.time() - t_feat_start
        logger.debug(f"Frame {proc_frame_id}: Extracted {len(extracted_features_this_frame)} features.")

        # --- Stage 3: Re-ID Association (Batched) ---
        t_reid_start = time.time()
        # Use the new batched function, pass the active triggers map
        final_assigned_global_ids_this_cycle = self._perform_reid_association_batched(
             extracted_features_this_frame,
             active_triggers_map
        )
        timings['reid'] = time.time() - t_reid_start
        logger.debug(f"Frame {proc_frame_id}: ReID association completed. Final assignments this cycle: {len(final_assigned_global_ids_this_cycle)}")


        # --- Stage 4: Combine Results, Project to Map ---
        t_project_start = time.time()
        projection_success_count = 0
        for cam_id, tracked_dets_np in current_frame_tracker_outputs.items():
             cam_config = self.config.cameras_config.get(cam_id)
             homography_matrix = cam_config.homography_matrix if cam_config else None

             logger.debug(f"[{cam_id}] Projection Check: Homography matrix is {'AVAILABLE' if homography_matrix is not None else 'MISSING'}")
             if tracked_dets_np.shape[0] > 0:
                 for track_data in tracked_dets_np:
                     # Expecting [x1,y1,x2,y2, tid, conf, cls, ...] min 7 cols
                     if len(track_data) >= 7:
                         try:
                             x1, y1, x2, y2 = map(float, track_data[0:4])
                             track_id = int(track_data[4])
                             conf = float(track_data[5])
                             cls = int(track_data[6])
                             current_track_key: TrackKey = (cam_id, track_id)

                             # Get Global ID (priority: this cycle's result -> persistent state)
                             global_id: Optional[GlobalID] = final_assigned_global_ids_this_cycle.get(
                                 current_track_key,
                                 self.track_to_global_id.get(current_track_key)
                             )

                             # Project to Map
                             map_coords: MapCoordinates = None
                             if homography_matrix is not None:
                                 image_point_x = (x1 + x2) / 2.0; image_point_y = y2 # Bottom center
                                 map_coords = project_point_to_map((image_point_x, image_point_y), homography_matrix)
                                 if map_coords: projection_success_count += 1
                                 # logger.debug(f"[{cam_id}] T:{track_id} Projection -> {map_coords}")

                             track_info_dict: TrackData = {
                                 'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                                 'track_id': track_id,
                                 'global_id': global_id,
                                 'conf': conf,
                                 'class_id': cls,
                                 'map_coords': map_coords
                             }
                             final_results_per_camera[cam_id].append(track_info_dict)
                             # logger.debug(f"[{cam_id}] T:{track_id} -> Appended final data.")

                         except (ValueError, IndexError, TypeError) as e:
                             logger.warning(f"[{cam_id}] Failed parse/project track data: {track_data}. Error: {e}", exc_info=False)

        timings['projection'] = time.time() - t_project_start
        logger.debug(f"Frame {proc_frame_id}: Projection stage finished. Successful projections: {projection_success_count}")


        # --- Stage 5: Update State and Cleanup ---
        # Use active track keys determined during tracking stage
        self._update_and_cleanup_state(active_track_keys_this_frame)

        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch

        return ProcessedBatchResult(
            results_per_camera=dict(final_results_per_camera),
            timings=dict(timings),
            processed_this_frame=True,
            handoff_triggers=all_handoff_triggers_this_batch # Return collected triggers
        )
