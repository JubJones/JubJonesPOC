"""Handles Re-ID association logic including gallery matching, conflict resolution, and EMA updates."""

import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any, Set, List

import numpy as np
from scipy.spatial.distance import cdist

# Project specific types and utils
from reid_poc.alias_types import (
    CameraID, TrackID, GlobalID, TrackKey, FeatureVector, HandoffTriggerInfo
)
from reid_poc.utils import normalize_embedding # Assuming normalize_embedding is in utils

logger = logging.getLogger(__name__)


def _calculate_similarity_matrix(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Calculates the cosine similarity matrix between query and gallery embeddings."""
    if query_embeddings.size == 0 or gallery_embeddings.size == 0:
        return None
    try:
        # Ensure embeddings are 2D
        if query_embeddings.ndim == 1:
             query_embeddings = query_embeddings.reshape(1, -1)
        if gallery_embeddings.ndim == 1:
             gallery_embeddings = gallery_embeddings.reshape(1, -1)

        distance_matrix = cdist(query_embeddings, gallery_embeddings, metric='cosine')
        similarity_matrix = 1.0 - distance_matrix
        return np.clip(similarity_matrix, -1.0, 1.0) # Clip to handle potential floating point inaccuracies
    except Exception as e:
        logger.error(f"Batched similarity calculation failed: {e}", exc_info=True)
        return None


def _prepare_reid_query_gallery(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    features_per_track: Dict[TrackKey, FeatureVector]
) -> Tuple[List[TrackKey], np.ndarray, List[GlobalID], np.ndarray, List[str]]:
    """Prepares normalized query and gallery embeddings for similarity calculation."""
    query_track_keys: List[TrackKey] = []
    query_embeddings_list: List[FeatureVector] = []
    for tk, feat in features_per_track.items():
        if feat is not None and np.isfinite(feat).all() and feat.size > 0:
            normalized_feat = normalize_embedding(feat)
            query_track_keys.append(tk)
            query_embeddings_list.append(normalized_feat)
        else:
            logger.warning(f"Skipping ReID for {tk}: Invalid embedding provided.")

    query_embeddings_np = np.array(query_embeddings_list, dtype=np.float32) if query_embeddings_list else np.empty((0,0), dtype=np.float32)

    gallery_gids: List[GlobalID] = []
    gallery_embeddings_list: List[FeatureVector] = []
    gallery_types: List[str] = [] # 'lost' or 'main'

    # Add lost gallery items first (higher priority)
    valid_lost_items = [
        (gid, data) for gid, data in pipeline_instance.lost_track_gallery.items()
        if data[0] is not None and np.isfinite(data[0]).all() and data[0].size > 0
    ]
    for gid, (emb, _) in valid_lost_items:
        gallery_gids.append(gid)
        gallery_embeddings_list.append(normalize_embedding(emb)) # Ensure gallery is normalized
        gallery_types.append('lost')

    # Add main gallery items
    valid_main_items = [
        (gid, emb) for gid, emb in pipeline_instance.reid_gallery.items()
        if emb is not None and np.isfinite(emb).all() and emb.size > 0
    ]
    for gid, emb in valid_main_items:
            # Avoid adding if already present from the lost gallery (can happen briefly)
        if gid not in gallery_gids[:len(valid_lost_items)]:
            gallery_gids.append(gid)
            gallery_embeddings_list.append(normalize_embedding(emb)) # Ensure gallery is normalized
            gallery_types.append('main')

    gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32) if gallery_embeddings_list else np.empty((0,0), dtype=np.float32)

    return query_track_keys, query_embeddings_np, gallery_gids, gallery_embeddings_np, gallery_types


def _get_relevant_handoff_cams(pipeline_instance: Any, target_cam_id: CameraID) -> Set[CameraID]:
    """Gets the target camera and any cameras configured to possibly overlap with it."""
    relevant_cams = {target_cam_id}
    # possible_overlaps_normalized is expected to be a set of tuples like {(cam1, cam2), (cam2, cam3)}
    for c1, c2 in pipeline_instance.config.possible_overlaps: # Use normalized overlaps from config
        if c1 == target_cam_id: relevant_cams.add(c2)
        elif c2 == target_cam_id: relevant_cams.add(c1)
    return relevant_cams


def _apply_handoff_filter(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    track_key: TrackKey,
    matched_gid: GlobalID,
    active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
) -> bool:
    """Checks if a potential ReID match should be filtered based on handoff context."""
    trigger_info = active_triggers_map.get(track_key)
    if not trigger_info:
        return True # No active trigger, don't filter

    target_cam_id = trigger_info.rule.target_cam_id
    relevant_cams = _get_relevant_handoff_cams(pipeline_instance, target_cam_id)
    last_seen_cam = pipeline_instance.global_id_last_seen_cam.get(matched_gid)

    if last_seen_cam is not None and last_seen_cam not in relevant_cams:
        logger.debug(f"Handoff Filter: Ignoring match GID {matched_gid} (last seen {last_seen_cam}) for {track_key} targeting {target_cam_id} (relevant: {relevant_cams}).")
        return False # Filter out this match
    return True # Allow assignment


def _perform_initial_reid_assignments(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    similarity_matrix: Optional[np.ndarray],
    query_track_keys: List[TrackKey],
    gallery_gids: List[GlobalID],
    gallery_types: List[str],
    active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
) -> Dict[TrackKey, Tuple[Optional[GlobalID], float]]:
    """Performs the initial matching based on similarity scores and applies handoff filters."""
    tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float]] = {} # Store (gid, score)

    if similarity_matrix is None or similarity_matrix.size == 0 or not query_track_keys:
        return tentative_assignments # Cannot perform assignments

    best_match_indices = np.argmax(similarity_matrix, axis=1)
    max_similarity_scores = similarity_matrix[np.arange(len(query_track_keys)), best_match_indices]

    for i, track_key in enumerate(query_track_keys):
        best_gallery_idx = best_match_indices[i]
        max_sim = max_similarity_scores[i]
        assigned_gid: Optional[GlobalID] = None

        if max_sim >= pipeline_instance.config.reid_similarity_threshold:
            matched_gid = gallery_gids[best_gallery_idx]
            matched_type = gallery_types[best_gallery_idx]

            # Apply handoff filtering if necessary
            if _apply_handoff_filter(pipeline_instance, track_key, matched_gid, active_triggers_map):
                assigned_gid = matched_gid
                logger.info(f"Tentative Match: {track_key} -> GID {assigned_gid} ({matched_type}) (Sim: {max_sim:.3f})")
            # If filtered, assigned_gid remains None

        tentative_assignments[track_key] = (assigned_gid, max_sim if assigned_gid is not None else -1.0)

    return tentative_assignments


def _assign_new_global_ids(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float]],
    query_track_keys: List[TrackKey],
    query_embeddings_np: np.ndarray
) -> Dict[TrackKey, Tuple[Optional[GlobalID], float]]:
    """Assigns new Global IDs to tracks that didn't get a tentative assignment."""
    updated_assignments = tentative_assignments.copy()
    query_key_to_index = {key: i for i, key in enumerate(query_track_keys)}

    for track_key in query_track_keys:
            # Check if track_key exists and has no assigned GID (is None)
        if track_key not in updated_assignments or updated_assignments[track_key][0] is None:
            new_gid = pipeline_instance.next_global_id
            pipeline_instance.next_global_id += 1

            query_idx = query_key_to_index.get(track_key)
            if query_idx is not None and query_idx < len(query_embeddings_np):
                # Add the *normalized* embedding directly to the main gallery
                pipeline_instance.reid_gallery[new_gid] = query_embeddings_np[query_idx]
                updated_assignments[track_key] = (new_gid, -1.0) # Assign new GID, score is not applicable
                pipeline_instance.global_id_last_seen_frame[new_gid] = pipeline_instance.processed_frame_counter # Mark when it was created
                logger.info(f"Assigned NEW Global ID {new_gid} to {track_key}")
            else:
                    logger.error(f"Could not find query embedding for {track_key} when assigning new GID {new_gid}. Skipping gallery add.")
                    updated_assignments[track_key] = (None, -1.0) # Cannot assign if embedding is missing

    return updated_assignments


def _resolve_reid_conflicts(
        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float]]
) -> Tuple[Dict[TrackKey, Tuple[Optional[GlobalID], float]], Set[TrackKey]]:
    """Resolves conflicts where multiple tracks in the same camera are assigned the same Global ID."""
    assignments_by_cam: Dict[CameraID, Dict[GlobalID, List[Tuple[TrackID, float]]]] = defaultdict(lambda: defaultdict(list))
    final_tentative = tentative_assignments.copy()
    reverted_keys: Set[TrackKey] = set()

    # Group assignments by camera and global ID
    for track_key, (gid, score) in tentative_assignments.items():
        if gid is not None: # Only consider assigned GIDs
            cam_id, track_id = track_key
            assignments_by_cam[cam_id][gid].append((track_id, score))

    # Resolve conflicts within each camera
    for cam_id, gid_map in assignments_by_cam.items():
        for gid, track_score_list in gid_map.items():
            if len(track_score_list) > 1: # Conflict detected
                # Sort by score (descending) to find the best match
                track_score_list.sort(key=lambda x: x[1], reverse=True)
                best_track_id, best_score = track_score_list[0]
                logger.warning(f"[{cam_id}] Conflict for GID {gid}. Tracks: {[(tid, f'{s:.3f}') for tid, s in track_score_list]}. Keeping T:{best_track_id}.")

                # Revert assignments for all conflicting tracks except the best one
                for i in range(1, len(track_score_list)):
                    reverted_track_id, reverted_score = track_score_list[i]
                    reverted_key = (cam_id, reverted_track_id)
                    reverted_keys.add(reverted_key)
                    final_tentative[reverted_key] = (None, -1.0) # Mark as unassigned for now
                    logger.warning(f"[{cam_id}] -> Reverting T:{reverted_track_id} (Score: {reverted_score:.3f}). Will attempt second pass.")

    return final_tentative, reverted_keys


def _update_gallery_ema(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    gid: GlobalID,
    new_embedding: FeatureVector,
    source_track_key: TrackKey,
    matched_type: str
):
    """Updates the gallery embedding for a GID using Exponential Moving Average (EMA)."""
    # Ensure the new embedding is normalized (should be already, but double-check)
    new_embedding = normalize_embedding(new_embedding)
    alpha = pipeline_instance.config.gallery_ema_alpha

    if matched_type == 'lost':
        if gid in pipeline_instance.lost_track_gallery:
            lost_embedding, _ = pipeline_instance.lost_track_gallery.pop(gid) # Remove from lost gallery
            # Combine lost embedding with new one using EMA
            updated_embedding = (alpha * normalize_embedding(lost_embedding) + (1.0 - alpha) * new_embedding)
            pipeline_instance.reid_gallery[gid] = normalize_embedding(updated_embedding) # Add/update in main gallery
            logger.debug(f"Moved GID {gid} from lost to main gallery, updated EMA from {source_track_key}.")
        elif gid in pipeline_instance.reid_gallery:
                # GID was already moved/reappeared quickly, update main gallery directly
                current_gallery_emb = pipeline_instance.reid_gallery[gid]
                updated_embedding = (alpha * current_gallery_emb + (1.0 - alpha) * new_embedding)
                pipeline_instance.reid_gallery[gid] = normalize_embedding(updated_embedding)
                logger.debug(f"GID {gid} already in main gallery; updated main gallery EMA with {source_track_key}.")
        else:
            # Edge case: Matched 'lost' but not found anywhere. Add directly to main.
            logger.warning(f"GID {gid} matched as 'lost' but not found. Adding new embedding from {source_track_key} directly to main gallery.")
            pipeline_instance.reid_gallery[gid] = new_embedding
    elif matched_type == 'main':
        if gid in pipeline_instance.reid_gallery:
            current_gallery_emb = pipeline_instance.reid_gallery[gid]
            updated_embedding = (alpha * current_gallery_emb + (1.0 - alpha) * new_embedding)
            pipeline_instance.reid_gallery[gid] = normalize_embedding(updated_embedding)
            logger.debug(f"Updated main gallery EMA for GID {gid} using {source_track_key}.")
        else:
            # Edge case: Matched 'main' but GID disappeared from gallery (e.g., pruned?). Add directly.
            logger.warning(f"GID {gid} matched as 'main' but not found in reid_gallery. Adding new embedding from {source_track_key} directly.")
            pipeline_instance.reid_gallery[gid] = new_embedding


def _finalize_reid_assignments(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    final_tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float]],
    reverted_keys: Set[TrackKey],
    query_track_keys: List[TrackKey],
    query_embeddings_np: np.ndarray,
    gallery_gids: List[GlobalID],
    gallery_types: List[str]
) -> Dict[TrackKey, GlobalID]:
    """Finalizes assignments for non-reverted tracks and updates global state and galleries."""
    final_assignments: Dict[TrackKey, GlobalID] = {}
    query_key_to_index = {key: i for i, key in enumerate(query_track_keys)}
    gallery_gid_to_info = {gid: (idx, gtype) for idx, (gid, gtype) in enumerate(zip(gallery_gids, gallery_types))}

    for track_key, (gid, score) in final_tentative_assignments.items():
        if track_key in reverted_keys or gid is None:
            continue # Skip reverted keys or those finally unassigned

        # Assign GID and update tracking state
        final_assignments[track_key] = gid
        pipeline_instance.track_to_global_id[track_key] = gid
        pipeline_instance.global_id_last_seen_cam[gid] = track_key[0] # Update last seen camera
        pipeline_instance.global_id_last_seen_frame[gid] = pipeline_instance.processed_frame_counter # Update last seen frame

        # Update gallery via EMA if it was a successful match (not a newly assigned ID)
        if score >= pipeline_instance.config.reid_similarity_threshold:
            query_idx = query_key_to_index.get(track_key)
            gallery_info = gallery_gid_to_info.get(gid)

            if query_idx is not None and gallery_info is not None and query_idx < len(query_embeddings_np):
                new_embedding = query_embeddings_np[query_idx]
                matched_type = gallery_info[1] # 'lost' or 'main'
                _update_gallery_ema(pipeline_instance, gid, new_embedding, track_key, matched_type)
            elif query_idx is None:
                    logger.error(f"Cannot update gallery for GID {gid}: Query embedding not found for {track_key}.")
            elif gallery_info is None:
                    logger.warning(f"Cannot update gallery for GID {gid} from {track_key}: GID not found in original gallery map (might be a very new ID?).")

    return final_assignments


def _handle_reverted_reid_assignments(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    reverted_keys: Set[TrackKey],
    query_track_keys: List[TrackKey],
    query_embeddings_np: np.ndarray,
    final_assignments: Dict[TrackKey, GlobalID],
    active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> Dict[TrackKey, GlobalID]:
    """Attempts a second ReID pass for tracks whose initial assignments were reverted due to conflicts."""
    updated_assignments = final_assignments.copy()
    query_key_to_index = {key: i for i, key in enumerate(query_track_keys)}

    for reverted_key in reverted_keys:
        logger.info(f"Attempting second pass matching for reverted track {reverted_key}...")
        cam_id, reverted_track_id = reverted_key
        query_idx = query_key_to_index.get(reverted_key)

        if query_idx is None or query_idx >= len(query_embeddings_np):
            logger.error(f"Could not find embedding for reverted key {reverted_key} during second pass. Assigning new ID.")
            new_gid = pipeline_instance.next_global_id; pipeline_instance.next_global_id += 1
            updated_assignments[reverted_key] = new_gid
            pipeline_instance.track_to_global_id[reverted_key] = new_gid # Update state
            pipeline_instance.global_id_last_seen_cam[new_gid] = cam_id
            pipeline_instance.global_id_last_seen_frame[new_gid] = pipeline_instance.processed_frame_counter
            # Cannot add to gallery without embedding
            continue

        reverted_embedding_normalized = query_embeddings_np[query_idx]

        # Re-prepare gallery (could be optimized by reusing parts, but this ensures current state)
        _, _, gallery_gids_2pass, gallery_embeddings_2pass, gallery_types_2pass = _prepare_reid_query_gallery(pipeline_instance, {}) # Pass empty features dict to just get current gallery

        second_pass_assigned_gid: Optional[GlobalID] = None
        max_second_pass_score: float = -1.0
        matched_type_2pass = 'unknown'

        if gallery_embeddings_2pass.size > 0:
                similarity_matrix_2pass = _calculate_similarity_matrix(reverted_embedding_normalized.reshape(1, -1), gallery_embeddings_2pass)

                if similarity_matrix_2pass is not None and similarity_matrix_2pass.size > 0:
                    best_match_idx_2pass = np.argmax(similarity_matrix_2pass)
                    max_sim_2pass = similarity_matrix_2pass[0, best_match_idx_2pass]

                    if max_sim_2pass >= pipeline_instance.config.reid_similarity_threshold:
                        potential_gid = gallery_gids_2pass[best_match_idx_2pass]
                        potential_type = gallery_types_2pass[best_match_idx_2pass]

                        # Check if this GID is already assigned to the winning track in the same camera
                        is_assigned_to_winner = False
                        for tk, assigned_gid in updated_assignments.items():
                            if tk[0] == cam_id and assigned_gid == potential_gid and tk != reverted_key:
                                is_assigned_to_winner = True
                                logger.warning(f"Second pass for {reverted_key} found GID {potential_gid}, but it's already assigned to {tk} in this camera. Ignoring.")
                                break

                        # Apply handoff filter again for the potential new match
                        if not is_assigned_to_winner and _apply_handoff_filter(pipeline_instance, reverted_key, potential_gid, active_triggers_map):
                            second_pass_assigned_gid = potential_gid
                            max_second_pass_score = max_sim_2pass
                            matched_type_2pass = potential_type
                            logger.info(f"Second Pass SUCCESS: Re-associating {reverted_key} with {matched_type_2pass.upper()} GID {second_pass_assigned_gid} (Sim: {max_second_pass_score:.3f}).")


        # Finalize assignment for the reverted key
        if second_pass_assigned_gid is not None:
            final_gid = second_pass_assigned_gid
            # Update gallery via EMA for the successful second pass match
            _update_gallery_ema(pipeline_instance, final_gid, reverted_embedding_normalized, reverted_key, matched_type_2pass)
        else:
            # Assign a new ID if second pass failed or yielded no valid match
            logger.info(f"Second pass failed for {reverted_key} or no match above threshold/passed filters. Assigning new ID.")
            new_gid = pipeline_instance.next_global_id; pipeline_instance.next_global_id += 1
            pipeline_instance.reid_gallery[new_gid] = reverted_embedding_normalized # Add embedding to gallery
            pipeline_instance.global_id_last_seen_frame[new_gid] = pipeline_instance.processed_frame_counter
            final_gid = new_gid

        # Update state for the reverted key with its final GID
        updated_assignments[reverted_key] = final_gid
        pipeline_instance.track_to_global_id[reverted_key] = final_gid
        pipeline_instance.global_id_last_seen_cam[final_gid] = cam_id
        pipeline_instance.global_id_last_seen_frame[final_gid] = pipeline_instance.processed_frame_counter

    return updated_assignments


def associate_reid_batched(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    extracted_features: Dict[TrackKey, FeatureVector],
    active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
) -> Dict[TrackKey, GlobalID]:
    """Performs batched Re-ID association, conflict resolution, and state updates."""
    if not extracted_features:
        return {}

    # 1. Prepare query and gallery embeddings
    query_keys, query_embeds, gal_gids, gal_embeds, gal_types = _prepare_reid_query_gallery(pipeline_instance, extracted_features)
    if query_embeds.size == 0:
        logger.debug("No valid query embeddings for ReID association.")
        # Assign new IDs to all tracks that had features extracted but failed preparation
        final_assignments = {}
        for tk in extracted_features.keys():
                # Check if it's already known from previous frames
                if tk not in pipeline_instance.track_to_global_id:
                    new_gid = pipeline_instance.next_global_id; pipeline_instance.next_global_id += 1
                    # Cannot add to gallery as embedding preparation failed
                    pipeline_instance.global_id_last_seen_frame[new_gid] = pipeline_instance.processed_frame_counter
                    final_assignments[tk] = new_gid
                    pipeline_instance.track_to_global_id[tk] = new_gid
                    pipeline_instance.global_id_last_seen_cam[new_gid] = tk[0]
                    logger.warning(f"Assigning new GID {new_gid} to {tk} due to failed embedding prep, cannot add to gallery.")
                else:
                    # If already known, try to return existing assignment
                    existing_gid = pipeline_instance.track_to_global_id.get(tk)
                    if existing_gid is not None:
                         final_assignments[tk] = existing_gid
        return final_assignments


    # 2. Calculate Similarity Matrix
    similarity_matrix = _calculate_similarity_matrix(query_embeds, gal_embeds)

    # 3. Initial Assignments (with Handoff Filter)
    tentative_assignments = _perform_initial_reid_assignments(
        pipeline_instance, similarity_matrix, query_keys, gal_gids, gal_types, active_triggers_map
    )

    # 4. Assign New IDs to Unmatched Tracks
    tentative_assignments_with_new = _assign_new_global_ids(
        pipeline_instance, tentative_assignments, query_keys, query_embeds
    )

    # 5. Resolve Conflicts (Within Camera)
    final_tentative_assignments, reverted_keys = _resolve_reid_conflicts(
            tentative_assignments_with_new
    )

    # 6. Finalize Non-Reverted Assignments & Update State/Gallery
    final_assignments = _finalize_reid_assignments(
        pipeline_instance, final_tentative_assignments, reverted_keys, query_keys, query_embeds, gal_gids, gal_types
    )

    # 7. Handle Reverted Tracks (Second Pass Matching)
    final_assignments_after_revert = _handle_reverted_reid_assignments(
        pipeline_instance, reverted_keys, query_keys, query_embeds, final_assignments, active_triggers_map
    )

    # The function implicitly updated the pipeline_instance.track_to_global_id map
    # Return value isn't strictly needed if state mutation is intended, but can be useful.
    return final_assignments_after_revert