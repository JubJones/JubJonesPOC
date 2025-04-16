# -*- coding: utf-8 -*-
"""Handles pipeline state updates, gallery management, and cleanup."""

import logging
from typing import Dict, Tuple, Optional, Any, Set, List

import numpy as np

# Project specific types
from reid_poc.alias_types import (
    CameraID, TrackID, GlobalID, TrackKey, FeatureVector
)

logger = logging.getLogger(__name__)


def _handle_disappeared_tracks(pipeline_instance: Any, current_active_track_keys: Set[TrackKey]):
    """Moves tracks that disappeared in the current frame to the lost gallery."""
    all_previous_track_keys = set(pipeline_instance.track_to_global_id.keys())
    disappeared_track_keys = all_previous_track_keys - current_active_track_keys
    current_proc_frame = pipeline_instance.processed_frame_counter

    for track_key in disappeared_track_keys:
        global_id = pipeline_instance.track_to_global_id.pop(track_key, None) # Remove from active mapping
        if global_id is not None:
            # Try to get the last known feature from the main gallery
            last_feature = pipeline_instance.reid_gallery.get(global_id)
            if last_feature is not None:
                # Add to lost gallery only if not already there (handles brief disappearances)
                if global_id not in pipeline_instance.lost_track_gallery:
                        # Use the last frame it was seen actively (recorded before this function)
                    last_active_frame = pipeline_instance.global_id_last_seen_frame.get(global_id, current_proc_frame) # Default to current frame if missing
                    pipeline_instance.lost_track_gallery[global_id] = (last_feature, last_active_frame)
                    logger.info(f"Track {track_key} (GID {global_id}) disappeared. Moved to lost gallery (timestamped frame {last_active_frame}).")
            else:
                logger.warning(f"Track {track_key} (GID {global_id}) disappeared, but no feature found in main gallery to move to lost.")
        # Clean up associated state regardless
        pipeline_instance.track_last_reid_frame.pop(track_key, None)


def _purge_lost_gallery(pipeline_instance: Any):
    """Removes tracks from the lost gallery that have exceeded the buffer time."""
    current_proc_frame = pipeline_instance.processed_frame_counter
    buffer_frames = pipeline_instance.config.lost_track_buffer_frames
    expired_lost_gids = [
        gid for gid, (_, frame_num_added) in pipeline_instance.lost_track_gallery.items()
        if (current_proc_frame - frame_num_added) > buffer_frames
    ]

    if expired_lost_gids:
        logger.info(f"Purging {len(expired_lost_gids)} expired GIDs from lost gallery (Buffer: {buffer_frames} frames).")
        for gid in expired_lost_gids:
            pipeline_instance.lost_track_gallery.pop(gid, None)


def _prune_main_gallery(pipeline_instance: Any):
    """Periodically removes inactive tracks from the main ReID gallery."""
    current_proc_frame = pipeline_instance.processed_frame_counter
    if pipeline_instance.prune_interval <= 0 or current_proc_frame == 0 or (current_proc_frame % pipeline_instance.prune_interval != 0):
        return # Pruning disabled or not time yet

    logger.info(f"--- Performing Main Gallery Pruning (Frame {current_proc_frame}) ---")
    prune_older_than_frame = current_proc_frame - pipeline_instance.prune_threshold
    gids_to_prune: List[GlobalID] = []
    # Iterate over a copy of keys to allow modification during iteration
    prune_candidates = list(pipeline_instance.global_id_last_seen_frame.keys())

    for gid in prune_candidates:
        last_seen = pipeline_instance.global_id_last_seen_frame.get(gid)
        if last_seen is None: continue # Should not happen, but safety check

        # Prune if last seen is too old AND it's not currently in the lost gallery (meaning it might reappear soon)
        if last_seen < prune_older_than_frame and gid not in pipeline_instance.lost_track_gallery:
                # Also ensure it's not an *active* track right now (edge case check)
                is_active = any(g == gid for g in pipeline_instance.track_to_global_id.values())
                if not is_active:
                    gids_to_prune.append(gid)


    if gids_to_prune:
        logger.info(f"Pruning {len(gids_to_prune)} inactive GIDs from main gallery (Threshold: {pipeline_instance.prune_threshold} frames). GIDs: {gids_to_prune[:10]}...")
        for gid in gids_to_prune:
            pipeline_instance.reid_gallery.pop(gid, None)
            pipeline_instance.global_id_last_seen_cam.pop(gid, None)
            pipeline_instance.global_id_last_seen_frame.pop(gid, None) # Remove completely from state
            logger.debug(f"Pruned GID {gid} from main gallery and associated state.")
    else:
        logger.info("No GIDs met criteria for main gallery pruning.")
    logger.info(f"Main Gallery Size after prune check: {len(pipeline_instance.reid_gallery)}")


def update_pipeline_state(pipeline_instance: Any, current_active_track_keys: Set[TrackKey]):
    """Updates frame counters, handles disappeared tracks, and performs gallery cleanup/pruning."""
    current_proc_frame = pipeline_instance.processed_frame_counter

    # 1. Update last seen frame for currently active global IDs
    active_global_ids_this_frame: Set[GlobalID] = set()
    for track_key in current_active_track_keys:
        global_id = pipeline_instance.track_to_global_id.get(track_key)
        if global_id is not None:
            pipeline_instance.global_id_last_seen_frame[global_id] = current_proc_frame
            active_global_ids_this_frame.add(global_id)
    # logger.debug(f"Frame {current_proc_frame}: Updated last_seen_frame for {len(active_global_ids_this_frame)} active GIDs.")

    # 2. Handle tracks that disappeared in this frame
    _handle_disappeared_tracks(pipeline_instance, current_active_track_keys)

    # 3. Purge tracks from the lost gallery that have expired
    _purge_lost_gallery(pipeline_instance)

    # 4. Periodically prune inactive tracks from the main gallery
    _prune_main_gallery(pipeline_instance)

    # 5. Final cleanup of potentially dangling state entries (belt-and-suspenders)
    # Ensure only currently active tracks remain in maps that should reflect current state
    keys_to_delete_reid_frame = set(pipeline_instance.track_last_reid_frame.keys()) - current_active_track_keys
    for key in keys_to_delete_reid_frame: pipeline_instance.track_last_reid_frame.pop(key, None)

    # track_to_global_id should have been handled by _handle_disappeared_tracks, but check again
    keys_to_delete_gid_map = set(pipeline_instance.track_to_global_id.keys()) - current_active_track_keys
    for key in keys_to_delete_gid_map:
        logger.warning(f"Found dangling key {key} in track_to_global_id after disappearance handling. Removing.")
        pipeline_instance.track_to_global_id.pop(key, None)