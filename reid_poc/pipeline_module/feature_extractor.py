"""Handles Re-ID feature extraction decisions and execution."""

import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any, Set, List

import numpy as np

# BoxMOT imports for type hints
try:
    from boxmot.appearance.backends.base_backend import BaseModelBackend
except ImportError:
    BaseModelBackend = type("BaseModelBackend", (object,), {})

# Project specific types
from reid_poc.alias_types import (
    CameraID, TrackID, TrackKey, FeatureVector, FrameData, HandoffTriggerInfo
)

logger = logging.getLogger(__name__)


def decide_reid_targets(
    pipeline_instance: Any, # MultiCameraPipeline instance for state access
    active_track_keys: Set[TrackKey],
    current_tracker_outputs: Dict[CameraID, np.ndarray],
    valid_frames: Dict[CameraID, FrameData],
    active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
) -> Dict[TrackKey, np.ndarray]:
    """Determines which active tracks need Re-ID feature extraction based on state and triggers."""
    tracks_to_extract_features_for: Dict[TrackKey, np.ndarray] = {}
    proc_frame_id = pipeline_instance.processed_frame_counter # Use the current processed frame counter

    for track_key in active_track_keys:
        cam_id, track_id = track_key

        # Find the corresponding track data row
        track_data_row = None
        cam_output = current_tracker_outputs.get(cam_id)
        if cam_output is not None and cam_output.shape[0] > 0:
            try:
                # Ensure track IDs are valid numbers before comparison
                valid_indices = np.where(np.isfinite(cam_output[:, 4]))[0]
                if valid_indices.size > 0:
                        # Find the index matching the current track_id within the valid rows
                    match_indices = np.where(cam_output[valid_indices, 4].astype(int) == track_id)[0]
                    if match_indices.size > 0:
                        track_data_row = cam_output[valid_indices[match_indices[0]]]
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error finding track data row for active key {track_key}: {e}", exc_info=False)

        if track_data_row is None:
            logger.debug(f"Could not find data row for active track key {track_key}, skipping ReID decision.")
            continue # Skip if track data wasn't found for this key

        original_frame_bgr = valid_frames.get(cam_id)
        if original_frame_bgr is None or original_frame_bgr.size == 0:
            continue # Skip if frame data is missing

        # Determine if ReID is needed
        is_known = track_key in pipeline_instance.track_to_global_id
        last_reid_attempt_proc_idx = pipeline_instance.track_last_reid_frame.get(track_key, -pipeline_instance.config.reid_refresh_interval_frames - 1)
        is_due_for_refresh = (proc_frame_id - last_reid_attempt_proc_idx) >= pipeline_instance.config.reid_refresh_interval_frames
        is_triggering_handoff_now = track_key in active_triggers_map

        # Trigger ReID if:
        # 1. The track is not currently associated with a Global ID (new or lost track reappearing).
        # 2. It's due for a periodic refresh.
        # 3. It triggered a handoff rule in this frame.
        trigger_reid = (not is_known) or is_due_for_refresh or is_triggering_handoff_now

        if trigger_reid:
            tracks_to_extract_features_for[track_key] = track_data_row
            pipeline_instance.track_last_reid_frame[track_key] = proc_frame_id # Record ReID attempt frame
            logger.debug(f"Marking {track_key} for ReID extraction (Known: {is_known}, Refresh: {is_due_for_refresh}, Handoff: {is_triggering_handoff_now})")

    return tracks_to_extract_features_for


def extract_reid_features_batched(
    pipeline_instance: Any, # MultiCameraPipeline instance for model/state access
    tracks_to_extract: Dict[TrackKey, np.ndarray],
    valid_frames: Dict[CameraID, FrameData]
) -> Dict[TrackKey, FeatureVector]:
    """Extracts Re-ID features for the selected tracks, batching per camera."""
    extracted_features: Dict[TrackKey, FeatureVector] = {}
    reid_model: Optional[BaseModelBackend] = pipeline_instance.reid_model # Get model from instance

    if not tracks_to_extract or reid_model is None:
        return extracted_features

    # Group tracks by camera for efficient batch processing
    tracks_grouped_by_cam: Dict[CameraID, List[np.ndarray]] = defaultdict(list)
    track_keys_per_cam: Dict[CameraID, List[TrackKey]] = defaultdict(list) # Keep track of original keys
    for track_key, track_data in tracks_to_extract.items():
        cam_id, _ = track_key
        tracks_grouped_by_cam[cam_id].append(track_data)
        track_keys_per_cam[cam_id].append(track_key)

    # Process each camera's batch
    for cam_id, tracks_data_list in tracks_grouped_by_cam.items():
        frame_bgr = valid_frames.get(cam_id)
        if frame_bgr is None or frame_bgr.size == 0 or not tracks_data_list:
            continue

        try:
            tracks_data_np = np.array(tracks_data_list)
            # Extract features using the private helper for a single frame/camera
            features_this_cam: Dict[TrackID, FeatureVector] = _extract_features_for_single_camera(reid_model, frame_bgr, tracks_data_np)

            # Map features back to their original TrackKey
            for track_id, feature in features_this_cam.items():
                # Find the original TrackKey(s) corresponding to this track_id in this camera's batch
                original_keys_for_tid = [tk for tk in track_keys_per_cam[cam_id] if tk[1] == track_id]
                if original_keys_for_tid:
                    # Use the first found key (should typically be only one per batch)
                    extracted_features[original_keys_for_tid[0]] = feature
                else:
                        logger.warning(f"[{cam_id}] Feature extracted for T:{track_id}, but couldn't map back to an original TrackKey in this batch.")

        except Exception as fe_err:
            logger.error(f"[{cam_id}] Error during batched feature extraction call: {fe_err}", exc_info=False)

    return extracted_features


def _extract_features_for_single_camera(
    reid_model: Optional[BaseModelBackend],
    frame_bgr: FrameData,
    tracked_dets_np: np.ndarray
) -> Dict[TrackID, FeatureVector]:
    """Extracts Re-ID features for tracks detected in a single camera frame."""
    features: Dict[TrackID, FeatureVector] = {}
    if reid_model is None or frame_bgr is None or frame_bgr.size == 0 or tracked_dets_np.shape[0] == 0:
        return features

    # Expecting xyxy, id, ... columns
    if tracked_dets_np.shape[1] < 5:
        logger.warning(f"Track data has unexpected shape {tracked_dets_np.shape}, expected >= 5 cols. Skipping feature extraction.")
        return features

    bboxes_xyxy = tracked_dets_np[:, 0:4].astype(np.float32)
    track_ids_float = tracked_dets_np[:, 4] # Track IDs are usually at index 4

    if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4:
            logger.warning(f"Bounding box data has unexpected shape {bboxes_xyxy.shape}. Skipping extraction.")
            return features

    try:
        # Use the ReID model's feature extraction method
        batch_features = reid_model.get_features(bboxes_xyxy, frame_bgr)

        if batch_features is not None and len(batch_features) == len(track_ids_float):
            for i, det_feature in enumerate(batch_features):
                # Validate feature before storing
                if det_feature is not None and np.isfinite(det_feature).all() and det_feature.size > 0:
                    try:
                        track_id = int(track_ids_float[i])
                        features[track_id] = det_feature
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid track ID {track_ids_float[i]} encountered during feature mapping.")
        elif batch_features is not None:
                logger.warning(f"Mismatch between number of features extracted ({len(batch_features)}) and number of tracks ({len(track_ids_float)}).")

    except Exception as e:
        logger.error(f"ReID model feature extraction call failed: {e}", exc_info=False) # Less verbose logging

    return features