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
    # Define dummy types if import fails to allow type hinting without crashing
    BaseModelBackend = type("BaseModelBackend", (object,), {})
    BaseTracker = type("BaseTracker", (object,), {})


# Use relative imports from other modules in the package
from .constants import DEFAULT_PRUNE_INTERVAL, DEFAULT_PRUNE_THRESHOLD_MULTIPLIER
from .feature_extractor import decide_reid_targets, extract_reid_features_batched
from .reid_association import associate_reid_batched
from .state_manager import update_pipeline_state

# Project specific types and utils (assuming they are importable)
from reid_poc.config import PipelineConfig, CameraConfig
from reid_poc.alias_types import (
    CameraID, TrackID, GlobalID, TrackKey, FeatureVector, BoundingBox, ExitRule,
    Detection, TrackData, FrameData, Timings, ProcessedBatchResult, ScaleFactors,
    HandoffTriggerInfo, QuadrantName, ExitDirection, MapCoordinates
)
# Assuming project_point_to_map is in a top-level utils module
from reid_poc.utils import project_point_to_map

logger = logging.getLogger(__name__)


class MultiCameraPipeline:
    """Handles multi-camera detection, tracking, Re-ID, association, and state management."""

    def __init__(
        self,
        config: PipelineConfig,
        detector: FasterRCNN,
        detector_transforms: Compose,
        reid_model: Optional[BaseModelBackend],
        trackers: Dict[CameraID, BaseTracker]
    ):
        """Initializes the pipeline with models, trackers, and configurations."""
        self.config = config
        self.device = config.device
        self.camera_ids = config.selected_cameras
        self.detector = detector
        self.detector_transforms = detector_transforms
        self.reid_model = reid_model
        self.trackers = trackers

        self._initialize_state()
        self._log_initialization_details()

    def _initialize_state(self):
        """Initializes all internal state variables for tracking and Re-ID."""
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}           # Main gallery: GlobalID -> FeatureVector
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {} # Lost gallery: GlobalID -> (FeatureVector, frame_last_seen_active)
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}          # Current mapping: (CamID, TrackID) -> GlobalID
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}     # Track last seen camera for a GlobalID
        self.global_id_last_seen_frame: Dict[GlobalID, int] = {}        # Track last seen frame for a GlobalID (used for pruning)
        self.track_last_reid_frame: Dict[TrackKey, int] = {}            # Track last frame ReID was attempted for a specific track
        self.next_global_id: GlobalID = 1                               # Counter for assigning new GlobalIDs
        self.processed_frame_counter: int = 0                           # Counter for processed frames/batches

        # Pruning configuration
        self.prune_interval = getattr(self.config, 'main_gallery_prune_interval_frames', DEFAULT_PRUNE_INTERVAL)
        default_threshold = int(DEFAULT_PRUNE_THRESHOLD_MULTIPLIER * self.config.lost_track_buffer_frames)
        self.prune_threshold = getattr(self.config, 'main_gallery_prune_threshold_frames', default_threshold)

    def _log_initialization_details(self):
        """Logs key configuration parameters during initialization."""
        logger.info(f"Initializing pipeline for cameras: {self.camera_ids} on device: {self.device.type}")
        logger.info(f" - Scene: {self.config.selected_scene}")
        logger.info(f" - Detector Confidence: {self.config.detection_confidence_threshold}")
        logger.info(f" - ReID Similarity Threshold: {self.config.reid_similarity_threshold}")
        logger.info(f" - ReID Refresh Interval (Proc. Frames): {self.config.reid_refresh_interval_frames}")
        logger.info(f" - Frame Skip Rate: {self.config.frame_skip_rate} (Note: Pipeline processes batches)")
        logger.info(f" - Tracker Type: {self.config.tracker_type}")
        logger.info(f" - Handoff Overlap Threshold: {self.config.min_bbox_overlap_ratio_in_quadrant:.2f}")
        logger.info(f" - Lost Track Buffer: {self.config.lost_track_buffer_frames} frames")
        logger.info(f" - BEV Map Plotting: {'Enabled' if self.config.enable_bev_map else 'Disabled'}")
        homography_count = sum(1 for cfg in self.config.cameras_config.values() if cfg.homography_matrix is not None)
        logger.info(f" - Homography matrices loaded for {homography_count}/{len(self.camera_ids)} cameras.")
        if self.prune_interval > 0:
            logger.info(f" - Main Gallery Pruning: Interval={self.prune_interval} frames, Threshold={self.prune_threshold} frames inactive.")
        else:
            logger.info(" - Main Gallery Pruning: Disabled.")

    # --------------------------------------------------------------------------
    # Pipeline Step Helpers (Detection, Tracking, Projection)
    # --------------------------------------------------------------------------

    def _preprocess_batch(self, frames: Dict[CameraID, FrameData]) -> Tuple[List[torch.Tensor], List[CameraID], List[Tuple[int, int]], List[ScaleFactors], Dict[CameraID, FrameData]]:
        """Prepares a batch of frames for the detector by resizing and transforming."""
        batch_input_tensors: List[torch.Tensor] = []
        batch_cam_ids: List[CameraID] = []
        batch_original_shapes: List[Tuple[int, int]] = []
        batch_scale_factors: List[ScaleFactors] = []
        valid_frames_for_processing: Dict[CameraID, FrameData] = {}

        for cam_id, frame_bgr in frames.items():
            if frame_bgr is None or frame_bgr.size == 0:
                continue

            valid_frames_for_processing[cam_id] = frame_bgr
            original_h, original_w = frame_bgr.shape[:2]
            frame_for_det = frame_bgr
            scale_x, scale_y = 1.0, 1.0

            # Optional resizing for detection efficiency
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
                    frame_for_det = frame_bgr # Fallback to original
                    scale_x, scale_y = 1.0, 1.0

            # Apply detector transforms
            try:
                img_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                input_tensor = self.detector_transforms(img_pil)
                batch_input_tensors.append(input_tensor.to(self.device))
                batch_cam_ids.append(cam_id)
                batch_original_shapes.append((original_h, original_w))
                batch_scale_factors.append((scale_x, scale_y))
            except Exception as transform_err:
                logger.error(f"[{cam_id}] Preprocessing failed: {transform_err}", exc_info=False)

        return batch_input_tensors, batch_cam_ids, batch_original_shapes, batch_scale_factors, valid_frames_for_processing

    def _detect_objects_batched(self, batch_input_tensors: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Performs object detection on the preprocessed batch of tensors."""
        if not batch_input_tensors:
            return []
        try:
            with torch.no_grad():
                use_amp_runtime = self.config.use_amp and self.device.type == 'cuda'
                with torch.cuda.amp.autocast(enabled=use_amp_runtime):
                    all_predictions = self.detector(batch_input_tensors)
            return all_predictions
        except Exception as e:
            logger.error(f"Object detection failed: {e}", exc_info=False)
            return []

    def _postprocess_detections(
        self,
        all_predictions: List[Dict[str, torch.Tensor]],
        batch_cam_ids: List[CameraID],
        batch_original_shapes: List[Tuple[int, int]],
        batch_scale_factors: List[ScaleFactors]
    ) -> Dict[CameraID, List[Detection]]:
        """Converts raw detector outputs to scaled bounding boxes and filters by class/confidence."""
        detections_per_camera: Dict[CameraID, List[Detection]] = defaultdict(list)
        if len(all_predictions) != len(batch_cam_ids):
            logger.warning(f"Mismatch between detector outputs ({len(all_predictions)}) and input batch size ({len(batch_cam_ids)}). Skipping postprocessing.")
            return detections_per_camera

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
                        # Scale box back to original frame dimensions
                        x1, y1, x2, y2 = box
                        orig_x1 = max(0.0, x1 * scale_x)
                        orig_y1 = max(0.0, y1 * scale_y)
                        orig_x2 = min(float(original_w - 1), x2 * scale_x)
                        orig_y2 = min(float(original_h - 1), y2 * scale_y)

                        # Ensure valid box dimensions
                        if orig_x2 > orig_x1 + 1 and orig_y2 > orig_y1 + 1:
                            bbox_orig = np.array([orig_x1, orig_y1, orig_x2, orig_y2], dtype=np.float32)
                            detections_per_camera[cam_id].append({
                                'bbox_xyxy': bbox_orig,
                                'conf': float(score),
                                'class_id': int(label)
                            })
            except Exception as postproc_err:
                logger.error(f"[{cam_id}] Error postprocessing detections: {postproc_err}", exc_info=False)

        return detections_per_camera

    def _track_and_check_handoffs(
        self,
        detections_per_camera: Dict[CameraID, List[Detection]],
        valid_frames: Dict[CameraID, FrameData]
    ) -> Tuple[Dict[CameraID, np.ndarray], Set[TrackKey], List[HandoffTriggerInfo]]:
        """Updates trackers for each camera with detections and checks for handoff triggers."""
        current_tracker_outputs: Dict[CameraID, np.ndarray] = {}
        active_track_keys: Set[TrackKey] = set()
        all_handoff_triggers: List[HandoffTriggerInfo] = []

        for cam_id in self.camera_ids:
            cam_config = self.config.cameras_config.get(cam_id)
            if not cam_config: continue # Should not happen if config validation is done

            tracker = self.trackers.get(cam_id)
            if not tracker:
                logger.warning(f"[{cam_id}] Tracker instance missing.")
                current_tracker_outputs[cam_id] = np.empty((0, 8)) # Use consistent shape for output
                continue

            # Prepare detections in the format required by the tracker (xyxy, conf, cls_id)
            cam_detections = detections_per_camera.get(cam_id, [])
            np_dets = np.empty((0, 6))
            if cam_detections:
                try:
                    np_dets = np.array([[*det['bbox_xyxy'], det['conf'], det['class_id']] for det in cam_detections], dtype=np.float32)
                except Exception as format_err:
                    logger.error(f"[{cam_id}] Failed to format detections for tracker: {format_err}", exc_info=False)

            # Get frame data (needed by some trackers, e.g., for appearance updates)
            original_frame_bgr = valid_frames.get(cam_id)
            frame_shape_orig = cam_config.frame_shape
            # Provide a dummy frame if the original is missing, using configured shape
            dummy_frame_shape = frame_shape_orig if frame_shape_orig else (1080, 1920) # Default fallback shape
            dummy_frame = original_frame_bgr if original_frame_bgr is not None else np.zeros((*dummy_frame_shape, 3), dtype=np.uint8)

            # Update tracker
            tracked_dets_np = np.empty((0, 8)) # Default to empty array with expected columns
            try:
                # BoxMOT trackers expect (N, 6) [xyxy, conf, cls] and return list of (M, 7+) [xyxy, id, conf, cls, ...]
                tracked_dets_list = tracker.update(np_dets, dummy_frame)
                if tracked_dets_list is not None and len(tracked_dets_list) > 0:
                    tracked_dets_np_maybe = np.array(tracked_dets_list)
                    # Validate output shape before using
                    if tracked_dets_np_maybe.ndim == 2 and tracked_dets_np_maybe.shape[1] >= 7: # Need at least xyxy, id, conf, cls
                        # Ensure it has 8 columns for consistency downstream (bbox, id, conf, cls, idx - idx might be optional)
                        if tracked_dets_np_maybe.shape[1] < 8:
                            # Pad with a default value (e.g., -1) if index or other columns are missing
                            padding = np.full((tracked_dets_np_maybe.shape[0], 8 - tracked_dets_np_maybe.shape[1]), -1.0)
                            tracked_dets_np = np.hstack((tracked_dets_np_maybe[:, :7], padding))
                        else:
                                tracked_dets_np = tracked_dets_np_maybe[:, :8] # Take first 8 columns

                        # Register active tracks
                        for track_data_row in tracked_dets_np:
                            try:
                                track_id = int(track_data_row[4])
                                if np.isfinite(track_id) and track_id >= 0: # Check for valid track ID
                                        active_track_keys.add((cam_id, track_id))
                            except (IndexError, ValueError, TypeError):
                                logger.warning(f"[{cam_id}] Invalid track ID found in tracker output row: {track_data_row}", exc_info=False)
                    else:
                            logger.warning(f"[{cam_id}] Tracker output has unexpected shape {tracked_dets_np_maybe.shape}. Expected (N, >=7). Treating as empty.")
            except Exception as e:
                logger.error(f"[{cam_id}] Tracker update failed: {e}", exc_info=True)

            current_tracker_outputs[cam_id] = tracked_dets_np

            # Check handoff triggers if enabled and tracks exist
            if self.config.min_bbox_overlap_ratio_in_quadrant > 0 and tracked_dets_np.shape[0] > 0:
                triggers_this_cam = self._check_handoff_triggers(
                    cam_id, tracked_dets_np, frame_shape_orig, cam_config.exit_rules
                )
                all_handoff_triggers.extend(triggers_this_cam)

        return current_tracker_outputs, active_track_keys, all_handoff_triggers

    def _check_handoff_triggers(
        self,
        cam_id: CameraID,
        tracked_dets_np: np.ndarray,
        frame_shape: Optional[Tuple[int, int]],
        cam_exit_rules: List[ExitRule]
    ) -> List[HandoffTriggerInfo]:
        """Checks if active tracks overlap significantly with predefined exit quadrants."""
        # NOTE: This method remains in core.py as it's tightly coupled with tracker output format and config
        triggers_found: List[HandoffTriggerInfo] = []
        if (not cam_exit_rules or
            not frame_shape or frame_shape[0] <= 0 or frame_shape[1] <= 0 or
            tracked_dets_np.shape[0] == 0 or
            self.config.min_bbox_overlap_ratio_in_quadrant <= 0):
            return triggers_found # No rules, invalid shape, no tracks, or disabled threshold

        H, W = frame_shape
        mid_x, mid_y = W // 2, H // 2

        # Define quadrant regions based on frame dimensions
        quadrant_regions: Dict[QuadrantName, Tuple[int, int, int, int]] = {
            'upper_left': (0, 0, mid_x, mid_y), 'upper_right': (mid_x, 0, W, mid_y),
            'lower_left': (0, mid_y, mid_x, H), 'lower_right': (mid_x, mid_y, W, H),
        }
        direction_to_quadrants: Dict[ExitDirection, List[QuadrantName]] = {
            'up': ['upper_left', 'upper_right'], 'down': ['lower_left', 'lower_right'],
            'left': ['upper_left', 'lower_left'], 'right': ['upper_right', 'lower_right'],
        }

        processed_track_ids_this_cam = set() # Ensure each track triggers at most one rule per frame

        for rule in cam_exit_rules:
            relevant_quadrant_names = direction_to_quadrants.get(rule.direction, [])
            if not relevant_quadrant_names: continue
            exit_regions_coords = [quadrant_regions[name] for name in relevant_quadrant_names if name in quadrant_regions]
            if not exit_regions_coords: continue

            for track_data in tracked_dets_np:
                if len(track_data) < 5: continue # Need at least xyxy, id
                try:
                    track_id = int(track_data[4])
                except (ValueError, IndexError, TypeError):
                    continue # Skip rows with invalid track IDs
                if track_id in processed_track_ids_this_cam:
                    continue # Already triggered a rule

                bbox = track_data[0:4].astype(np.float32)
                x1, y1, x2, y2 = map(int, bbox)
                bbox_w, bbox_h = x2 - x1, y2 - y1
                if bbox_w <= 0 or bbox_h <= 0: continue
                bbox_area = float(bbox_w * bbox_h)

                # Calculate intersection area with the relevant quadrants for this rule
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
                        source_track_key=source_track_key, rule=rule, source_bbox=bbox
                    )
                    triggers_found.append(trigger_info)
                    processed_track_ids_this_cam.add(track_id)
                    logger.info(
                        f"HANDOFF TRIGGER: Track {source_track_key} matched rule '{rule.direction}' -> Cam [{rule.target_cam_id}] Area [{rule.target_entry_area}]."
                    )
                    # Break inner loop once a track triggers a rule
                    break # Move to the next rule or finish if no more rules

        return triggers_found

    def _project_tracks_and_finalize_results(
        self,
        current_tracker_outputs: Dict[CameraID, np.ndarray],
    ) -> Dict[CameraID, List[TrackData]]:
        """Projects track foot points to map coordinates (if homography exists) and formats final output."""
        final_results_per_camera: Dict[CameraID, List[TrackData]] = defaultdict(list)
        projection_success_count = 0

        for cam_id, tracked_dets_np in current_tracker_outputs.items():
            cam_config = self.config.cameras_config.get(cam_id)
            homography_matrix = cam_config.homography_matrix if cam_config else None
            # logger.debug(f"[{cam_id}] Projection Check: Homography matrix is {'AVAILABLE' if homography_matrix is not None else 'MISSING'}")

            if tracked_dets_np.shape[0] > 0:
                for track_data in tracked_dets_np:
                    # Ensure track data has expected columns (xyxy, id, conf, cls)
                    if len(track_data) >= 7:
                        try:
                            x1, y1, x2, y2 = map(float, track_data[0:4])
                            track_id = int(track_data[4])
                            conf = float(track_data[5])
                            cls = int(track_data[6])

                            # Check for invalid track_id from tracker output
                            if not np.isfinite(track_id) or track_id < 0:
                                # logger.warning(f"[{cam_id}] Skipping track data with invalid track ID: {track_id}")
                                continue

                            current_track_key: TrackKey = (cam_id, track_id)
                            # Retrieve the Global ID assigned in the ReID step or from previous state
                            # Access state directly via self here
                            global_id: Optional[GlobalID] = self.track_to_global_id.get(current_track_key)

                            # Project foot point (center bottom) to map coordinates if homography available
                            map_coords: Optional[MapCoordinates] = None
                            if homography_matrix is not None:
                                image_point_x = (x1 + x2) / 2.0
                                image_point_y = y2 # Use bottom-center point
                                map_coords = project_point_to_map((image_point_x, image_point_y), homography_matrix)
                                if map_coords: projection_success_count += 1

                            track_info_dict: TrackData = {
                                'bbox_xyxy': np.array([x1, y1, x2, y2], dtype=np.float32),
                                'track_id': track_id,
                                'global_id': global_id, # Can be None if association failed or track is lost
                                'conf': conf,
                                'class_id': cls,
                                'map_coords': map_coords # Can be None
                            }
                            final_results_per_camera[cam_id].append(track_info_dict)
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"[{cam_id}] Failed parse/project track data row: {track_data}. Error: {e}", exc_info=False)

        # logger.debug(f"Frame {self.processed_frame_counter}: Projection stage finished. Successful projections: {projection_success_count}")
        return final_results_per_camera

    # --------------------------------------------------------------------------
    # Main Processing Method
    # --------------------------------------------------------------------------

    def process_frame_batch_full(self, frames: Dict[CameraID, FrameData], frame_idx_global: int) -> ProcessedBatchResult:
        """Processes a batch of frames through the full pipeline: Detect -> Track -> Re-ID -> Associate."""
        t_start_batch = time.time()
        timings: Timings = defaultdict(float)

        # Increment processed frame counter (used for timestamps and intervals)
        self.processed_frame_counter += 1
        proc_frame_id = self.processed_frame_counter
        logger.debug(f"--- Processing Batch Start (Global Frame: {frame_idx_global}, Processed Counter: {proc_frame_id}) ---")

        # --- Stage 1a: Preprocess Batch ---
        t_prep_start = time.time()
        batch_tensors, batch_cids, batch_shapes, batch_scales, valid_frames = self._preprocess_batch(frames)
        timings['preprocess'] = time.time() - t_prep_start

        # --- Stage 1b: Detection ---
        t_detect_start = time.time()
        predictions = self._detect_objects_batched(batch_tensors)
        timings['detection_batched'] = time.time() - t_detect_start

        # --- Stage 1c: Postprocess Detections ---
        t_postproc_start = time.time()
        detections_per_cam = self._postprocess_detections(predictions, batch_cids, batch_shapes, batch_scales)
        timings['postprocess_scale'] = time.time() - t_postproc_start

        # --- Stage 1d & 1e: Tracking & Handoff Check ---
        t_track_handoff_start = time.time()
        tracker_outputs, active_keys, handoff_triggers = self._track_and_check_handoffs(detections_per_cam, valid_frames)
        # Create map for quick lookup of triggers by track key
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo] = {
            trigger.source_track_key: trigger for trigger in handoff_triggers
        }
        timings['tracking_and_handoff_check'] = time.time() - t_track_handoff_start
        logger.debug(f"Frame {proc_frame_id}: Active track keys: {len(active_keys)}, Handoff triggers: {len(handoff_triggers)}")


        # --- Stage 2a: Decide which tracks need ReID (from feature_extractor module) ---
        t_reid_decision_start = time.time()
        tracks_for_reid = decide_reid_targets(self, active_keys, tracker_outputs, valid_frames, active_triggers_map)
        timings['reid_decision'] = time.time() - t_reid_decision_start
        logger.debug(f"Frame {proc_frame_id}: Decided to extract ReID for {len(tracks_for_reid)} tracks.")


        # --- Stage 2b: Extract ReID Features (from feature_extractor module) ---
        t_feat_start = time.time()
        extracted_features = extract_reid_features_batched(self, tracks_for_reid, valid_frames)
        timings['feature_ext'] = time.time() - t_feat_start
        logger.debug(f"Frame {proc_frame_id}: Extracted {len(extracted_features)} features.")


        # --- Stage 3: Re-ID Association (Batched) (from reid_association module) ---
        t_reid_start = time.time()
        # This function updates the pipeline's state (self.track_to_global_id etc.) directly
        _ = associate_reid_batched(self, extracted_features, active_triggers_map)
        timings['reid'] = time.time() - t_reid_start
        associated_count = sum(1 for tk in active_keys if tk in self.track_to_global_id)
        logger.debug(f"Frame {proc_frame_id}: ReID association completed. {associated_count}/{len(active_keys)} active tracks have GlobalIDs.")


        # --- Stage 4: Combine Results, Project to Map ---
        t_project_start = time.time()
        final_results_per_camera = self._project_tracks_and_finalize_results(tracker_outputs)
        timings['projection'] = time.time() - t_project_start


        # --- Stage 5: Update State and Cleanup (Includes Pruning) (from state_manager module) ---
        t_state_update_start = time.time()
        update_pipeline_state(self, active_keys) # Pass the set of keys active in *this* frame
        timings['state_update_cleanup'] = time.time() - t_state_update_start


        # --- Final Timing ---
        timings['total'] = time.time() - t_start_batch
        logger.debug(f"--- Processing Batch End (Total: {timings['total']:.4f}s) ---")


        return ProcessedBatchResult(
            results_per_camera=dict(final_results_per_camera), # Convert back to dict
            timings=dict(timings), # Convert back to dict
            processed_this_frame=True, # Assuming skipping isn't implemented here
            handoff_triggers=handoff_triggers # Return triggers detected in this batch
        )