# FILE: reid_poc/alias_types.py
"""Defines shared type aliases and data structures for the pipeline."""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List, NamedTuple, Set
from dataclasses import dataclass, field

# --- Type Aliases ---
CameraID = str
TrackID = int
GlobalID = int
TrackKey = Tuple[CameraID, TrackID]
FeatureVector = np.ndarray
BoundingBox = np.ndarray  # xyxy format [x1, y1, x2, y2]
MapCoordinate = Tuple[float, float] # (X, Y) coordinate on the BEV map
Detection = Dict[str, Any]  # Typically {'bbox_xyxy': BoundingBox, 'conf': float, 'class_id': int}
# TrackData now includes optional map_coord
TrackData = Dict[str, Any]  # Typically includes bbox, track_id, global_id, conf, class_id, map_coord?
FrameData = Optional[np.ndarray]  # BGR image
Timings = Dict[str, float]
ScaleFactors = Tuple[float, float]  # (scale_x, scale_y)
ExitDirection = str # 'up', 'down', 'left', 'right' (represents rule trigger direction)
QuadrantName = str # 'upper_left', 'upper_right', 'lower_left', 'lower_right'
HomographyMatrix = np.ndarray # 3x3 matrix for perspective transform
HomographyMap = Dict[CameraID, Optional[HomographyMatrix]] # Map CameraID to its H matrix

# --- Handoff Configuration Structures ---
@dataclass
class ExitRule:
    """Defines a rule for triggering a handoff based on exit direction."""
    direction: ExitDirection # Direction rule applies to (e.g., 'down', 'left')
    target_cam_id: CameraID
    target_entry_area: str # Descriptive: 'upper right', 'upper left' etc. (of target cam)
    notes: Optional[str] = None

@dataclass
class CameraHandoffConfig:
    """Configuration specific to a camera for handoff purposes."""
    id: CameraID
    # source_path will be derived from main config's dataset path
    frame_shape: Optional[Tuple[int, int]] = None # (height, width), auto-detected
    exit_rules: List[ExitRule] = field(default_factory=list)


# --- Data Structures for Processing Results ---
class HandoffTriggerInfo(NamedTuple):
    """Holds information about a triggered handoff event for a specific track."""
    source_track_key: TrackKey
    rule: ExitRule
    source_bbox: BoundingBox # BBox that triggered the rule in the source camera

# ProcessedBatchResult now includes bev_coordinates
class ProcessedBatchResult(NamedTuple):
    """Holds the results and timings after processing a batch of frames."""
    results_per_camera: Dict[CameraID, List[TrackData]] # TrackData MAY contain map_coord
    timings: Timings
    processed_this_frame: bool  # Indicates if this frame was processed or skipped
    handoff_triggers: List[HandoffTriggerInfo] # List of triggers detected in this frame
    bev_coordinates: Dict[GlobalID, MapCoordinate] # Aggregated BEV coordinates for this frame
