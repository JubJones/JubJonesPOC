"""Defines shared type aliases and data structures for the pipeline."""

from typing import Dict, Any, Tuple, Optional, List, NamedTuple

import numpy as np

# --- Type Aliases ---
CameraID = str
TrackID = int
GlobalID = int
TrackKey = Tuple[CameraID, TrackID]
FeatureVector = np.ndarray
BoundingBox = np.ndarray  # xyxy format
Detection = Dict[str, Any]  # Typically {'bbox_xyxy': BoundingBox, 'conf': float, 'class_id': int}
TrackData = Dict[str, Any]  # Typically includes bbox, track_id, global_id, conf, class_id
FrameData = Optional[np.ndarray]  # BGR image
Timings = Dict[str, float]
ScaleFactors = Tuple[float, float]  # (scale_x, scale_y)


# --- Data Structures ---
class ProcessedBatchResult(NamedTuple):
    """Holds the results and timings after processing a batch of frames."""
    results_per_camera: Dict[CameraID, List[TrackData]]
    timings: Timings
    processed_this_frame: bool  # Indicates if this frame was processed or skipped
