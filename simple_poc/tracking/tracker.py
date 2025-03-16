from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from simple_poc.tracking.map import (
    compute_homography, create_map_visualization
)


class PersonTracker:
    """Tracks people in video frames and visualizes their positions on a map."""

    def __init__(self, model_path: str, map_width: int = 600, map_height: int = 800):
        """Initialize tracker with YOLO model and tracking parameters."""
        self.model = YOLO(model_path)

        # Dictionary mapping track_id to list of (x,y) position history
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.map_width: int = map_width
        self.map_height: int = map_height

        self.selected_track_id: Optional[int] = None
        self.current_boxes: Dict[str, List[np.ndarray]] = {}  # Key: camera_id
        self.current_track_ids: Dict[str, List[int]] = {}  # Key: camera_id

        # Dictionary mapping track_id to cropped person image
        self.person_crops: Dict[str, Dict[int, np.ndarray]] = {}

        # Homography matrix (initialized when first frame is processed)
        self.H: Optional[np.ndarray] = None

        # Frame dimensions (set during first frame processing)
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None

        # Source and destination points for homography
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None

    def select_person(self, track_id: int) -> str:
        """Toggle selection of a person by track ID and return status message."""
        if self.selected_track_id == track_id:
            self.selected_track_id = None
            return f"Deselected person {track_id}"
        self.selected_track_id = track_id
        return f"Selected person {track_id}"

    def update_person_crops(self, frame: np.ndarray, camera_id: str = '') -> None:
        current_ids: Set[int] = set()

        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])

        for box, track_id in zip(boxes, track_ids):
            current_ids.add(track_id)

        # Extract bounding box coordinates (center x,y,width,height to top-left and bottom-right)
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)  # Top-left corner
            x2, y2 = int(x + w / 2), int(y + h / 2)  # Bottom-right corner

            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)

            # Only create crop if dimensions are valid
            if x2 > x1 and y2 > y1:
                # Extract person crop from frame
                crop = frame[y1:y2, x1:x2].copy()

                # Resize crop to consistent height while maintaining aspect ratio
                target_height = 120
                aspect_ratio = w / h
                crop_width = int(target_height * aspect_ratio)
                crop_resized = cv2.resize(crop, (crop_width, target_height))

                # Store the cropped image
                if camera_id not in self.person_crops:
                    self.person_crops[camera_id] = {}
                # Convert crop to RGB before storing
                self.person_crops[camera_id][track_id] = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)  # FIX: Convert to RGB

        # Remove crops for people no longer detected (except selected person)
        if camera_id in self.person_crops:
            ids_to_remove = [id for id in self.person_crops[camera_id].keys() if id not in current_ids]
            for id in ids_to_remove:
                # Keep the selected person's crop even if temporarily not detected
                if id != self.selected_track_id:
                    if camera_id in self.person_crops:
                        self.person_crops[camera_id].pop(id, None)

    def process_frame(self, frame: np.ndarray, paused: bool = False, camera_id: str = '') -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        if frame is None:
            return None, None

        # Initialize homography if this is the first frame
        if self.H is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.H, self.src_points, self.dst_points = compute_homography(
                self.frame_width, self.frame_height, self.map_width, self.map_height
            )

        if not paused:
            results = self.model.track(frame, persist=True)

            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                self.current_boxes[camera_id] = results[0].boxes.xywh.cpu().tolist()
                self.current_track_ids[camera_id] = results[0].boxes.id.int().cpu().tolist()
                self.update_person_crops(frame, camera_id)

        annotated_frame = self.create_annotated_frame(frame.copy(), camera_id)
        map_img = create_map_visualization(
            self.map_width, self.map_height, self.dst_points,
            self.current_boxes.get(camera_id, []),
            self.current_track_ids.get(camera_id, []),
            self.track_history, self.H, self.selected_track_id
        )

        # Convert annotated_frame and map_img to RGB before returning.
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) if annotated_frame is not None else None  # FIX
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB) if map_img is not None else None # FIX

        return annotated_frame, map_img

    def create_annotated_frame(self, frame: np.ndarray, camera_id: str = '') -> np.ndarray:
        # Draw region of interest polygon
        cv2.polylines(frame, [self.src_points.astype(np.int32).reshape((-1, 1, 2))],
                      True, (0, 255, 255), 2)

        # Process each tracked person for this camera
        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])

        for box, track_id in zip(boxes, track_ids):
            # Skip if we're focusing on a selected person and this isn't them
            if self.selected_track_id is not None and track_id != self.selected_track_id:
                continue

            x, cy, w, h = box  # Center x,y coordinates plus width,height
            is_selected = (track_id == self.selected_track_id)

            # Use different colors and thicknesses for selected vs regular detections
            color = (0, 0, 255) if is_selected else (0, 255, 0)  # Red if selected, green otherwise
            thickness = 3 if is_selected else 2

            # Draw bounding box with ID
            x1, y1 = int(x - w / 2), int(cy - h / 2)  # Convert to top-left corner
            x2, y2 = int(x + w / 2), int(cy + h / 2)  # Convert to bottom-right corner
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Show tracking status text
        status = "Tracking"
        if self.selected_track_id is not None:
            status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def process_multiple_frames(self, frames: Dict[str, np.ndarray], paused: bool = False) -> Tuple[
        Dict[str, np.ndarray], Optional[np.ndarray]]:
        annotated_frames = {}
        combined_boxes = {}
        combined_track_ids = {}

        for camera_id, frame in frames.items():
            if camera_id not in combined_boxes:
                combined_boxes[camera_id] = []
                combined_track_ids[camera_id] = []

            annotated_frame, _ = self.process_frame(frame, paused, camera_id)
            annotated_frames[camera_id] = annotated_frame

            # Append the current boxes and track_ids for this camera
            if camera_id in self.current_boxes:
                combined_boxes[camera_id].extend(self.current_boxes[camera_id])
            if camera_id in self.current_track_ids:
                combined_track_ids[camera_id].extend(self.current_track_ids[camera_id])

        # Update current_boxes and current_track_ids for all cameras
        self.current_boxes = combined_boxes
        self.current_track_ids = combined_track_ids

        map_img = create_map_visualization(
            self.map_width, self.map_height, self.dst_points,
            [box for boxes in self.current_boxes.values() for box in boxes],
            [track_id for ids in self.current_track_ids.values() for track_id in ids],
            self.track_history, self.H, self.selected_track_id
        )

        return annotated_frames, map_img