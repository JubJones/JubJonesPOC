from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from simple_poc.tracking.map import compute_homography, create_map_visualization
from simple_poc.tracking.strategies import (
    DetectionTrackingStrategy,
    YoloStrategy,
    RTDetrStrategy,
    FasterRCNNStrategy,
)


class PersonTracker:
    """
    Manages person detection and tracking using a selected strategy,
    maintains track history, handles person selection, and generates
    visualizations (annotated frames, map view, person crops).

    Uses a SINGLE strategy instance shared across all camera processing calls
    within a `process_multiple_frames` batch.
    """

    # Define constants for detection keys and placeholder IDs
    DETECTION_KEY_PREFIX = "det_"
    PLACEHOLDER_TRACK_ID = -1

    def __init__(
        self,
        model_path: str,
        model_type: str,
        map_width: int = 600,
        map_height: int = 800,
    ):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.map_width = map_width
        self.map_height = map_height

        # --- Initialize the single, shared detection/tracking strategy ---
        print(
            f"Initializing SINGLE strategy instance. Type: {self.model_type}, Path: {self.model_path}"
        )
        try:
            if self.model_type == "yolo":
                self.strategy: DetectionTrackingStrategy = YoloStrategy(self.model_path)
            elif self.model_type == "rtdetr":
                self.strategy: DetectionTrackingStrategy = RTDetrStrategy(
                    self.model_path
                )
            elif self.model_type == "fasterrcnn":
                self.strategy: DetectionTrackingStrategy = FasterRCNNStrategy(
                    self.model_path
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
            print("Strategy initialized successfully.")
        except Exception as e:
            # Log the fatal error and raise a specific runtime error
            error_msg = f"FATAL ERROR: Failed to initialize tracking strategy '{self.model_type}' with path '{self.model_path}': {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        # -------------------------------------------------------------------

        # State variables
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(
            list
        )  # Stores global history (bottom-center points)
        self.selected_track_id: Optional[int] = None

        # Per-camera results from the *last* processing step
        self.latest_boxes_per_camera: Dict[str, List[List[float]]] = defaultdict(list)
        self.latest_track_ids_per_camera: Dict[str, List[int]] = defaultdict(list)
        self.latest_confidences_per_camera: Dict[str, List[float]] = defaultdict(list)

        # Stores RGB crops for gallery display {camera_id: {track_id_or_det_key: crop_image}}
        self.person_crops: Dict[str, Dict[Union[int, str], np.ndarray]] = defaultdict(
            dict
        )

        # Homography and frame dimension attributes (can be set externally, e.g., by app.py)
        self.homography_matrix: Optional[np.ndarray] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.source_points: Optional[np.ndarray] = (
            None  # Points in frame corresponding to map ROI
        )
        self.destination_points: Optional[np.ndarray] = (
            None  # Points on map defining the ROI
        )

    def select_person(self, track_id: Optional[Union[int, str]]) -> str:
        """Selects or deselects a person track for highlighting."""
        if isinstance(track_id, str) and track_id.startswith(self.DETECTION_KEY_PREFIX):
            return "Cannot select simple detections (det_*)."
        if not isinstance(track_id, (int, type(None))):  # Allow None or int
            return (
                "Invalid ID format for selection. Must be an integer track ID or None."
            )

        # Handle deselection via None input
        if track_id is None:
            if self.selected_track_id is not None:
                deselected_id = self.selected_track_id
                self.selected_track_id = None
                return f"Deselected person {deselected_id}."
            else:
                return "No person was selected."

        # Handle selection/deselection via integer ID input
        if track_id == self.PLACEHOLDER_TRACK_ID:
            return "Cannot select placeholder detections (-1)."

        if self.selected_track_id == track_id:  # Toggle off if already selected
            self.selected_track_id = None
            return f"Deselected person {track_id}."
        else:  # Select the new track ID
            self.selected_track_id = track_id
            return f"Selected person {track_id}."

    def _update_person_crops_for_camera(
        self, frame_bgr: np.ndarray, camera_id: str
    ) -> None:
        """Generates and stores resized RGB crops for the gallery from one camera's results."""
        if self.frame_height is None or self.frame_width is None:
            # Try to set dimensions if not already set
            if frame_bgr is not None and frame_bgr.size > 0:
                self.frame_height, self.frame_width = frame_bgr.shape[:2]
            else:
                print(
                    f"Warning [update_person_crops]: Cannot update crops for cam {camera_id}, frame dimensions unknown."
                )
                return  # Cannot crop without knowing frame size

        # Use the latest results stored for this camera
        boxes_xywh = self.latest_boxes_per_camera.get(camera_id, [])
        track_ids = self.latest_track_ids_per_camera.get(camera_id, [])

        # Ensure the dictionary for this camera exists
        if camera_id not in self.person_crops:
            self.person_crops[camera_id] = {}

        current_crops_for_cam = {}  # Build new crops dict for this frame/camera

        for index, (box, track_id) in enumerate(zip(boxes_xywh, track_ids)):
            # Determine the key for storage (integer track ID or string detection key)
            storage_key: Union[int, str]
            if isinstance(track_id, int) and track_id != self.PLACEHOLDER_TRACK_ID:
                storage_key = track_id
            else:
                storage_key = (
                    f"{self.DETECTION_KEY_PREFIX}{index}"  # Use index for detections
                )

            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
                print(
                    f"Warning [update_person_crops]: Invalid box format for key {storage_key}, cam {camera_id}. Skipping."
                )
                continue

            center_x, center_y, width, height = box
            if width <= 0 or height <= 0:
                # print(f"Warning [update_person_crops]: Non-positive dimensions for key {storage_key}, cam {camera_id}. Skipping.")
                continue

            # Calculate bounding box corners (xyxy format)
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            # Clip coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.frame_width - 1, x2)
            y2 = min(self.frame_height - 1, y2)

            # Extract crop if dimensions are valid after clipping
            if x2 > x1 and y2 > y1:
                if frame_bgr is None or frame_bgr.size == 0:
                    print(
                        f"Warning [update_person_crops]: Cannot crop from invalid frame for cam {camera_id}."
                    )
                    continue  # Skip if frame is bad

                crop_bgr = frame_bgr[y1:y2, x1:x2].copy()
                if crop_bgr.size == 0:
                    print(
                        f"Warning [update_person_crops]: Empty crop generated for key {storage_key}, cam {camera_id}. Skipping."
                    )
                    continue

                # Resize crop to a fixed height while maintaining aspect ratio
                target_height = 120
                aspect_ratio = width / height if height > 0 else 1
                target_width = max(1, int(target_height * aspect_ratio))

                try:
                    crop_resized_bgr = cv2.resize(
                        crop_bgr, (target_width, target_height)
                    )
                    # Store as RGB for Gradio Image component
                    crop_resized_rgb = cv2.cvtColor(crop_resized_bgr, cv2.COLOR_BGR2RGB)
                    current_crops_for_cam[storage_key] = crop_resized_rgb
                except cv2.error as e:
                    print(
                        f"Warning [update_person_crops]: cv2.resize failed for key {storage_key}, cam {camera_id}: {e}"
                    )
                    continue  # Skip if resize fails

        # Replace the old crops for this camera with the newly generated ones
        self.person_crops[camera_id] = current_crops_for_cam

    def _process_single_camera_frame(
        self, frame_bgr: Optional[np.ndarray], camera_id: str, paused: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Processes a single frame from one camera using the shared strategy.
        Updates internal state (latest results, crops, history) for this camera.
        Generates annotated frame and map visualization based on current state.

        Args:
            frame_bgr: The input frame (BGR) or None if loading failed.
            camera_id: Identifier for the camera source.
            paused: If True, skips running the detection/tracking strategy.

        Returns:
            Tuple containing:
            - Annotated frame (RGB NumPy array) or placeholder.
            - Map visualization (RGB NumPy array) or placeholder.
        """
        # --- Handle Invalid Frame Input ---
        if frame_bgr is None:
            blank_h = self.frame_height or 480
            blank_w = self.frame_width or 640
            blank_annotated = np.zeros((blank_h, blank_w, 3), dtype=np.uint8)
            blank_map = np.full(
                (self.map_height, self.map_width, 3), 255, dtype=np.uint8
            )  # White map
            cv2.putText(
                blank_annotated,
                f"No Frame ({camera_id})",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                blank_map,
                f"No Frame ({camera_id})",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
            # Convert to RGB for Gradio output
            return cv2.cvtColor(blank_annotated, cv2.COLOR_BGR2RGB), cv2.cvtColor(
                blank_map, cv2.COLOR_BGR2RGB
            )

        # --- Ensure Homography (if not already set externally) ---
        # Note: Typically set by app.py once, but this handles the first frame case
        if self.homography_matrix is None:
            if self.frame_height is None or self.frame_width is None:
                if frame_bgr.size > 0:
                    self.frame_height, self.frame_width = frame_bgr.shape[:2]

            if self.frame_height and self.frame_width:
                try:
                    print(
                        f"Tracker: Computing homography ({self.frame_width}x{self.frame_height} -> {self.map_width}x{self.map_height})"
                    )
                    (
                        self.homography_matrix,
                        self.source_points,
                        self.destination_points,
                    ) = compute_homography(
                        self.frame_width,
                        self.frame_height,
                        self.map_width,
                        self.map_height,
                    )
                    if self.homography_matrix is None:
                        raise ValueError("Homography computation returned None")
                except Exception as e:
                    print(f"Error computing homography: {e}")
                    self.homography_matrix = None  # Ensure it remains None on failure

        # If homography is still missing after attempt, return error state
        if self.homography_matrix is None:
            error_map = np.full(
                (self.map_height, self.map_width, 3), 240, dtype=np.uint8
            )  # Grey map
            cv2.putText(
                error_map,
                "Homography Error",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            # Return original frame (converted to RGB) and error map
            return cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB), cv2.cvtColor(
                error_map, cv2.COLOR_BGR2RGB
            )

        # --- Run Detection/Tracking Strategy (if not paused) ---
        if not paused:
            try:
                # Use the single shared strategy instance
                boxes_xywh, track_ids, confidences = self.strategy.process_frame(
                    frame_bgr
                )

                # Store the latest results for this specific camera
                self.latest_boxes_per_camera[camera_id] = boxes_xywh
                self.latest_track_ids_per_camera[camera_id] = track_ids
                self.latest_confidences_per_camera[camera_id] = confidences

                # Update the global track history based on results from this camera
                for box, track_id in zip(boxes_xywh, track_ids):
                    # Only update history for valid, tracked objects (not detections)
                    if (
                        isinstance(track_id, int)
                        and track_id != self.PLACEHOLDER_TRACK_ID
                    ):
                        x, cy, w, h = box
                        by = cy + h / 2
                        if w > 0 and h > 0:
                            self.track_history[track_id].append((float(x), float(by)))

                self._update_person_crops_for_camera(frame_bgr, camera_id)

            except Exception as e:
                print(
                    f"Error processing frame with {self.model_type} strategy for camera {camera_id}: {e}"
                )
                self.latest_boxes_per_camera[camera_id] = []
                self.latest_track_ids_per_camera[camera_id] = []
                self.latest_confidences_per_camera[camera_id] = []
                # Optionally clear crops too, or leave stale ones? Clearing seems safer.
                if camera_id in self.person_crops:
                    self.person_crops[camera_id].clear()
        # --- If paused, we use the previously stored latest results ---
        else:
            self._update_person_crops_for_camera(frame_bgr, camera_id)

        # --- Create Visualizations using current state ---
        annotated_frame_bgr = self._create_annotated_frame(frame_bgr.copy(), camera_id)

        # Map Visualization (using latest results for this camera AND global history)
        map_img_bgr = create_map_visualization(
            self.map_width,
            self.map_height,
            self.destination_points,  # Use stored destination points
            self.latest_boxes_per_camera.get(
                camera_id, []
            ),  # Boxes from current camera
            self.latest_track_ids_per_camera.get(
                camera_id, []
            ),  # IDs from current camera
            self.track_history,  # Global track history
            self.homography_matrix,  # Stored homography
            self.selected_track_id,  # Optional selected ID
        )

        # Convert to RGB for Gradio output
        annotated_frame_rgb = (
            cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
            if annotated_frame_bgr is not None
            else None
        )
        map_img_rgb = (
            cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)
            if map_img_bgr is not None
            else None
        )

        return annotated_frame_rgb, map_img_rgb

    def _create_annotated_frame(
        self, frame_bgr: np.ndarray, camera_id: str
    ) -> np.ndarray:
        """Draws boxes, IDs, ROI, and status text onto a frame copy."""
        if frame_bgr is None:
            h = self.frame_height or 480
            w = self.frame_width or 640
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                f"Invalid Frame ({camera_id})",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return placeholder

        # Draw ROI defined by source points used for homography
        if self.source_points is not None:
            try:
                # Reshape needed for polylines: (N, 1, 2)
                roi_points = self.source_points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame_bgr,
                    [roi_points],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=2,
                )  # Yellow ROI
            except Exception as e:
                print(
                    f"Warning: Could not draw ROI polygon for camera {camera_id}. Error: {e}"
                )

        # Get latest results for this camera
        boxes = self.latest_boxes_per_camera.get(camera_id, [])
        track_ids = self.latest_track_ids_per_camera.get(camera_id, [])
        confidences = self.latest_confidences_per_camera.get(camera_id, [])

        # Ensure confidence list matches box list length if something went wrong
        if len(confidences) != len(boxes):
            confidences = [0.0] * len(boxes)

        # Draw bounding boxes and labels
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
                continue
            cx, cy, w, h = box
            if w <= 0 or h <= 0:
                continue

            is_tracked = (
                isinstance(track_id, int) and track_id != self.PLACEHOLDER_TRACK_ID
            )
            is_selected = is_tracked and (track_id == self.selected_track_id)

            # Determine color and thickness
            color = (0, 0, 255)  # Red for selected
            if not is_selected:
                color = (
                    (0, 255, 0) if is_tracked else (255, 0, 0)
                )  # Green for tracked, Blue for detection
            thickness = 3 if is_selected else 2

            # Calculate corner coordinates (xyxy)
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)

            # Draw rectangle (only if valid coordinates)
            if x1 < x2 and y1 < y2:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

                # Prepare and draw label text
                id_text = f"ID:{track_id}" if is_tracked else "Det"
                conf_text = f":{conf:.2f}"
                label = id_text + conf_text
                text_y_pos = (
                    y1 - 5 if y1 > 10 else y1 + 15
                )  # Adjust label position near top edge
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1, text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness,
                )

        # Draw overall status text
        status = f"Tracking ({self.model_type.upper()} - Cam: {camera_id})"
        if self.selected_track_id is not None:
            status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(
            frame_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )  # Red status text

        return frame_bgr

    def process_multiple_frames(
        self, frames_bgr: Dict[str, Optional[np.ndarray]], paused: bool = False
    ) -> Tuple[Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        """
        Processes frames from multiple cameras using the shared strategy.
        """
        annotated_frames_rgb: Dict[str, Optional[np.ndarray]] = {}
        all_processed_boxes_for_map = []
        all_processed_track_ids_for_map = []

        # --- Process each camera frame individually ---
        for camera_id, frame_bgr in frames_bgr.items():
            annotated_frame_rgb, _ = self._process_single_camera_frame(
                frame_bgr, camera_id, paused
            )
            annotated_frames_rgb[camera_id] = annotated_frame_rgb

            # Collect the latest results stored by _process_single_camera_frame for the combined map
            all_processed_boxes_for_map.extend(
                self.latest_boxes_per_camera.get(camera_id, [])
            )
            all_processed_track_ids_for_map.extend(
                self.latest_track_ids_per_camera.get(camera_id, [])
            )

        # --- Generate the Combined Map Visualization (Still not working) ---
        map_img_rgb: Optional[np.ndarray] = None
        if self.homography_matrix is not None and self.destination_points is not None:
            map_img_bgr = create_map_visualization(
                self.map_width,
                self.map_height,
                self.destination_points,
                all_processed_boxes_for_map,  # All boxes from this batch
                all_processed_track_ids_for_map,  # All IDs from this batch
                self.track_history,  # Updated global history
                self.homography_matrix,
                self.selected_track_id,
            )
            map_img_rgb = (
                cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)
                if map_img_bgr is not None
                else None
            )
        else:
            # Create placeholder map if homography is missing
            map_img_bgr = np.full(
                (self.map_height, self.map_width, 3), 255, dtype=np.uint8
            )  # White map
            cv2.putText(
                map_img_bgr,
                "Waiting for Homography",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)

        # The gallery display uses self.person_crops which was updated during the
        # _process_single_camera_frame calls within this method.
        return annotated_frames_rgb, map_img_rgb
