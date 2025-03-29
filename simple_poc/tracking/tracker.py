from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from simple_poc.tracking.map import (
    compute_homography, create_map_visualization
)
# Import the strategies
from simple_poc.tracking.strategies import (
    DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy
)


class PersonTracker:
    """Tracks people in video frames and visualizes their positions on a map using a specified strategy."""

    def __init__(self, model_path: str, model_type: str, map_width: int = 600, map_height: int = 800):
        """
        Initialize tracker with a detection/tracking strategy.

        Args:
            model_path: Path to the model file (or identifier for torchvision).
            model_type: Type of model ('yolo', 'rtdetr', 'fasterrcnn').
            map_width: Width of the top-down map visualization.
            map_height: Height of the top-down map visualization.
        """
        self.model_path = model_path
        self.model_type = model_type

        # --- Strategy Instantiation ---
        print(f"Attempting to load model type '{model_type}' from path/identifier '{model_path}'")
        if model_type.lower() == 'yolo':
            self.strategy: DetectionTrackingStrategy = YoloStrategy(model_path)
        elif model_type.lower() == 'rtdetr':
            self.strategy: DetectionTrackingStrategy = RTDetrStrategy(model_path)
        elif model_type.lower() == 'fasterrcnn':
            # FasterRCNN might use a placeholder path if using default weights
            self.strategy: DetectionTrackingStrategy = FasterRCNNStrategy(model_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'yolo', 'rtdetr', or 'fasterrcnn'.")
        print("Strategy initialized.")
        # -----------------------------

        # Dictionary mapping track_id to list of (x,y) position history
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.map_width: int = map_width
        self.map_height: int = map_height

        self.selected_track_id: Optional[int] = None
        self.current_boxes: Dict[str, List[np.ndarray]] = defaultdict(list)  # Key: camera_id
        self.current_track_ids: Dict[str, List[int]] = defaultdict(list)  # Key: camera_id
        # Optional: Store confidences if needed later
        self.current_confidences: Dict[str, List[float]] = defaultdict(list)  # Key: camera_id

        # Dictionary mapping track_id to cropped person image
        # Note: This works best with actual track IDs. For FasterRCNN (placeholder IDs),
        # the gallery might show multiple crops for the same detection across frames.
        self.person_crops: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)

        # Homography matrix (initialized when first frame is processed)
        self.H: Optional[np.ndarray] = None

        # Frame dimensions (set during first frame processing)
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None

        # Source and destination points for homography
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None

    def select_person(self, track_id: Optional[int]) -> str:  # Allow None for deselection
        """Toggle selection of a person by track ID and return status message."""
        # Handle deselection explicitly if track_id is None
        if track_id is None:
            if self.selected_track_id is not None:
                deselected_id = self.selected_track_id
                self.selected_track_id = None
                return f"Deselected person {deselected_id}"
            else:
                return "No person was selected."

        # Handle selection/toggling
        if self.selected_track_id == track_id:
            self.selected_track_id = None
            # Check if the ID is a placeholder (-1) which shouldn't be tracked
            if track_id == -1:
                return "Cannot track placeholder detections."
            return f"Deselected person {track_id}"
        else:
            # Check if the ID is a placeholder (-1) which shouldn't be tracked
            if track_id == -1:
                return "Cannot track placeholder detections."
            self.selected_track_id = track_id
            return f"Selected person {track_id}"

    def update_person_crops(self, frame: np.ndarray, camera_id: str = '') -> None:
        """Updates the stored cropped images of detected/tracked persons."""
        current_ids_in_frame: Set[int] = set()

        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])

        if not boxes or not track_ids:
            # If no boxes/tracks for this camera, clear its crops (except selected)
            if camera_id in self.person_crops:
                ids_to_remove = [id for id in self.person_crops[camera_id].keys() if id != self.selected_track_id]
                for id in ids_to_remove:
                    self.person_crops[camera_id].pop(id, None)
            return  # Nothing to update

        # Process current frame's detections
        for box, track_id in zip(boxes, track_ids):
            # Skip placeholder IDs for gallery updates
            if track_id == -1:
                continue

            current_ids_in_frame.add(track_id)

            x, y, w, h = box  # center_x, center_y, width, height
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
                # Prevent division by zero if height is somehow 0
                aspect_ratio = w / h if h > 0 else 1
                crop_width = int(target_height * aspect_ratio)
                # Ensure width is at least 1 pixel
                crop_width = max(1, crop_width)
                try:
                    crop_resized = cv2.resize(crop, (crop_width, target_height))
                except cv2.error as e:
                    print(
                        f"Warning: cv2.resize failed for track {track_id}. Box: {box}, Crop shape: {crop.shape}, Target: ({crop_width}, {target_height}). Error: {e}")
                    continue  # Skip this crop

                # Store the cropped image (convert to RGB for display consistency)
                if camera_id not in self.person_crops:
                    self.person_crops[camera_id] = {}
                self.person_crops[camera_id][track_id] = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

        # Remove crops for people no longer detected in this specific camera (except selected person)
        if camera_id in self.person_crops:
            ids_to_remove = [id for id in self.person_crops[camera_id].keys() if id not in current_ids_in_frame]
            for id_to_remove in ids_to_remove:
                # Keep the selected person's crop even if temporarily not detected in this frame/camera
                if id_to_remove != self.selected_track_id:
                    # Check again if key exists before popping
                    if id_to_remove in self.person_crops[camera_id]:
                        self.person_crops[camera_id].pop(id_to_remove, None)

    def process_frame(self, frame: np.ndarray, paused: bool = False, camera_id: str = '') -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """Processes a single frame using the chosen strategy."""
        if frame is None:
            print(f"Warning: Received None frame for camera {camera_id}")
            # Return placeholders or handle as appropriate
            # Create blank placeholders matching expected dimensions if possible
            blank_annotated = np.zeros((self.frame_height or 480, self.frame_width or 640, 3), dtype=np.uint8)
            blank_map = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(blank_annotated, f"No Frame Data ({camera_id})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.putText(blank_map, f"No Frame Data ({camera_id})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # Convert to RGB before returning
            blank_annotated_rgb = cv2.cvtColor(blank_annotated, cv2.COLOR_BGR2RGB)
            blank_map_rgb = cv2.cvtColor(blank_map, cv2.COLOR_BGR2RGB)
            return blank_annotated_rgb, blank_map_rgb

        # Initialize homography if this is the first frame
        if self.H is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            if self.frame_height is None or self.frame_width is None:
                print("Error: Frame dimensions are None, cannot compute homography.")
                # Handle error appropriately, maybe return None or raise exception
                return None, None  # Or return placeholders as above

            print(f"Calculating homography for frame size: {self.frame_width}x{self.frame_height}")
            try:
                self.H, self.src_points, self.dst_points = compute_homography(
                    self.frame_width, self.frame_height, self.map_width, self.map_height
                )
                if self.H is None or self.src_points is None or self.dst_points is None:
                    print("Error: Homography computation failed.")
                    # Handle error: return None, raise, or use default map
                    # For now, create blank map
                    map_img = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8)  # Light gray
                    cv2.putText(map_img, "Homography Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    return frame, cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)  # Return original frame and error map

            except Exception as e:
                print(f"Error computing homography: {e}")
                # Create blank map on error
                map_img = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8)  # Light gray
                cv2.putText(map_img, "Homography Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                # Convert map to RGB before returning
                map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
                # Convert original frame to RGB
                annotated_frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                return annotated_frame_rgb, map_img_rgb

        if not paused:
            # --- Delegate processing to the strategy ---
            try:
                # Pass the BGR frame directly
                boxes_xywh, track_ids, confidences = self.strategy.process_frame(frame)
                # Store results for this camera
                self.current_boxes[camera_id] = boxes_xywh
                self.current_track_ids[camera_id] = track_ids
                self.current_confidences[camera_id] = confidences  # Store confidences too
                self.update_person_crops(frame, camera_id)  # Update crops based on new detections
            except Exception as e:
                print(f"Error processing frame with {self.model_type} strategy for camera {camera_id}: {e}")
                # Clear previous results for this camera on error to avoid showing stale data
                self.current_boxes[camera_id] = []
                self.current_track_ids[camera_id] = []
                self.current_confidences[camera_id] = []
                # Optionally update crops to remove old ones if processing fails
                self.update_person_crops(frame, camera_id)  # This will clear non-selected crops

        # --- Visualization (remains largely the same) ---
        annotated_frame = self.create_annotated_frame(frame.copy(), camera_id)

        # Create map using potentially updated current boxes/tracks
        # Ensure homography components are valid before creating map
        map_img = None
        if self.H is not None and self.dst_points is not None:
            # Get the current data for the specific camera, default to empty lists if not found
            current_cam_boxes = self.current_boxes.get(camera_id, [])
            current_cam_track_ids = self.current_track_ids.get(camera_id, [])

            map_img = create_map_visualization(
                self.map_width, self.map_height, self.dst_points,
                current_cam_boxes,  # Use data specific to this camera
                current_cam_track_ids,  # Use data specific to this camera
                self.track_history,  # History is global (or could be made per-camera)
                self.H,
                self.selected_track_id
            )
        else:
            # Create a blank map if homography isn't ready
            map_img = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Convert annotated_frame and map_img to RGB before returning for Gradio.
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) if annotated_frame is not None else None
        map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB) if map_img is not None else None

        return annotated_frame_rgb, map_img_rgb

    def create_annotated_frame(self, frame: np.ndarray, camera_id: str = '') -> np.ndarray:
        """Draws bounding boxes, IDs, and ROI on the frame."""
        # Draw region of interest polygon if points are defined
        if self.src_points is not None:
            try:
                cv2.polylines(frame, [self.src_points.astype(np.int32).reshape((-1, 1, 2))],
                              True, (0, 255, 255), 2)
            except Exception as e:
                print(f"Warning: Could not draw ROI polygon. Error: {e}")

        # Process each tracked person for this camera
        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])
        confidences = self.current_confidences.get(camera_id, [])  # Get confidences

        # Ensure lengths match, fallback if confidences weren't stored for some reason
        if len(confidences) != len(boxes):
            confidences = [0.0] * len(boxes)  # Placeholder confidence

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            # Skip drawing placeholder IDs (-1) or only show box without ID?
            # Let's draw them but make it clear they are placeholders.
            # Or, option: skip if selected_track_id is set and this is -1
            if self.selected_track_id is not None and track_id == -1:
                continue

            is_selected = (track_id == self.selected_track_id and track_id != -1)  # Cannot select -1

            # Use different colors and thicknesses
            color = (0, 0, 255) if is_selected else (0, 255, 0)
            if track_id == -1:
                color = (255,  0, 0)  # Blue for placeholder detections (BGR)
            thickness = 3 if is_selected else 2

            # Draw bounding box
            x, cy, w, h = box  # Center x,y coordinates plus width,height
            x1, y1 = int(x - w / 2), int(cy - h / 2)  # Convert to top-left corner
            x2, y2 = int(x + w / 2), int(cy + h / 2)  # Convert to bottom-right corner

            # Ensure box coordinates are valid before drawing
            if x1 < x2 and y1 < y2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                id_text = f"ID:{track_id}" if track_id != -1 else "Det"
                conf_text = f":{conf:.2f}"  # Add confidence score
                cv2.putText(frame, id_text + conf_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Show tracking status text
        status = f"Tracking ({self.model_type.upper()})"
        if self.selected_track_id is not None:
            status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def process_multiple_frames(self, frames: Dict[str, np.ndarray], paused: bool = False) -> Tuple[
        Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        """Processes frames from multiple cameras and generates a combined map."""
        annotated_frames_rgb = {}
        # These will be updated within process_frame for each camera
        # self.current_boxes = {} # Don't reset here, process_frame handles per-camera
        # self.current_track_ids = {}
        # self.current_confidences = {}

        # Clear track history only if needed (e.g., on dataset load, not every multi-frame step)
        # self.track_history.clear() # Might be too aggressive here

        all_processed_boxes = []
        all_processed_track_ids = []

        for camera_id, frame in frames.items():
            annotated_frame_rgb, _ = self.process_frame(frame, paused, camera_id)  # map generated later
            annotated_frames_rgb[camera_id] = annotated_frame_rgb

            # Collect the results that were just stored by process_frame for this camera
            all_processed_boxes.extend(self.current_boxes.get(camera_id, []))
            all_processed_track_ids.extend(self.current_track_ids.get(camera_id, []))

        # Now create the combined map using the aggregated results from all cameras processed in this step
        # Check if homography is ready
        map_img_rgb = None
        if self.H is not None and self.dst_points is not None:
            # Need to update track history *before* creating the map visualization
            # Iterate through the latest results for *all* cameras to update history
            temp_history = defaultdict(list)  # Use a temporary history for this step's updates
            for cam_id in frames.keys():
                boxes = self.current_boxes.get(cam_id, [])
                track_ids = self.current_track_ids.get(cam_id, [])
                for box, track_id in zip(boxes, track_ids):
                    if track_id != -1:  # Only track valid IDs
                        x, cy, w, h = box
                        # Use bottom center point for position on ground
                        y = cy + h / 2
                        # Use a temporary dict to avoid modifying self.track_history directly here
                        # We'll update self.track_history after generating the map for this step
                        temp_history[track_id].append((float(x), float(y)))

            # Update the main track history *after* processing all cameras for this frame step
            for track_id, points in temp_history.items():
                self.track_history[track_id].extend(points)
                # Limit history length
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id] = self.track_history[track_id][-30:]

            # Create map using all boxes/ids from *this processing step*
            map_img_bgr = create_map_visualization(
                self.map_width, self.map_height, self.dst_points,
                all_processed_boxes,  # Combined boxes from all cameras in this step
                all_processed_track_ids,  # Combined IDs from all cameras in this step
                self.track_history,  # Use the updated global history
                self.H,
                self.selected_track_id
            )
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB) if map_img_bgr is not None else None
        else:
            # Create blank map if homography not ready
            map_img_bgr = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img_bgr, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)

        # Clear per-camera crops before next update cycle?
        # This depends on whether you want the gallery to persist across frames even if a person
        # temporarily disappears. Let's keep them persistent for now, update_person_crops handles removal.
        # self.person_crops.clear() # Probably don't want this here

        return annotated_frames_rgb, map_img_rgb
