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
        Initialize tracker configuration. Strategies are initialized later per camera.

        Args:
            model_path: Path to the model file (or identifier for torchvision).
            model_type: Type of model ('yolo', 'rtdetr', 'fasterrcnn').
            map_width: Width of the top-down map visualization.
            map_height: Height of the top-down map visualization.
        """
        self.model_path = model_path
        self.model_type = model_type
        self.strategies_per_camera: Dict[str, DetectionTrackingStrategy] = {}
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.map_width: int = map_width
        self.map_height: int = map_height
        self.selected_track_id: Optional[int] = None
        # Store detections per camera for the current frame step
        self.current_boxes: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.current_track_ids: Dict[str, List[int]] = defaultdict(list)
        self.current_confidences: Dict[str, List[float]] = defaultdict(list)
        self.person_crops: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict) # Store RGB crops

        self.H: Optional[np.ndarray] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None

    def initialize_strategies(self, camera_ids: List[str]) -> None:
        """Initializes a separate tracking strategy instance for each camera."""
        print(f"Initializing strategies for cameras: {camera_ids}")
        self.strategies_per_camera.clear()
        for cam_id in camera_ids:
            print(f"  Initializing strategy for camera: {cam_id} (Type: {self.model_type})")
            try:
                if self.model_type.lower() == 'yolo':
                    strategy = YoloStrategy(self.model_path)
                elif self.model_type.lower() == 'rtdetr':
                    strategy = RTDetrStrategy(self.model_path)
                elif self.model_type.lower() == 'fasterrcnn':
                    strategy = FasterRCNNStrategy(self.model_path)
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}")
                self.strategies_per_camera[cam_id] = strategy
            except Exception as e:
                raise RuntimeError(f"Failed to initialize strategy for camera {cam_id}") from e
        print("All camera strategies initialized.")

    def select_person(self, track_id: Optional[int]) -> str:
        """Toggle selection of a person by track ID and return status message."""
        if track_id is None:
            if self.selected_track_id is not None:
                deselected_id = self.selected_track_id
                self.selected_track_id = None
                return f"Deselected person {deselected_id}"
            else:
                return "No person was selected."

        if self.selected_track_id == track_id:
            self.selected_track_id = None
            if track_id == -1:
                 return "Cannot track placeholder detections." # Should not happen
            return f"Deselected person {track_id}"
        else:
            if track_id == -1:
                return "Cannot select placeholder detections. Click 'Clear Selection' or select a valid ID."
            self.selected_track_id = track_id
            return f"Selected person {track_id}"

    def update_person_crops(self, frame: np.ndarray, camera_id: str = '') -> None:
        """
        Updates the stored cropped images for the gallery based on current detections
        for a specific camera. Stores crops per camera.
        """
        # Ensure frame dimensions are set
        if self.frame_height is None or self.frame_width is None:
             if frame is not None and frame.size > 0:
                  self.frame_height, self.frame_width = frame.shape[:2]
             else:
                  print(f"Warning [update_person_crops]: Frame dimensions not set and frame invalid for camera {camera_id}. Cannot crop.")
                  return

        # Use data specific to this camera for this frame step
        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])

        # Ensure the inner dictionary exists for this camera
        if camera_id not in self.person_crops:
            self.person_crops[camera_id] = {}

        current_ids_in_frame_for_cam: Set[int] = set() # Track IDs seen in this frame for *this* camera

        # Add or update crops for people detected in this frame by this camera
        for box, track_id in zip(boxes, track_ids):
            if track_id == -1: # Skip placeholder detections
                continue

            current_ids_in_frame_for_cam.add(track_id)

            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
                print(f"Warning [update_person_crops]: Invalid box format for track {track_id} on camera {camera_id}. Skipping crop.")
                continue
            x, y, w, h = box
            if w <= 0 or h <= 0:
                 print(f"Warning [update_person_crops]: Invalid box dimensions (w={w}, h={h}) for track {track_id} on camera {camera_id}. Skipping crop.")
                 continue

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)

            if x2 > x1 and y2 > y1:
                if frame is None or frame.size == 0: continue # Defensive check
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0: continue # Defensive check

                target_height = 120
                aspect_ratio = w / h if h > 0 else 1
                crop_width = max(1, int(target_height * aspect_ratio))
                try:
                    crop_resized = cv2.resize(crop, (crop_width, target_height))
                    # Store the RGB crop in the per-camera dictionary
                    self.person_crops[camera_id][track_id] = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    print(f"Warning [update_person_crops]: cv2.resize failed for track {track_id} on cam {camera_id}. Error: {e}")
                    continue # Skip this crop

        # --- REMOVED THE CROP REMOVAL LOGIC ---
        # We will accumulate crops in self.person_crops across frames and cameras.
        # Stale crop removal needs a different strategy (e.g., time-based or reset per step).
        # For now, the gallery will show all unique IDs ever detected via the flattening logic.

        # # OLD removal logic (REMOVED):
        # ids_present_in_dict = list(self.person_crops[camera_id].keys())
        # for existing_id in ids_present_in_dict:
        #     if existing_id not in current_ids_in_frame_for_cam and existing_id != self.selected_track_id:
        #          if existing_id in self.person_crops[camera_id]:
        #             self.person_crops[camera_id].pop(existing_id, None)
        # -----------------------------------------

    def clear_old_person_crops(self, current_frame_detections: Dict[str, Set[int]]):
         """
         Removes crops for track IDs that are no longer present in the latest detection frame,
         unless they are the selected track ID.
         (Alternative strategy to clean up gallery)

         Args:
             current_frame_detections: Dict mapping camera_id to a set of track_ids detected in the current frame.
         """
         all_currently_visible_ids = set()
         for cam_id, ids_in_cam in current_frame_detections.items():
              all_currently_visible_ids.update(ids_in_cam)

         # Add selected ID to prevent its removal even if temporarily lost
         if self.selected_track_id is not None:
              all_currently_visible_ids.add(self.selected_track_id)

         # Iterate through stored crops and remove if ID is not currently visible
         # Need to iterate carefully to avoid modifying dict during iteration
         cameras_to_check = list(self.person_crops.keys())
         for cam_id in cameras_to_check:
              ids_to_check_in_cam = list(self.person_crops[cam_id].keys())
              for track_id in ids_to_check_in_cam:
                   if track_id not in all_currently_visible_ids:
                        if cam_id in self.person_crops and track_id in self.person_crops[cam_id]:
                             self.person_crops[cam_id].pop(track_id, None)
              # Remove camera entry if it becomes empty
              if cam_id in self.person_crops and not self.person_crops[cam_id]:
                   self.person_crops.pop(cam_id, None)


    def process_frame(self, frame: np.ndarray, paused: bool = False, camera_id: str = '') -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """Processes a single frame using the strategy specific to the camera_id."""
        # (Error handling for None frame - unchanged)
        if frame is None:
            print(f"Warning: Received None frame for camera {camera_id}")
            blank_h = self.frame_height or 480
            blank_w = self.frame_width or 640
            blank_annotated = np.zeros((blank_h, blank_w, 3), dtype=np.uint8)
            blank_map = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(blank_annotated, f"No Frame Data ({camera_id})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
            cv2.putText(blank_map, f"No Frame Data ({camera_id})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            blank_annotated_rgb = cv2.cvtColor(blank_annotated, cv2.COLOR_BGR2RGB)
            blank_map_rgb = cv2.cvtColor(blank_map, cv2.COLOR_BGR2RGB)
            return blank_annotated_rgb, blank_map_rgb

        # (Homography initialization - unchanged)
        if self.H is None:
            if self.frame_height is None or self.frame_width is None:
                 if frame.size > 0: self.frame_height, self.frame_width = frame.shape[:2]
            if not (self.frame_height and self.frame_width):
                 print("Error: Frame dimensions are None/invalid, cannot compute homography.")
                 # Return raw frame (converted to RGB) and error map
                 error_map = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8)
                 cv2.putText(error_map, "Homography Error (Bad Frame)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                 return cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB), cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

            print(f"Calculating homography for frame size: {self.frame_width}x{self.frame_height}")
            try:
                self.H, self.src_points, self.dst_points = compute_homography(self.frame_width, self.frame_height, self.map_width, self.map_height)
                if self.H is None or self.src_points is None or self.dst_points is None: raise ValueError("Homography computation failed")
            except Exception as e:
                print(f"Error computing homography: {e}")
                self.H = None # Ensure H is None on exception
                error_map = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8)
                cv2.putText(error_map, "Homography Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                return cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB), cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

        # (Get camera-specific strategy - unchanged)
        strategy = self.strategies_per_camera.get(camera_id)
        if not strategy:
             print(f"Error: No detection/tracking strategy found for camera_id '{camera_id}'. Cannot process.")
             self.current_boxes[camera_id] = []
             self.current_track_ids[camera_id] = []
             self.current_confidences[camera_id] = []
             annotated_frame = self.create_annotated_frame(frame.copy(), camera_id)
             error_map = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8)
             cv2.putText(error_map, f"No Strategy for {camera_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
             return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

        # Process frame if not paused
        if not paused:
            try:
                boxes_xywh, track_ids, confidences = strategy.process_frame(frame)
                # Store results specific to this camera *for this frame step*
                self.current_boxes[camera_id] = boxes_xywh
                self.current_track_ids[camera_id] = track_ids
                self.current_confidences[camera_id] = confidences
                # Update person crops (adds/updates RGB crops to self.person_crops[camera_id])
                self.update_person_crops(frame, camera_id) # Uses BGR frame
            except Exception as e:
                print(f"Error processing frame with {self.model_type} strategy for camera {camera_id}: {e}")
                self.current_boxes[camera_id] = []
                self.current_track_ids[camera_id] = []
                self.current_confidences[camera_id] = []
                # Optionally clear crops for this camera on error? update_person_crops handles this now.
                # self.person_crops[camera_id] = {} # Or just let update handle it

        # (Visualization - create annotated frame - unchanged)
        annotated_frame = self.create_annotated_frame(frame.copy(), camera_id)

        # (Visualization - create map based on this camera + global history - unchanged)
        map_img = None
        if self.H is not None and self.dst_points is not None:
            current_cam_boxes = self.current_boxes.get(camera_id, [])
            current_cam_track_ids = self.current_track_ids.get(camera_id, [])
            temp_history_update = defaultdict(list)
            for box, track_id in zip(current_cam_boxes, current_cam_track_ids):
                 if track_id != -1:
                    x, cy, w, h = box
                    y = cy + h / 2
                    if w > 0 and h > 0: temp_history_update[track_id].append((float(x), float(y)))
            for track_id, points in temp_history_update.items():
                self.track_history[track_id].extend(points)
                if len(self.track_history[track_id]) > 30: self.track_history[track_id] = self.track_history[track_id][-30:]

            map_img = create_map_visualization(
                self.map_width, self.map_height, self.dst_points,
                current_cam_boxes, current_cam_track_ids,
                self.track_history, self.H, self.selected_track_id
            )
        else:
            map_img = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # (Convert to RGB - unchanged)
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) if annotated_frame is not None else None
        map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB) if map_img is not None else None

        return annotated_frame_rgb, map_img_rgb

    # (create_annotated_frame - unchanged from previous version)
    def create_annotated_frame(self, frame: np.ndarray, camera_id: str = '') -> np.ndarray:
        if frame is None:
             h = self.frame_height or 480; w = self.frame_width or 640
             placeholder = np.zeros((h, w, 3), dtype=np.uint8)
             cv2.putText(placeholder, f"Invalid Frame ({camera_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             return placeholder
        if self.src_points is not None:
            try: cv2.polylines(frame, [self.src_points.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 255), 2)
            except Exception as e: print(f"Warning: Could not draw ROI polygon. Error: {e}")
        strategy_exists = camera_id in self.strategies_per_camera
        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])
        confidences = self.current_confidences.get(camera_id, [])
        if len(confidences) != len(boxes): confidences = [0.0] * len(boxes)
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4: continue
            x, cy, w, h = box;
            if w <= 0 or h <= 0: continue
            is_selected = (track_id == self.selected_track_id and track_id != -1)
            color = (0, 0, 255) if is_selected else ((255, 0, 0) if track_id == -1 else (0, 255, 0))
            thickness = 3 if is_selected else 2
            x1, y1 = int(x - w / 2), int(cy - h / 2); x2, y2 = int(x + w / 2), int(cy + h / 2)
            if x1 < x2 and y1 < y2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                id_text = f"ID:{track_id}" if track_id != -1 else "Det"
                conf_text = f":{conf:.2f}"; text_y = y1 - 5 if y1 > 10 else y1 + 10
                cv2.putText(frame, id_text + conf_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        status = f"Tracking ({self.model_type.upper()} - Cam: {camera_id})"
        if not strategy_exists: status = f"No Strategy Loaded (Cam: {camera_id})"
        if self.selected_track_id is not None: status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def process_multiple_frames(self, frames: Dict[str, np.ndarray], paused: bool = False) -> Tuple[
        Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        """Processes frames from multiple cameras using their respective strategies and generates a combined map."""
        annotated_frames_rgb = {}
        all_processed_boxes_for_map = []
        all_processed_track_ids_for_map = []
        detections_this_step = defaultdict(set) # Collect detections for potential cleanup

        # --- Optional: Clear person crops at the start of the step ---
        # Uncomment this if you want the gallery to *only* show people detected in the *current* frame step.
        # Otherwise, it will accumulate across steps until manually cleared or pruned.
        # self.person_crops.clear()
        # -------------------------------------------------------------

        # Process each camera
        for camera_id, frame_bgr in frames.items():
            # Process frame using its specific strategy, update history, get annotated frame (RGB)
            annotated_frame_rgb, _ = self.process_frame(frame_bgr, paused, camera_id)
            annotated_frames_rgb[camera_id] = annotated_frame_rgb

            # Collect results for the combined map and for cleanup
            current_cam_boxes = self.current_boxes.get(camera_id, [])
            current_cam_track_ids = self.current_track_ids.get(camera_id, [])
            all_processed_boxes_for_map.extend(current_cam_boxes)
            all_processed_track_ids_for_map.extend(current_cam_track_ids)
            detections_this_step[camera_id].update(tid for tid in current_cam_track_ids if tid != -1)

        map_img_rgb = None
        if self.H is not None and self.dst_points is not None:
            map_img_bgr = create_map_visualization(
                self.map_width, self.map_height, self.dst_points,
                all_processed_boxes_for_map, all_processed_track_ids_for_map,
                self.track_history, self.H, self.selected_track_id
            )
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB) if map_img_bgr is not None else None
        else:
            map_img_bgr = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img_bgr, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)

        return annotated_frames_rgb, map_img_rgb
