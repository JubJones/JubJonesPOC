from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from simple_poc.tracking.map import (
    compute_homography, create_map_visualization
)
from simple_poc.tracking.strategies import (
    DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy
)


class PersonTracker:
    """Tracks people in video frames and visualizes their positions on a map using a specified strategy."""

    # ( __init__ , initialize_strategies, select_person remain unchanged from last version)
    def __init__(self, model_path: str, model_type: str, map_width: int = 600, map_height: int = 800):
        self.model_path = model_path
        self.model_type = model_type
        self.strategies_per_camera: Dict[str, DetectionTrackingStrategy] = {}
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.map_width: int = map_width
        self.map_height: int = map_height
        self.selected_track_id: Optional[int] = None
        self.current_boxes: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.current_track_ids: Dict[str, List[int]] = defaultdict(list)
        self.current_confidences: Dict[str, List[float]] = defaultdict(list)
        self.person_crops: Dict[str, Dict[Union[int, str], np.ndarray]] = defaultdict(dict) # Key can be int or str (det_...)
        self.H: Optional[np.ndarray] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None

    def initialize_strategies(self, camera_ids: List[str]) -> None:
        print(f"Initializing strategies for cameras: {camera_ids}")
        self.strategies_per_camera.clear()
        for cam_id in camera_ids:
            print(f"  Initializing strategy for camera: {cam_id} (Type: {self.model_type})")
            try:
                if self.model_type.lower() == 'yolo': strategy = YoloStrategy(self.model_path)
                elif self.model_type.lower() == 'rtdetr': strategy = RTDetrStrategy(self.model_path)
                elif self.model_type.lower() == 'fasterrcnn': strategy = FasterRCNNStrategy(self.model_path)
                else: raise ValueError(f"Unsupported model_type: {self.model_type}")
                self.strategies_per_camera[cam_id] = strategy
            except Exception as e: raise RuntimeError(f"Failed to initialize strategy for camera {cam_id}") from e
        print("All camera strategies initialized.")

    def select_person(self, track_id: Optional[int]) -> str:
         # Handle selecting/deselecting based on numerical track IDs
         # We won't allow selecting the string-based detection keys
        if isinstance(track_id, str) and track_id.startswith("det_"):
             return "Cannot select simple detections, only tracked IDs."

        if track_id is None:
            if self.selected_track_id is not None:
                deselected_id = self.selected_track_id
                self.selected_track_id = None
                return f"Deselected person {deselected_id}"
            else:
                return "No person was selected."

        if not isinstance(track_id, int): # Ensure it's an integer ID
             return "Invalid ID format for selection."

        if self.selected_track_id == track_id:
            self.selected_track_id = None
            if track_id == -1: return "Cannot track placeholder detections."
            return f"Deselected person {track_id}"
        else:
            if track_id == -1: return "Cannot select placeholder detections."
            self.selected_track_id = track_id
            return f"Selected person {track_id}"


    def update_person_crops(self, frame: np.ndarray, camera_id: str = '') -> None:
        """
        Updates the stored cropped images for the gallery based on current detections
        for a specific camera. Handles both tracked IDs (int) and simple detections (str key).
        """
        if self.frame_height is None or self.frame_width is None:
             if frame is not None and frame.size > 0: self.frame_height, self.frame_width = frame.shape[:2]
             else: return # Cannot crop

        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])

        if camera_id not in self.person_crops: self.person_crops[camera_id] = {}

        # Use enumerate to get index for generating temporary detection keys
        for index, (box, track_id) in enumerate(zip(boxes, track_ids)):

            # Determine the key to use for storing the crop
            if track_id != -1:
                storage_key = track_id # Use the actual track ID (int)
            else:
                # Generate a temporary key for simple detections (FasterRCNN)
                storage_key = f"det_{index}" # Use index within this frame/camera

            # --- Cropping Logic (mostly unchanged) ---
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4: continue # Skip invalid box data
            x, y, w, h = box
            if w <= 0 or h <= 0: continue # Skip invalid dimensions

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)

            if x2 > x1 and y2 > y1:
                if frame is None or frame.size == 0: continue
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0: continue

                target_height = 120
                aspect_ratio = w / h if h > 0 else 1
                crop_width = max(1, int(target_height * aspect_ratio))
                try:
                    crop_resized = cv2.resize(crop, (crop_width, target_height))
                    # Store the RGB crop using the determined storage_key (int or str)
                    self.person_crops[camera_id][storage_key] = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    print(f"Warning [update_person_crops]: cv2.resize failed for key {storage_key} on cam {camera_id}. Error: {e}")
                    continue
            # --- End Cropping Logic ---

        # Crop removal logic remains removed/commented out for accumulation behavior

    # (clear_old_person_crops - unchanged, handles int IDs)
    def clear_old_person_crops(self, current_frame_detections: Dict[str, Set[int]]):
         """ ... """ # (code omitted for brevity, assumes it only targets integer IDs)
         all_currently_visible_ids = set()
         for cam_id, ids_in_cam in current_frame_detections.items():
              # Only consider integer track IDs for cleanup
              all_currently_visible_ids.update(tid for tid in ids_in_cam if isinstance(tid, int))
         if self.selected_track_id is not None: all_currently_visible_ids.add(self.selected_track_id)
         cameras_to_check = list(self.person_crops.keys())
         for cam_id in cameras_to_check:
              keys_to_check_in_cam = list(self.person_crops[cam_id].keys())
              for key in keys_to_check_in_cam:
                   # Only remove integer track IDs that are no longer visible
                   if isinstance(key, int) and key not in all_currently_visible_ids:
                        if cam_id in self.person_crops and key in self.person_crops[cam_id]:
                             self.person_crops[cam_id].pop(key, None)
              if cam_id in self.person_crops and not self.person_crops[cam_id]: self.person_crops.pop(cam_id, None)


    # (process_frame - unchanged from last version)
    def process_frame(self, frame: np.ndarray, paused: bool = False, camera_id: str = '') -> Tuple[ Optional[np.ndarray], Optional[np.ndarray]]:
        # ... (implementation unchanged) ...
        if frame is None:
            # ... (handle None frame) ...
            return blank_annotated_rgb, blank_map_rgb
        if self.H is None:
            # ... (handle homography) ...
             if not (self.frame_height and self.frame_width): # If homography fails/no frame dims
                  # ... (return error map) ...
                  return cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB), cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)
             # ... (try compute_homography) ...
             # ... (handle homography error) ...

        strategy = self.strategies_per_camera.get(camera_id)
        if not strategy:
            # ... (handle no strategy) ...
            return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

        if not paused:
            try:
                boxes_xywh, track_ids, confidences = strategy.process_frame(frame)
                self.current_boxes[camera_id] = boxes_xywh
                self.current_track_ids[camera_id] = track_ids
                self.current_confidences[camera_id] = confidences
                # This now correctly handles track_id == -1 internally
                self.update_person_crops(frame, camera_id)
            except Exception as e:
                print(f"Error processing frame with {self.model_type} strategy for camera {camera_id}: {e}")
                self.current_boxes[camera_id] = []
                self.current_track_ids[camera_id] = []
                self.current_confidences[camera_id] = []

        annotated_frame = self.create_annotated_frame(frame.copy(), camera_id)
        map_img = None
        if self.H is not None and self.dst_points is not None:
            # ... (update history, create map) ...
            current_cam_boxes = self.current_boxes.get(camera_id, [])
            current_cam_track_ids = self.current_track_ids.get(camera_id, [])
            temp_history_update = defaultdict(list)
            for box, track_id in zip(current_cam_boxes, current_cam_track_ids):
                 if isinstance(track_id, int) and track_id != -1: # Only update history for tracked IDs
                    x, cy, w, h = box; y = cy + h / 2
                    if w > 0 and h > 0: temp_history_update[track_id].append((float(x), float(y)))
            for track_id, points in temp_history_update.items():
                self.track_history[track_id].extend(points)
                if len(self.track_history[track_id]) > 30: self.track_history[track_id] = self.track_history[track_id][-30:]
            map_img = create_map_visualization( self.map_width, self.map_height, self.dst_points, current_cam_boxes, current_cam_track_ids, self.track_history, self.H, self.selected_track_id )
        else:
            # ... (create blank map) ...
            map_img = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) if annotated_frame is not None else None
        map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB) if map_img is not None else None
        return annotated_frame_rgb, map_img_rgb


    # (create_annotated_frame - unchanged)
    def create_annotated_frame(self, frame: np.ndarray, camera_id: str = '') -> np.ndarray:
        # ... (implementation unchanged) ...
        if frame is None: # ... (handle None frame) ...
             return placeholder
        if self.src_points is not None: # ... (draw ROI) ...
             pass
        strategy_exists = camera_id in self.strategies_per_camera
        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])
        confidences = self.current_confidences.get(camera_id, [])
        if len(confidences) != len(boxes): confidences = [0.0] * len(boxes)
        for box, track_id, conf in zip(boxes, track_ids, confidences):
             # ... (check box, skip bad dims) ...
             if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4: continue
             x, cy, w, h = box;
             if w <= 0 or h <= 0: continue
             # Determine selection based on integer track ID only
             is_selected = isinstance(track_id, int) and (track_id == self.selected_track_id and track_id != -1)
             color = (0, 0, 255) if is_selected else ((255, 0, 0) if track_id == -1 else (0, 255, 0))
             thickness = 3 if is_selected else 2
             x1, y1 = int(x - w / 2), int(cy - h / 2); x2, y2 = int(x + w / 2), int(cy + h / 2)
             if x1 < x2 and y1 < y2:
                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                 id_text = f"ID:{track_id}" if isinstance(track_id, int) and track_id != -1 else "Det" # Label -1 as Det
                 conf_text = f":{conf:.2f}"; text_y = y1 - 5 if y1 > 10 else y1 + 10
                 cv2.putText(frame, id_text + conf_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        # ... (draw status text) ...
        status = f"Tracking ({self.model_type.upper()} - Cam: {camera_id})"
        if not strategy_exists: status = f"No Strategy Loaded (Cam: {camera_id})"
        if self.selected_track_id is not None: status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame


    # (process_multiple_frames - unchanged from last version)
    def process_multiple_frames(self, frames: Dict[str, np.ndarray], paused: bool = False) -> Tuple[ Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        # ... (implementation unchanged) ...
        annotated_frames_rgb = {}
        all_processed_boxes_for_map = []
        all_processed_track_ids_for_map = []
        detections_this_step = defaultdict(set)

        # Optional: Clear crops at start
        # self.person_crops.clear()

        for camera_id, frame_bgr in frames.items():
            annotated_frame_rgb, _ = self.process_frame(frame_bgr, paused, camera_id)
            annotated_frames_rgb[camera_id] = annotated_frame_rgb
            current_cam_boxes = self.current_boxes.get(camera_id, [])
            current_cam_track_ids = self.current_track_ids.get(camera_id, [])
            all_processed_boxes_for_map.extend(current_cam_boxes)
            all_processed_track_ids_for_map.extend(current_cam_track_ids)
            # Collect integer IDs for potential cleanup
            detections_this_step[camera_id].update(tid for tid in current_cam_track_ids if isinstance(tid, int) and tid != -1)

        # Optional: Cleanup old crops
        # self.clear_old_person_crops(detections_this_step)

        map_img_rgb = None
        if self.H is not None and self.dst_points is not None:
            # ... (create combined map) ...
            map_img_bgr = create_map_visualization( self.map_width, self.map_height, self.dst_points, all_processed_boxes_for_map, all_processed_track_ids_for_map, self.track_history, self.H, self.selected_track_id )
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB) if map_img_bgr is not None else None
        else:
            # ... (create blank map) ...
            map_img_bgr = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img_bgr, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)

        return annotated_frames_rgb, map_img_rgb