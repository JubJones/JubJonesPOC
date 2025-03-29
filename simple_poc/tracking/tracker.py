from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from simple_poc.tracking.map import (
    compute_homography, create_map_visualization
)
from simple_poc.tracking.strategies import (
    DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy
)


class PersonTracker:
    """Handles detection, tracking (via strategy), and map visualization."""

    def __init__(self, model_path: str, model_type: str, map_width: int = 600, map_height: int = 800):
        self.model_path = model_path
        self.model_type = model_type

        print(f"Initializing SINGLE strategy instance. Type: {self.model_type}, Path: {self.model_path}")
        try:
            if self.model_type.lower() == 'yolo':
                self.strategy: DetectionTrackingStrategy = YoloStrategy(self.model_path)
            elif self.model_type.lower() == 'rtdetr':
                self.strategy: DetectionTrackingStrategy = RTDetrStrategy(self.model_path)
            elif self.model_type.lower() == 'fasterrcnn':
                self.strategy: DetectionTrackingStrategy = FasterRCNNStrategy(self.model_path)
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
            print("Strategy initialized.")
        except Exception as e:
             print(f"FATAL ERROR initializing strategy: {e}")
             raise RuntimeError(f"Failed to initialize strategy") from e
        # ---------------------------------------------

        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list) # Global history
        self.map_width: int = map_width
        self.map_height: int = map_height
        self.selected_track_id: Optional[int] = None

        self.current_boxes: Dict[str, List[np.ndarray]] = defaultdict(list) # Stores last results per camera
        self.current_track_ids: Dict[str, List[int]] = defaultdict(list)
        self.current_confidences: Dict[str, List[float]] = defaultdict(list)
        self.person_crops: Dict[str, Dict[Union[int, str], np.ndarray]] = defaultdict(dict) # Stores RGB crops per cam / track_or_det_key

        self.H: Optional[np.ndarray] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None


    def select_person(self, track_id: Optional[int]) -> str:
        if isinstance(track_id, str) and track_id.startswith("det_"): return "Cannot select simple detections."
        if track_id is None:
            if self.selected_track_id is not None:
                deselected_id = self.selected_track_id
                self.selected_track_id = None
                return f"Deselected person {deselected_id}"
            else: return "No person was selected."
        if not isinstance(track_id, int): return "Invalid ID format for selection."
        if self.selected_track_id == track_id:
            self.selected_track_id = None
            if track_id == -1: return "Cannot track placeholder detections."
            return f"Deselected person {track_id}"
        else:
            if track_id == -1: return "Cannot select placeholder detections."
            self.selected_track_id = track_id
            return f"Selected person {track_id}"


    def update_person_crops(self, frame: np.ndarray, camera_id: str = '') -> None:
        """Generates and stores crops for gallery (handles tracked IDs and detections)."""
        if self.frame_height is None or self.frame_width is None:
             if frame is not None and frame.size > 0: self.frame_height, self.frame_width = frame.shape[:2]
             else: return

        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])
        if camera_id not in self.person_crops: self.person_crops[camera_id] = {}

        for index, (box, track_id) in enumerate(zip(boxes, track_ids)):
            storage_key = track_id if track_id != -1 else f"det_{index}"

            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4: continue
            x, y, w, h = box
            if w <= 0 or h <= 0: continue

            x1, y1 = int(x - w / 2), int(y - h / 2); x2, y2 = int(x + w / 2), int(y + h / 2)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)

            if x2 > x1 and y2 > y1:
                if frame is None or frame.size == 0: continue
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0: continue
                target_height = 120
                aspect_ratio = w / h if h > 0 else 1
                crop_width = max(1, int(target_height * aspect_ratio))
                try:
                    crop_resized = cv2.resize(crop, (crop_width, target_height))
                    self.person_crops[camera_id][storage_key] = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB) # Store RGB
                except cv2.error as e:
                    print(f"Warning [update_person_crops]: cv2.resize failed key {storage_key} cam {camera_id}: {e}")
                    continue


    def process_frame(self, frame: np.ndarray, paused: bool = False, camera_id: str = '') -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """Processes a single frame using the SHARED strategy."""
        if frame is None:
            # Handle None frame (create placeholders)
            blank_h = self.frame_height or 480; blank_w = self.frame_width or 640
            blank_annotated = np.zeros((blank_h, blank_w, 3), dtype=np.uint8)
            blank_map = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(blank_annotated, f"No Frame ({camera_id})", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
            cv2.putText(blank_map, f"No Frame ({camera_id})", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            return cv2.cvtColor(blank_annotated, cv2.COLOR_BGR2RGB), cv2.cvtColor(blank_map, cv2.COLOR_BGR2RGB)

        # Initialize homography on first valid frame
        if self.H is None:
            if self.frame_height is None or self.frame_width is None:
                 if frame.size > 0: self.frame_height, self.frame_width = frame.shape[:2]
            if not (self.frame_height and self.frame_width):
                 error_map = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8) # Grey map
                 cv2.putText(error_map, "Homography Error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                 return cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB), cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)
            try:
                self.H, self.src_points, self.dst_points = compute_homography(self.frame_width, self.frame_height, self.map_width, self.map_height)
                if self.H is None: raise ValueError("Homography failed")
            except Exception as e:
                print(f"Error computing homography: {e}"); self.H = None
                error_map = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8)
                cv2.putText(error_map, "Homography Error", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                return cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB), cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

        # Process frame using the single strategy instance if not paused
        if not paused:
            try:
                # Use the single shared strategy instance
                boxes_xywh, track_ids, confidences = self.strategy.process_frame(frame)
                self.current_boxes[camera_id] = boxes_xywh
                self.current_track_ids[camera_id] = track_ids
                self.current_confidences[camera_id] = confidences
                self.update_person_crops(frame, camera_id) # Update crops based on these results
            except Exception as e:
                print(f"Error processing frame with {self.model_type} strategy for camera {camera_id}: {e}")
                self.current_boxes[camera_id] = []
                self.current_track_ids[camera_id] = []
                self.current_confidences[camera_id] = []

        # Create annotated frame and map visualization for this camera view
        annotated_frame = self.create_annotated_frame(frame.copy(), camera_id)
        map_img = None
        if self.H is not None and self.dst_points is not None:
            current_cam_boxes = self.current_boxes.get(camera_id, [])
            current_cam_track_ids = self.current_track_ids.get(camera_id, [])

            # Update global track history (only for integer track IDs)
            temp_history_update = defaultdict(list)
            for box, track_id in zip(current_cam_boxes, current_cam_track_ids):
                 if isinstance(track_id, int) and track_id != -1:
                    x, cy, w, h = box; y = cy + h / 2 # Use bottom-center point for map position
                    if w > 0 and h > 0: temp_history_update[track_id].append((float(x), float(y)))
            for track_id, points in temp_history_update.items():
                self.track_history[track_id].extend(points)
                if len(self.track_history[track_id]) > 30: self.track_history[track_id] = self.track_history[track_id][-30:] # Limit history

            map_img = create_map_visualization(
                self.map_width, self.map_height, self.dst_points,
                current_cam_boxes, current_cam_track_ids,
                self.track_history, self.H, self.selected_track_id
            )
        else:
            map_img = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) if annotated_frame is not None else None
        map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB) if map_img is not None else None
        return annotated_frame_rgb, map_img_rgb


    def create_annotated_frame(self, frame: np.ndarray, camera_id: str = '') -> np.ndarray:
        """Draws boxes, IDs, ROI, and status text on the frame."""
        if frame is None:
             h = self.frame_height or 480; w = self.frame_width or 640
             placeholder = np.zeros((h, w, 3), dtype=np.uint8)
             cv2.putText(placeholder, f"Invalid Frame ({camera_id})", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             return placeholder

        # Draw ROI
        if self.src_points is not None:
            try: cv2.polylines(frame, [self.src_points.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 255), 2)
            except Exception as e: print(f"Warning: Could not draw ROI polygon. Error: {e}")

        boxes = self.current_boxes.get(camera_id, [])
        track_ids = self.current_track_ids.get(camera_id, [])
        confidences = self.current_confidences.get(camera_id, [])
        if len(confidences) != len(boxes): confidences = [0.0] * len(boxes)

        # Draw boxes and IDs
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4: continue
            x, cy, w, h = box
            if w <= 0 or h <= 0: continue
            is_selected = isinstance(track_id, int) and (track_id == self.selected_track_id and track_id != -1)
            color = (0, 0, 255) if is_selected else ((255, 0, 0) if track_id == -1 else (0, 255, 0)) # Red selected, Blue det, Green track
            thickness = 3 if is_selected else 2
            x1, y1 = int(x - w / 2), int(cy - h / 2); x2, y2 = int(x + w / 2), int(cy + h / 2)
            if x1 < x2 and y1 < y2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                id_text = f"ID:{track_id}" if isinstance(track_id, int) and track_id != -1 else "Det"
                conf_text = f":{conf:.2f}"; text_y = y1 - 5 if y1 > 10 else y1 + 10
                cv2.putText(frame, id_text + conf_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Draw Status Text
        status = f"Tracking ({self.model_type.upper()} - Cam: {camera_id})"
        if self.selected_track_id is not None: status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red status text
        return frame


    def process_multiple_frames(self, frames: Dict[str, np.ndarray], paused: bool = False) -> Tuple[
        Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        """Processes multiple frames using the shared strategy, returns annotated frames and combined map."""
        annotated_frames_rgb = {}
        all_processed_boxes_for_map = []
        all_processed_track_ids_for_map = []

        # Process each camera using the single shared strategy via process_frame
        for camera_id, frame_bgr in frames.items():
            annotated_frame_rgb, _ = self.process_frame(frame_bgr, paused, camera_id)
            annotated_frames_rgb[camera_id] = annotated_frame_rgb

            # Collect results stored by process_frame for the combined map
            current_cam_boxes = self.current_boxes.get(camera_id, [])
            current_cam_track_ids = self.current_track_ids.get(camera_id, [])
            all_processed_boxes_for_map.extend(current_cam_boxes)
            all_processed_track_ids_for_map.extend(current_cam_track_ids)


        if self.H is not None and self.dst_points is not None:
            # History was updated within process_frame calls
            map_img_bgr = create_map_visualization(
                self.map_width, self.map_height, self.dst_points,
                all_processed_boxes_for_map,
                all_processed_track_ids_for_map,
                self.track_history,
                self.H,
                self.selected_track_id
            )
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB) if map_img_bgr is not None else None
        else:
            map_img_bgr = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)
            cv2.putText(map_img_bgr, "Waiting for Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            map_img_rgb = cv2.cvtColor(map_img_bgr, cv2.COLOR_BGR2RGB)

        # Gallery uses self.person_crops which was updated during process_frame calls
        return annotated_frames_rgb, map_img_rgb
