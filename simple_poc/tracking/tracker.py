from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from simple_poc.tracking.homography import compute_homography


class PersonTracker:
    def __init__(self, model_path, map_width=600, map_height=800):
        self.model = YOLO(model_path)
        self.track_history = defaultdict(list)
        self.map_width, self.map_height = map_width, map_height
        self.selected_track_id = None
        self.current_boxes = []
        self.current_track_ids = []
        self.person_crops = {}
        self.H = None

    def select_person(self, track_id):
        if self.selected_track_id == track_id:
            self.selected_track_id = None
            return f"Deselected person {track_id}"
        self.selected_track_id = track_id
        return f"Selected person {track_id}"

    def update_person_crops(self, frame):
        current_ids = set()
        for box, track_id in zip(self.current_boxes, self.current_track_ids):
            current_ids.add(track_id)
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                target_height = 120
                aspect_ratio = w / h
                crop_width = int(target_height * aspect_ratio)
                crop_resized = cv2.resize(crop, (crop_width, target_height))
                self.person_crops[track_id] = crop_resized

        # Remove crops for people no longer detected
        ids_to_remove = [id for id in self.person_crops.keys() if id not in current_ids]
        for id in ids_to_remove:
            if id != self.selected_track_id:
                self.person_crops.pop(id, None)

    def process_frame(self, frame, paused=False):
        if frame is None:
            return None, None

        # Initialize homography if needed
        if self.H is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.H, self.src_points, self.dst_points = compute_homography(
                self.frame_width, self.frame_height, self.map_width, self.map_height
            )

        if not paused:
            # Run tracking
            results = self.model.track(frame, persist=True)
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                self.current_boxes = results[0].boxes.xywh.cpu()
                self.current_track_ids = results[0].boxes.id.int().cpu().tolist()
                self.update_person_crops(frame)

        # Create visualizations
        annotated_frame = self.create_annotated_frame(frame.copy())
        map_img = self.create_map_visualization()

        return annotated_frame, map_img

    def create_annotated_frame(self, frame):
        # Draw ROI polygon
        cv2.polylines(frame, [self.src_points.astype(np.int32).reshape((-1, 1, 2))],
                      True, (0, 255, 255), 2)

        # Process each tracked person
        for box, track_id in zip(self.current_boxes, self.current_track_ids):
            if self.selected_track_id is not None and track_id != self.selected_track_id:
                continue

            x, cy, w, h = box
            is_selected = (track_id == self.selected_track_id)
            color = (0, 0, 255) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2

            # Draw bounding box with ID
            x1, y1 = int(x - w / 2), int(cy - h / 2)
            x2, y2 = int(x + w / 2), int(cy + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Show status text
        status = "Tracking"
        if self.selected_track_id is not None:
            status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def create_map_visualization(self):
        img_map = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)

        # Draw ROI polygon on map
        cv2.polylines(img_map, [self.dst_points.astype(np.int32).reshape((-1, 1, 2))],
                      True, (255, 0, 0), 2)

        # Draw tracked people and their trajectories
        self._draw_tracked_people_on_map(img_map)
        self._draw_trajectories_on_map(img_map)

        return img_map

    def _draw_tracked_people_on_map(self, img_map):
        for box, track_id in zip(self.current_boxes, self.current_track_ids):
            if self.selected_track_id is not None and track_id != self.selected_track_id:
                continue

            x, cy, w, h = box
            y = cy + h / 2  # Bottom center
            is_selected = (track_id == self.selected_track_id)
            color = (0, 0, 255) if is_selected else (0, 255, 0)

            # Update track history
            self.track_history[track_id].append((float(x), float(y)))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Transform to map coordinates and plot
            bottom_center = np.array([[[float(x), float(y)]]], dtype=np.float32)
            pt_transformed = cv2.perspectiveTransform(bottom_center, self.H)
            pt_mapped = (int(pt_transformed[0][0][0]), int(pt_transformed[0][0][1]))

            # Draw on map
            radius = 7 if is_selected else 5
            cv2.circle(img_map, pt_mapped, radius, color, -1)
            cv2.putText(img_map, str(track_id), (pt_mapped[0] + 5, pt_mapped[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_trajectories_on_map(self, img_map):
        for track_id, track in self.track_history.items():
            if self.selected_track_id is not None and track_id != self.selected_track_id:
                continue

            if len(track) >= 2:
                is_selected = (track_id == self.selected_track_id)
                color = (0, 0, 255) if is_selected else (200, 200, 200)
                thickness = 3 if is_selected else 2

                track_np = np.array(track, dtype=np.float32).reshape(-1, 1, 2)
                track_transformed = cv2.perspectiveTransform(track_np, self.H)
                points = track_transformed.reshape(-1, 2).astype(np.int32)
                cv2.polylines(img_map, [points], False, color, thickness)
