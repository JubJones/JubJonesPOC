from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


class PersonTracker:
    def __init__(self, model_path, video_path):
        # Init model and video
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")

        # Get dimensions
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise Exception("Unable to read from video source")
        self.frame_height, self.frame_width = frame.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Tracking setup
        self.track_history = defaultdict(list)
        self.map_width, self.map_height = 600, 800
        self.H, self.src_points, self.dst_points = self.compute_homography()

        # UI state
        self.selected_track_id = None
        self.paused = False
        self.current_boxes = np.array([])
        self.current_track_ids = []
        self.click_margin = 10
        self.last_frame = None
        self.processing_clicks = False  # Flag to prevent click processing during frame updates
        self.click_count = 0
        self.frame_count = 0

        # Setup windows and mouse callback
        cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL for better event handling
        cv2.setMouseCallback("YOLO Tracking", self.mouse_callback)

    def compute_homography(self):
        # Same as before
        src_points = np.float32([
            [self.frame_width * 0.1, self.frame_height * 0.1],
            [self.frame_width * 0.9, self.frame_height * 0.1],
            [self.frame_width * 0.9, self.frame_height * 0.95],
            [self.frame_width * 0.1, self.frame_height * 0.95]
        ])

        dst_points = np.float32([
            [0, 0],
            [self.map_width, 0],
            [self.map_width, self.map_height],
            [0, self.map_height]
        ])

        return cv2.getPerspectiveTransform(src_points, dst_points), src_points, dst_points

    def mouse_callback(self, event, x, y, flags, param):
        # Process click events more reliably
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_count += 1
            print(f"Click {self.click_count} at ({x}, {y}) - Frame {self.frame_count}")

            # Store click coordinates to process in main loop
            self.process_click(x, y)

    def process_click(self, x, y):
        """Process click coordinates in the main thread for better reliability"""
        if self.current_boxes.size == 0 or len(self.current_track_ids) == 0:
            print("No boxes available to click")
            return

        clicked = False

        # Check each box with a margin for easier clicking
        for i, (box, track_id) in enumerate(zip(self.current_boxes, self.current_track_ids)):
            cx, cy, w, h = box
            margin = self.click_margin

            # Add margin to make clicking easier
            x1, y1 = int(cx - w / 2) - margin, int(cy - h / 2) - margin
            x2, y2 = int(cx + w / 2) + margin, int(cy + h / 2) + margin

            # Debug info
            print(f"Box {track_id}: ({x1},{y1}) to ({x2},{y2})")

            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked = True
                # Toggle selection
                if self.selected_track_id == track_id:
                    self.selected_track_id = None
                    print(f"Deselected person {track_id}")
                else:
                    self.selected_track_id = track_id
                    print(f"Selected person {track_id}")
                break

        # If clicked outside all boxes, deselect
        if not clicked:
            self.selected_track_id = None
            print("Clicked outside, deselected all")

    def run(self):
        while self.cap.isOpened():
            self.frame_count += 1

            if not self.paused:
                success, frame = self.cap.read()
                if not success:
                    break

                # Store current frame
                self.last_frame = frame.copy()

                # Run tracking
                results = self.model.track(frame, persist=True)

                # Store current detection data with safety checks
                if (hasattr(results[0].boxes, 'id') and
                        results[0].boxes.id is not None and
                        len(results[0].boxes.id) > 0):
                    self.current_boxes = results[0].boxes.xywh.cpu().numpy()
                    self.current_track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    self.current_boxes = np.array([])
                    self.current_track_ids = []

            # Create visualization using last_frame
            if self.last_frame is None:
                continue

            annotated_frame = self.last_frame.copy()
            img_map = np.full((self.map_height, self.map_width, 3), 255, dtype=np.uint8)

            # Draw ROI polygon
            cv2.polylines(annotated_frame, [self.src_points.astype(np.int32).reshape((-1, 1, 2))],
                          True, (0, 255, 255), 2)
            cv2.polylines(img_map, [self.dst_points.astype(np.int32).reshape((-1, 1, 2))],
                          True, (255, 0, 0), 2)

            # Process each tracked person
            for box, track_id in zip(self.current_boxes, self.current_track_ids):
                x, cy, w, h = box
                y = cy + h / 2  # Bottom center

                # Determine if this is the selected person
                is_selected = (track_id == self.selected_track_id)
                color = (0, 0, 255) if is_selected else (0, 255, 0)
                thickness = 3 if is_selected else 2

                # Draw bounding box with ID
                x1, y1 = int(x - w / 2), int(cy - h / 2)
                x2, y2 = int(x + w / 2), int(cy + h / 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                # Draw clickable area
                margin = self.click_margin
                cv2.rectangle(annotated_frame,
                              (x1 - margin, y1 - margin),
                              (x2 + margin, y2 + margin),
                              (255, 255, 0), 1, cv2.LINE_AA)

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

            # Draw trajectories on map
            for track_id, track in self.track_history.items():
                if len(track) >= 2:
                    is_selected = (track_id == self.selected_track_id)
                    color = (0, 0, 255) if is_selected else (200, 200, 200)
                    thickness = 3 if is_selected else 2

                    track_np = np.array(track, dtype=np.float32).reshape(-1, 1, 2)
                    try:
                        track_transformed = cv2.perspectiveTransform(track_np, self.H)
                        points = track_transformed.reshape(-1, 2).astype(np.int32)
                        cv2.polylines(img_map, [points], False, color, thickness)
                    except cv2.error:
                        # Handle potential transform errors
                        pass

            # Show status text
            status = f"{'Paused' if self.paused else 'Playing'} | Frame: {self.frame_count} | Clicks: {self.click_count}"
            if self.selected_track_id is not None:
                status += f" | Tracking ID: {self.selected_track_id}"
            cv2.putText(annotated_frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show click instructions
            cv2.putText(annotated_frame, "Click on a person to select/deselect",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Display frames
            cv2.imshow("YOLO Tracking", annotated_frame)
            cv2.imshow("Map", img_map)

            # Important: Give ample time for event processing
            # This is the key to fixing your click issues
            key = cv2.waitKey(30) & 0xFF  # Increase wait time to 30ms

            if key == ord('q'):
                break
            elif key == ord(' '):  # Pause/play
                self.paused = not self.paused
                print(f"Video {'paused' if self.paused else 'resumed'}")
            elif key == ord('c'):  # Clear selection
                self.selected_track_id = None
                print("Selection cleared")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "yolo11n.pt"
    video_path = "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/test.avi"

    tracker = PersonTracker(model_path, video_path)
    tracker.run()
