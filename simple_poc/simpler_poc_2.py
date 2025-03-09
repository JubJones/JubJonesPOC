import cv2
import numpy as np
import gradio as gr
from collections import defaultdict
from ultralytics import YOLO
import time
import threading


class PersonTracker:
    def __init__(self, model_path):
        # Init model
        self.model = YOLO(model_path)

        # Tracking setup
        self.track_history = defaultdict(list)
        self.map_width, self.map_height = 600, 800

        # UI state
        self.selected_track_id = None
        self.current_boxes = []
        self.current_track_ids = []
        self.person_crops = {}  # Store cropped images of detected people

    def compute_homography(self, frame_width, frame_height):
        src_points = np.float32([
            [frame_width * 0.1, frame_height * 0.1],
            [frame_width * 0.9, frame_height * 0.1],
            [frame_width * 0.9, frame_height * 0.95],
            [frame_width * 0.1, frame_height * 0.95]
        ])

        dst_points = np.float32([
            [0, 0],
            [self.map_width, 0],
            [self.map_width, self.map_height],
            [0, self.map_height]
        ])

        return cv2.getPerspectiveTransform(src_points, dst_points), src_points, dst_points

    def select_person(self, track_id):
        """Select a person to track by ID"""
        if self.selected_track_id == track_id:
            self.selected_track_id = None
            return f"Deselected person {track_id}"
        else:
            self.selected_track_id = track_id
            return f"Selected person {track_id}"

    def update_person_crops(self, frame):
        """Update the person crops dictionary with the latest detections"""
        current_ids = set()

        for box, track_id in zip(self.current_boxes, self.current_track_ids):
            current_ids.add(track_id)
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width - 1, x2), min(self.frame_height - 1, y2)

            if x2 > x1 and y2 > y1:  # Valid crop dimensions
                crop = frame[y1:y2, x1:x2].copy()
                # Resize crop to fit in sidebar (maintain aspect ratio)
                target_height = 120
                aspect_ratio = w / h
                crop_width = int(target_height * aspect_ratio)
                crop_resized = cv2.resize(crop, (crop_width, target_height))
                self.person_crops[track_id] = crop_resized

        # Remove crops for people no longer detected
        ids_to_remove = [id for id in self.person_crops.keys() if id not in current_ids]
        for id in ids_to_remove:
            if id != self.selected_track_id:  # Keep selected person even if temporarily not detected
                self.person_crops.pop(id, None)

    def process_frame(self, frame, paused=False):
        """Process a single frame and return the annotated frame and map"""
        if frame is None:
            return None, None

        # Get dimensions for homography if not already set
        if not hasattr(self, 'H'):
            self.frame_height, self.frame_width = frame.shape[:2]
            self.H, self.src_points, self.dst_points = self.compute_homography(self.frame_width, self.frame_height)

        if not paused:
            # Run tracking
            results = self.model.track(frame, persist=True)

            # Store current detection data
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                self.current_boxes = results[0].boxes.xywh.cpu()
                self.current_track_ids = results[0].boxes.id.int().cpu().tolist()

                # Update person crops
                self.update_person_crops(frame)

        # Create visualization
        annotated_frame = frame.copy()
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

            # If we have a selected person and this isn't them, skip drawing
            if self.selected_track_id is not None and not is_selected:
                continue

            color = (0, 0, 255) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2

            # Draw bounding box with ID
            x1, y1 = int(x - w / 2), int(cy - h / 2)
            x2, y2 = int(x + w / 2), int(cy + h / 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

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
            # Skip if we have a selected person and this isn't them
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

        # Show status text
        status = "Tracking"
        if self.selected_track_id is not None:
            status += f" | Selected ID: {self.selected_track_id}"
        cv2.putText(annotated_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated_frame, img_map


def create_gradio_ui():
    model_path = "yolo11n.pt"  # Update with your model path
    tracker = PersonTracker(model_path)

    # Video playback state
    video_state = {
        "cap": None,
        "total_frames": 0,
        "current_frame": 0,
        "playing": False,
        "video_path": None,
        "play_thread": None,
        "stop_thread": False
    }

    def process_video(video_path):
        """Process video and return frames for display"""
        # Close any previously opened video
        if video_state["cap"] is not None:
            video_state["cap"].release()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Store video state
        video_state["cap"] = cap
        video_state["video_path"] = video_path
        video_state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_state["current_frame"] = 0

        # Read first frame to initialize
        success, frame = cap.read()
        if not success:
            return None, None, []

        # Process the frame
        annotated_frame, map_img = tracker.process_frame(frame)

        return annotated_frame, map_img

    def get_frame(frame_index, video_path):
        """Get a specific frame from the video"""
        if video_state["cap"] is None or video_path != video_state["video_path"]:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            video_state["cap"] = cap
            video_state["video_path"] = video_path
            video_state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            cap = video_state["cap"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        return frame if success else None

    def update_frame(video_path, frame_index, paused):
        """Update the current frame based on slider position"""
        if not video_path:
            return None, None, gr.HTML("")

        frame = get_frame(frame_index, video_path)
        if frame is None:
            return None, None, gr.HTML("<p>Error: Could not read frame</p>")

        # Process the frame
        annotated_frame, map_img = tracker.process_frame(frame, paused)

        # Create HTML for person gallery with click handlers
        gallery_html = create_gallery_html(tracker.person_crops, tracker.selected_track_id)

        return annotated_frame, map_img, gallery_html

    def create_gallery_html(person_crops, selected_track_id):
        """Create HTML for the person gallery with working buttons"""
        if not person_crops:
            return ""

        gallery_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"

        for track_id, crop in person_crops.items():
            # Convert crop to base64 for embedding in HTML
            _, buffer = cv2.imencode('.jpg', crop)
            img_str = buffer.tobytes()
            import base64
            img_base64 = base64.b64encode(img_str).decode('utf-8')

            # Determine if this person is selected
            is_selected = (track_id == selected_track_id)
            border_style = "border: 3px solid red;" if is_selected else "border: 1px solid #ddd;"

            # Create a div for each person with image and button
            gallery_html += f"""
            <div style='text-align: center; margin-bottom: 10px;'>
                <img src='data:image/jpeg;base64,{img_base64}'
                     style='width: auto; height: 120px; {border_style}'>
                <br>
                <button
                    onclick='document.getElementById("track_button_{track_id}").click()'
                    style='margin-top: 5px; padding: 5px 10px;
                           background-color: {"#ff6b6b" if is_selected else "#4CAF50"};
                           color: white; border: none; border-radius: 4px; cursor: pointer;'>
                    {"Untrack" if is_selected else "Track"} ID: {track_id}
                </button>
            </div>
            """

        gallery_html += "</div>"
        return gallery_html

    def track_person(track_id):
        """Track a specific person by ID"""
        result = tracker.select_person(int(track_id))
        return result

    def play_thread_function():
        """Background thread function for video playback"""
        fps = 15  # Target frames per second
        frame_time = 1.0 / fps

        while not video_state["stop_thread"] and video_state["playing"]:
            start_time = time.time()

            # Skip if we've reached the end
            if video_state["current_frame"] >= video_state["total_frames"] - 1:
                video_state["playing"] = False
                break

            # Increment the frame counter
            video_state["current_frame"] += 1

            # Sleep to maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

    def toggle_playback(paused):
        """Toggle video playback"""
        video_state["playing"] = not paused

        # If we're starting playback, create a new thread
        if video_state["playing"]:
            if video_state["play_thread"] is not None:
                video_state["stop_thread"] = True
                if video_state["play_thread"].is_alive():
                    video_state["play_thread"].join(1.0)

            video_state["stop_thread"] = False
            video_state["play_thread"] = threading.Thread(
                target=play_thread_function,
                daemon=True
            )
            video_state["play_thread"].start()

        return f"Video {'paused' if paused else 'playing'}"

    def advance_frame(video_path, frame_slider, paused):
        """Advance to the next frame"""
        if not video_path:
            return gr.update(), "No video loaded"

        next_frame = min(frame_slider + 1, video_state["total_frames"] - 1)
        if next_frame == video_state["total_frames"] - 1:
            video_state["playing"] = False
            return gr.update(value=next_frame), "End of video reached"

        return gr.update(value=next_frame), f"Advanced to frame {next_frame}"

    def check_playback_progress(video_path, frame_slider, paused):
        """Check playback progress and update UI"""
        if not video_state["playing"] or paused:
            return gr.update(), gr.update(), gr.update()

        # If playing, update slider to match current frame
        current = video_state["current_frame"]
        if current != frame_slider and current < video_state["total_frames"]:
            # Get and process the current frame
            frame = get_frame(current, video_path)
            if frame is not None:
                annotated_frame, map_img = tracker.process_frame(frame, False)
                gallery_html = create_gallery_html(tracker.person_crops, tracker.selected_track_id)
                return gr.update(value=current), annotated_frame, map_img, gallery_html

        return gr.update(), gr.update(), gr.update(), gr.update()

    def on_start(video_input, video_path):
        """Start video playback"""
        path = video_input if video_input else video_path
        annotated_frame, map_img = process_video(path)
        if annotated_frame is None:
            return "Error: Could not open video", gr.update(), gr.update(value=True), None, None, gr.HTML("")

        # Set playing state
        video_state["playing"] = True
        video_state["current_frame"] = 0

        # Start the playback thread
        toggle_playback(False)

        return (
            "Video loaded and playing",
            gr.update(maximum=video_state["total_frames"]-1, value=0),
            gr.update(value=False),
            annotated_frame,
            map_img,
            create_gallery_html(tracker.person_crops, tracker.selected_track_id)
        )

    def on_clear():
        """Clear person selection"""
        tracker.selected_track_id = None
        return "Cleared selection"

    def update_ui_from_frame(frame_index, video_path, paused):
        """Update UI elements based on current frame"""
        frame = get_frame(frame_index, video_path)
        if frame is None:
            return None, None, gr.HTML("<p>Error reading frame</p>")

        # Process the frame
        annotated_frame, map_img = tracker.process_frame(frame, paused)

        # Create gallery HTML
        gallery_html = create_gallery_html(tracker.person_crops, tracker.selected_track_id)

        return annotated_frame, map_img, gallery_html

    with gr.Blocks() as demo:
        gr.Markdown("# Person Tracking with YOLO")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")
                video_path = gr.Textbox(label="Or enter video path", value="test.avi")

                with gr.Row():
                    start_btn = gr.Button("Start Tracking")
                    pause_checkbox = gr.Checkbox(label="Pause", value=True)

                with gr.Row():
                    frame_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="Frame Position"
                    )
                    next_frame_btn = gr.Button("Next Frame")

                clear_btn = gr.Button("Clear Selection")
                refresh_btn = gr.Button("Refresh Display")

                # Hidden buttons for person tracking - IMPORTANT for button clicks to work
                with gr.Column(visible=False):
                    track_buttons = {i: gr.Button(f"Track {i}", elem_id=f"track_button_{i}")
                                    for i in range(1, 50)}  # Support up to 50 track IDs

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Tracking View"):
                        video_output = gr.Image(label="Tracking")
                    with gr.TabItem("Map View"):
                        map_output = gr.Image(label="Map")

        gr.Markdown("## Detected People")
        gallery_output = gr.HTML()
        status_output = gr.Textbox(label="Status")

        # Event handlers
        start_btn.click(
            on_start,
            inputs=[video_input, video_path],
            outputs=[status_output, frame_slider, pause_checkbox, video_output, map_output, gallery_output]
        )

        pause_checkbox.change(
            toggle_playback,
            inputs=[pause_checkbox],
            outputs=[status_output]
        )

        frame_slider.change(
            update_ui_from_frame,
            inputs=[frame_slider, video_path, pause_checkbox],
            outputs=[video_output, map_output, gallery_output]
        )

        next_frame_btn.click(
            advance_frame,
            inputs=[video_path, frame_slider, pause_checkbox],
            outputs=[frame_slider, status_output]
        ).then(
            update_ui_from_frame,
            inputs=[frame_slider, video_path, pause_checkbox],
            outputs=[video_output, map_output, gallery_output]
        )

        clear_btn.click(
            on_clear,
            outputs=[status_output]
        ).then(
            update_ui_from_frame,
            inputs=[frame_slider, video_path, pause_checkbox],
            outputs=[video_output, map_output, gallery_output]
        )

        # Refresh button to manually update the display based on current playback position
        refresh_btn.click(
            check_playback_progress,
            inputs=[video_path, frame_slider, pause_checkbox],
            outputs=[frame_slider, video_output, map_output, gallery_output]
        )

        # Connect track buttons to select specific persons
        for i, btn in track_buttons.items():
            btn.click(
                track_person,
                inputs=gr.Number(value=i, visible=False),
                outputs=[status_output]
            ).then(
                update_ui_from_frame,
                inputs=[frame_slider, video_path, pause_checkbox],
                outputs=[video_output, map_output, gallery_output]
            )

    return demo


if __name__ == "__main__":
    demo = create_gradio_ui()
    demo.launch(share=True)
