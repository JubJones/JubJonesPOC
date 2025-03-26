# ================================================
# File: simple_poc/ui/app.py
# ================================================
import os
import cv2
import gradio as gr
import numpy as np
from collections import defaultdict # Added import

from simple_poc.tracking.tracker import PersonTracker
from simple_poc.ui.gallery import create_gallery_html
# Import map functions directly needed for GT mode map generation
from simple_poc.tracking.map import compute_homography, create_map_visualization


class MTMMCTrackerApp:
    def __init__(self, model_path="yolo11n.pt", map_width=400, map_height=600): # Added map dimensions
        self.tracker = PersonTracker(model_path, map_width=map_width, map_height=map_height)
        self.dataset_path = None
        self.camera_dirs = []
        self.current_frame_index = 0
        self.paused = True
        self.gt_data = {} # Key: camera_id, Value: dict {frame_id: [(person_id, x, y, w, h), ...]}
        self.mode = "Model Detection"

        # --- New attributes for GT map ---
        self.gt_track_history = defaultdict(lambda: defaultdict(list)) # Key: camera_id, Key: person_id, Value: list[(cx, cy_bottom)]
        self.H = None # Homography matrix
        self.src_points = None # Source points for homography
        self.dst_points = None # Destination points for homography
        self.map_width = map_width
        self.map_height = map_height
        self.frame_width = None # Store frame dimensions
        self.frame_height = None # Store frame dimensions
        # --- End new attributes ---

    def load_gt_data_for_camera(self, camera_dir):
        """Loads ground truth data from a gt.txt file for a specific camera."""
        gt_path = os.path.join(camera_dir, 'gt', 'gt.txt')
        gt_data_cam = {}
        try:
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        # print(f"Warning: Skipping malformed line in {gt_path}: {line.strip()}")
                        continue
                    # Use 1-based frame ID from file
                    frame_id, person_id, x, y, w, h = map(float, parts[:6])
                    frame_id = int(frame_id)
                    person_id = int(person_id)
                    if frame_id not in gt_data_cam:
                        gt_data_cam[frame_id] = []
                    # Store as (person_id, x, y, w, h) - using original top-left coords
                    gt_data_cam[frame_id].append((person_id, x, y, w, h))
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found at {gt_path}. No GT boxes or map points for this camera.")
            return {}
        except ValueError as e:
            print(f"Error parsing line in {gt_path}: {line.strip()} - {e}")
        return gt_data_cam

    def draw_gt_boxes(self, frame, frame_index, camera_id):
        """Draws ground truth bounding boxes on the frame using loaded gt_data."""
        if camera_id not in self.gt_data:
            return frame

        camera_gt = self.gt_data[camera_id]
        frame_id_to_check = frame_index + 1 # gt.txt frame IDs are 1-based

        if frame_id_to_check in camera_gt:
            for person_id, x, y, w, h in camera_gt[frame_id_to_check]:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                # Check if box is valid before drawing
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def _ensure_homography(self, frame):
        """Calculates and stores homography if not already done."""
        if self.H is None and frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
            print(f"Calculating homography for frame size: {self.frame_width}x{self.frame_height}")
            try:
                self.H, self.src_points, self.dst_points = compute_homography(
                    self.frame_width, self.frame_height, self.map_width, self.map_height
                )
                # Also store dst_points in the tracker instance if needed there
                self.tracker.dst_points = self.dst_points
            except Exception as e:
                print(f"Error computing homography: {e}")
                self.H = None # Ensure it remains None if calculation fails

    def build_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# MTMMC Person Tracking")

            with gr.Row():
                with gr.Column(scale=1):
                    # ... (Keep UI elements as before, maybe adjust labels slightly)
                    dataset_path = gr.Textbox(label="Dataset Path (Scene Level, e.g., /path/to/train/s01/)", value="/Volumes/HDD/MTMMC/train/train/s01/")
                    mode_dropdown = gr.Dropdown(
                        ["Model Detection", "Ground Truth"], label="Mode", value="Model Detection"
                    )
                    start_btn = gr.Button("Start Tracking / Load Data")
                    pause_checkbox = gr.Checkbox(label="Pause", value=True)
                    frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Frame Position")
                    next_frame_btn = gr.Button("Next Frame")
                    clear_btn = gr.Button("Clear Selection (Model Mode)")
                    refresh_btn = gr.Button("Refresh Display")

                    with gr.Column(visible=False):
                         track_buttons = {i: gr.Button(f"Track {i}", elem_id=f"track_button_{i}")
                                        for i in range(1, 100)}

                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Tracking View"):
                            image_output = gr.Image(label="Combined Camera View", type="numpy")
                        # Updated Tab Label
                        with gr.TabItem("Map View(s)"):
                            map_output = gr.Image(label="Combined Top-Down Map(s)", type="numpy") # Will show combined grid

            gr.Markdown("## Detected People (Model Mode Only)")
            gallery_output = gr.HTML()
            status_output = gr.Textbox(label="Status", interactive=False)

            # --- Event Handlers (Keep as before) ---
            start_btn.click(
                self._on_start,
                inputs=[dataset_path, mode_dropdown],
                outputs=[status_output, frame_slider, pause_checkbox, image_output, map_output, gallery_output]
            )
            mode_dropdown.change(
                self._on_mode_change,
                inputs=[mode_dropdown],
                outputs=[status_output]
            ).then(
                self._on_frame_change,
                inputs=[frame_slider],
                outputs=[image_output, map_output, gallery_output]
            )
            pause_checkbox.change(
                self._toggle_playback,
                inputs=[pause_checkbox],
                outputs=[status_output]
            )
            frame_slider.change(
                self._on_frame_change,
                inputs=[frame_slider],
                outputs=[image_output, map_output, gallery_output]
            )
            next_frame_btn.click(
                self._on_next_frame,
                inputs=[frame_slider],
                outputs=[frame_slider, status_output]
            ).then(
                self._on_frame_change,
                inputs=[frame_slider],
                outputs=[image_output, map_output, gallery_output]
            )
            clear_btn.click(
                self._on_clear_selection,
                outputs=[status_output]
            ).then(
                 self._on_frame_change,
                 inputs=[frame_slider],
                 outputs=[image_output, map_output, gallery_output]
            )
            refresh_btn.click(
                self._on_refresh,
                inputs=[frame_slider],
                outputs=[frame_slider, image_output, map_output, gallery_output]
            )
            for i, btn in track_buttons.items():
                btn.click(
                    self._on_track_person,
                    inputs=gr.Number(value=i, visible=False),
                    outputs=[status_output]
                ).then(
                    self._on_frame_change,
                    inputs=[frame_slider],
                    outputs=[image_output, map_output, gallery_output]
                )

        return demo

    def _on_mode_change(self, mode):
        self.mode = mode
        # Reset states if switching modes might be useful
        self.tracker.selected_track_id = None
        # Optionally clear history when switching? Depends on desired behavior.
        # self.gt_track_history.clear()
        # self.tracker.track_history.clear()
        return f"Mode changed to {mode}. Display will update."

    def _on_start(self, dataset_path, mode):
        self.dataset_path = dataset_path
        self.mode = mode
        self.gt_data = {} # Clear previous GT data
        self.gt_track_history.clear() # Clear GT history on new start
        self.tracker = PersonTracker(self.tracker.model.ckpt_path, map_width=self.map_width, map_height=self.map_height) # Re-initialize tracker

        # Reset homography and frame dimensions
        self.H = None
        self.src_points = None
        self.dst_points = None
        self.frame_width = None
        self.frame_height = None


        if not os.path.isdir(dataset_path):
             return f"Error: Dataset path not found: {dataset_path}", gr.update(), gr.update(value=True), None, None, ""

        try:
            self.camera_dirs = sorted([os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if
                                       os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('c')])
        except Exception as e:
             return f"Error listing camera directories: {e}", gr.update(), gr.update(value=True), None, None, ""


        if not self.camera_dirs:
            return "Error: No camera directories (cXX) found in path", gr.update(), gr.update(value=True), None, None, ""

        # --- Load GT Data ---
        found_gt = False
        for cam_dir in self.camera_dirs:
            cam_id = os.path.basename(cam_dir)
            gt_for_cam = self.load_gt_data_for_camera(cam_dir)
            if gt_for_cam: # Only add if data was actually loaded
                self.gt_data[cam_id] = gt_for_cam
                found_gt = True
        if self.mode == "Ground Truth" and not found_gt:
             print("Warning: Could not load any gt.txt files for Ground Truth mode.")
             # Continue anyway, will just show empty maps/frames

        self.current_frame_index = 0
        self.paused = True # Start paused

        # --- Determine max frames & Calculate Initial Homography ---
        max_frames = 0
        initial_frames = self._load_frames(0) # Load frame 0 to get dimensions
        if initial_frames:
             # Use the first available frame to calculate homography
             first_valid_frame = next((f for f in initial_frames.values() if f is not None), None)
             self._ensure_homography(first_valid_frame)

             # Determine max frames based on first camera dir
             first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
             if os.path.exists(first_cam_rgb_dir):
                 try:
                    frame_files = sorted([f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')])
                    max_frames = len(frame_files)
                 except Exception as e:
                     return f"Error reading frames from {first_cam_rgb_dir}: {e}", gr.update(), gr.update(value=True), None, None, ""
             else:
                 # This case should ideally be caught by _load_frames returning empty/None
                 print(f"Warning: RGB directory not found for first camera: {first_cam_rgb_dir}")
        else:
             return "Error: Could not load initial frames.", gr.update(), gr.update(value=True), None, None, ""


        # --- Process initial frame (frame 0) ---
        # Pass the already loaded initial frames to avoid loading again
        output_image, map_img, gallery_html = self._process_and_get_outputs(0, initial_frames)

        return (
            f"Dataset loaded ({len(self.camera_dirs)} cameras). Mode: {self.mode}. Ready.",
            gr.update(maximum=max_frames - 1 if max_frames > 0 else 0, value=0),
            gr.update(value=self.paused),
            output_image,
            map_img,
            gallery_html
        )

    def _toggle_playback(self, paused):
        self.paused = paused
        return f"Playback {'paused' if paused else 'resumed (manual frame stepping only)'}"

    # Modified to accept pre-loaded frames optionally
    def _process_and_get_outputs(self, frame_index, preloaded_frames=None):
        """Loads frames (if not provided), processes based on mode, calculates maps, and returns display outputs."""
        # Load frames only if not provided (e.g., during _on_start)
        frames = preloaded_frames if preloaded_frames is not None else self._load_frames(frame_index)
        annotated_frames_bgr = {}
        map_img_rgb = None # Final combined map for display
        gallery_html = ""

        if not frames:
             return None, None, "<p>Error loading frames.</p>"

        # Ensure homography is calculated using the first valid frame if needed
        first_valid_frame = next((f for f in frames.values() if f is not None), None)
        self._ensure_homography(first_valid_frame)

        if self.mode == "Model Detection":
            # Process with tracker (returns RGB images)
            annotated_frames_rgb, map_img_model_rgb = self.tracker.process_multiple_frames(frames, self.paused)
            annotated_frames_bgr = {cam_id: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    for cam_id, frame in annotated_frames_rgb.items() if frame is not None}
            map_img_rgb = map_img_model_rgb # Use the tracker's map directly (already RGB)
            gallery_html = create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)

        elif self.mode == "Ground Truth":
            map_images_bgr_per_cam = {} # Store individual BGR maps per camera
            frame_id_to_check = frame_index + 1 # GT uses 1-based index

            for cam_id, frame in frames.items():
                current_cam_boxes_xywh = [] # Format [center_x, center_y, w, h] for map drawing
                current_cam_ids = []

                # Annotate frame with GT boxes
                if frame is not None:
                    annotated_frames_bgr[cam_id] = self.draw_gt_boxes(frame.copy(), frame_index, cam_id)
                else:
                    annotated_frames_bgr[cam_id] = None # Keep None if frame was None

                # Prepare data for this camera's map and update history
                if cam_id in self.gt_data and frame_id_to_check in self.gt_data[cam_id]:
                    for person_id, x, y, w, h in self.gt_data[cam_id][frame_id_to_check]:
                         # Check for valid box dimensions before processing for map
                         if w > 0 and h > 0:
                            center_x = x + w / 2
                            center_y = y + h / 2
                            bottom_y = y + h # Point for perspective transform

                            current_cam_boxes_xywh.append([center_x, center_y, w, h])
                            current_cam_ids.append(person_id)

                            # Update history for this specific camera and person
                            # Use bottom_center (cx, cy+h/2) or (cx, y+h) for map position? Let's use (cx, bottom_y)
                            self.gt_track_history[cam_id][person_id].append((center_x, bottom_y))
                            # Limit history length
                            if len(self.gt_track_history[cam_id][person_id]) > 30:
                                 self.gt_track_history[cam_id][person_id].pop(0)

                # Generate map for this camera if homography is available
                if self.H is not None and self.dst_points is not None:
                    # Filter history to only include IDs currently visible in this camera
                    history_for_this_cam_map = {pid: self.gt_track_history[cam_id][pid]
                                                for pid in current_cam_ids if pid in self.gt_track_history[cam_id]}

                    map_img_cam_bgr = create_map_visualization(
                        self.map_width, self.map_height, self.dst_points,
                        current_cam_boxes_xywh,
                        current_cam_ids,
                        history_for_this_cam_map, # Pass filtered history for this camera
                        self.H,
                        selected_track_id=None # No selection highlight for GT maps yet
                    )
                    map_images_bgr_per_cam[cam_id] = map_img_cam_bgr
                else:
                    # If no homography, create a blank placeholder map
                    map_images_bgr_per_cam[cam_id] = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8) # Light gray
                    cv2.putText(map_images_bgr_per_cam[cam_id], "No Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)


            # Combine the individual camera maps into a grid
            combined_map_bgr = self._combine_frames(map_images_bgr_per_cam) # Reuse combine logic
            # Convert final combined map to RGB for Gradio
            map_img_rgb = cv2.cvtColor(combined_map_bgr, cv2.COLOR_BGR2RGB) if combined_map_bgr is not None else None
            gallery_html = "" # No gallery for GT mode

        else: # Fallback case
            annotated_frames_bgr = frames
            map_img_rgb = None
            gallery_html = ""

        # Combine potentially annotated BGR frames for the main view
        output_image_bgr = self._combine_frames(annotated_frames_bgr)
        # Convert final combined tracking view to RGB for Gradio display
        output_image_rgb = cv2.cvtColor(output_image_bgr, cv2.COLOR_BGR2RGB) if output_image_bgr is not None else None

        return output_image_rgb, map_img_rgb, gallery_html


    def _on_frame_change(self, frame_index):
        if self.dataset_path is None or not self.camera_dirs:
             return None, None, "<p>Dataset not loaded. Click Start.</p>"

        self.current_frame_index = int(frame_index)
        # Pass None for preloaded_frames, so it loads them inside
        output_image, map_img, gallery_html = self._process_and_get_outputs(self.current_frame_index, None)
        return output_image, map_img, gallery_html

    def _on_next_frame(self, current_slider_value):
        if self.dataset_path is None or not self.camera_dirs:
             return current_slider_value, "Dataset not loaded."

        max_frame_index = 0
        first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
        if os.path.exists(first_cam_rgb_dir):
             try:
                frame_files = [f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')]
                max_frame_index = len(frame_files) - 1
             except Exception:
                 return current_slider_value, "Error reading frame count."
        else:
             return current_slider_value, "First camera RGB directory missing."

        next_frame_index = min(int(current_slider_value) + 1, max_frame_index)
        self.current_frame_index = next_frame_index
        return gr.update(value=self.current_frame_index), f"Advanced to frame {self.current_frame_index}"

    def _on_clear_selection(self):
        if self.mode == "Model Detection":
            # Use None track_id in select_person to clear
            status = self.tracker.select_person(None)
            return status
        else:
            return "Clear selection only applicable in Model Detection mode."

    def _on_refresh(self, frame_slider):
        # Re-process the current frame index
        frame_index = int(self.current_frame_index)
         # Pass None for preloaded_frames, so it loads them inside
        output_image, map_img, gallery_html = self._process_and_get_outputs(frame_index, None)
        # Return update for slider value plus the new outputs
        return gr.update(value=frame_index), output_image, map_img, gallery_html


    def _on_track_person(self, track_id):
        if self.mode == "Model Detection":
             status = self.tracker.select_person(int(track_id))
             return status
        else:
             return "Tracking selection only applicable in Model Detection mode."

    def _load_frames(self, frame_index):
        """Loads BGR frames for the given index from all camera directories."""
        frames = {}
        if not self.camera_dirs:
            return frames

        # Estimate max frames (less critical here, mainly for bounds check)
        max_frames_est = 0
        first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
        if os.path.exists(first_cam_rgb_dir):
            try:
                frame_files = [f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')]
                max_frames_est = len(frame_files)
            except Exception:
                 pass # Ignore error here, just an estimate

        # Basic bounds check
        if frame_index < 0: frame_index = 0
        if max_frames_est > 0 and frame_index >= max_frames_est:
             frame_index = max_frames_est -1
             print(f"Adjusted frame index to max available: {frame_index}")

        frame_file_name = f'{frame_index:06d}.jpg'
        h_placeholder = self.frame_height if self.frame_height else 1080
        w_placeholder = self.frame_width if self.frame_width else 1920

        for cam_dir in self.camera_dirs:
            cam_id = os.path.basename(cam_dir)
            image_path = os.path.join(cam_dir, 'rgb', frame_file_name)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    frames[cam_id] = img
                    # Update frame dimensions if not set yet (using first valid loaded frame)
                    if self.frame_height is None or self.frame_width is None:
                        self.frame_height, self.frame_width = img.shape[:2]
                else:
                    print(f"Warning: Failed to read image {image_path}, using black placeholder.")
                    frames[cam_id] = np.zeros((h_placeholder, w_placeholder, 3), dtype=np.uint8)
            else:
                # print(f"Info: Image not found at {image_path} for frame {frame_index}, using black placeholder.")
                frames[cam_id] = np.zeros((h_placeholder, w_placeholder, 3), dtype=np.uint8)
        return frames

    def _combine_frames(self, frames_dict):
        """Combines multiple BGR frames/maps into a grid. Handles None values."""
        if not frames_dict:
            return None

        valid_frames = {k: f for k, f in frames_dict.items() if f is not None}
        if not valid_frames:
             # If all inputs were None, return a small placeholder
             return np.zeros((100, 100, 3), dtype=np.uint8)

        num_items = len(valid_frames)
        frame_list = list(valid_frames.values()) # Use only valid frames for grid calculation

        if num_items == 1:
            return frame_list[0]

        rows = int(np.ceil(np.sqrt(num_items)))
        cols = int(np.ceil(num_items / rows))

        # Get dimensions from the first valid frame/map. Assume consistency.
        height, width, _ = frame_list[0].shape
        combined_image = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

        item_idx = 0
        for i in range(rows):
            for j in range(cols):
                if item_idx < num_items:
                    combined_image[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = frame_list[item_idx]
                    item_idx += 1
                # else: leave remaining grid cells black

        return combined_image


if __name__ == "__main__":
    app = MTMMCTrackerApp(model_path="yolov8n.pt") # Adjust model path if needed
    demo = app.build_ui()
    demo.launch()