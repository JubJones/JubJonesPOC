import os
import cv2
import gradio as gr
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union # Added Union

from simple_poc.tracking.tracker import PersonTracker # Assumes tracker.py is correctly placed
from simple_poc.ui.gallery import create_gallery_html # Assumes gallery.py is correctly placed
from simple_poc.tracking.map import compute_homography, create_map_visualization # Assumes map.py is correctly placed


class MTMMCTrackerApp:

    def __init__(self, model_path="yolov8n.pt", model_type='yolo', map_width=400, map_height=600):
        self.model_path = model_path
        self.model_type = model_type
        self.map_width = map_width
        self.map_height = map_height
        try:
            self.tracker = PersonTracker(model_path, model_type=model_type, map_width=map_width, map_height=map_height)
        except Exception as e:
             print(f"FATAL ERROR during initial Tracker creation: {e}")
             self.tracker = None # Indicate failure

        self.dataset_path = None
        self.camera_dirs = []
        self.camera_ids = []
        self.current_frame_index = 0
        self.paused = True
        self.gt_data = {} # Ground Truth data storage
        self.mode = "Model Detection"

        # GT specific attributes
        self.gt_track_history = defaultdict(lambda: defaultdict(list))
        self.H = None
        self.src_points = None
        self.dst_points = None
        self.frame_width = None
        self.frame_height = None


    def load_gt_data_for_camera(self, camera_dir):
        """Loads ground truth data from gt.txt for a specific camera."""
        gt_path = os.path.join(camera_dir, 'gt', 'gt.txt')
        gt_data_cam = {}
        try:
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue
                    try:
                        frame_id, person_id, x, y, w, h = map(float, parts[:6])
                        frame_id = int(frame_id); person_id = int(person_id)
                        if frame_id not in gt_data_cam: gt_data_cam[frame_id] = []
                        gt_data_cam[frame_id].append((person_id, x, y, w, h))
                    except ValueError: continue # Skip lines with bad numbers
        except FileNotFoundError: return {}
        except Exception as e: print(f"Error reading GT file {gt_path}: {e}"); return {}
        return gt_data_cam


    def draw_gt_boxes(self, frame, frame_index, camera_id):
        """Draws GT boxes on a frame (BGR)."""
        if frame is None: return None
        if camera_id not in self.gt_data: return frame

        camera_gt = self.gt_data[camera_id]
        frame_id_to_check = frame_index + 1 # GT is 1-based

        if frame_id_to_check in camera_gt:
            for person_id, x, y, w, h in camera_gt[frame_id_to_check]:
                if w > 0 and h > 0:
                    x1, y1 = int(x), int(y); x2, y2 = int(x + w), int(y + h)
                    try:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green
                        text_y = y1 - 10 if y1 > 15 else y1 + 15
                        cv2.putText(frame, f"ID: {person_id}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception as e: print(f"Error drawing GT box {person_id} cam {camera_id}: {e}")
        return frame


    def _ensure_homography(self, frame):
        """Calculates homography matrix H if not already done."""
        if self.H is None and frame is not None and frame.size > 0:
            if self.frame_height is None or self.frame_width is None:
                 self.frame_height, self.frame_width = frame.shape[:2]
            if self.frame_height and self.frame_width:
                print(f"Calculating homography ({self.frame_width}x{self.frame_height})")
                try:
                    self.H, self.src_points, self.dst_points = compute_homography(
                        self.frame_width, self.frame_height, self.map_width, self.map_height)
                    if self.H is None: raise ValueError("Homography compute failed")
                    # Share geometry with tracker instance
                    if self.tracker:
                         self.tracker.H = self.H
                         self.tracker.src_points = self.src_points
                         self.tracker.dst_points = self.dst_points
                         self.tracker.frame_width = self.frame_width
                         self.tracker.frame_height = self.frame_height
                    print("Homography calculated.")
                except Exception as e:
                    print(f"Error computing homography: {e}")
                    self.H = None; self.src_points = None; self.dst_points = None
                    if self.tracker: self.tracker.H = None # Clear in tracker too


    def build_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# MTMMC Person Tracking")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**Model:** {self.model_type.upper()} (`{self.model_path}`)")
                    dataset_path = gr.Textbox(label="Dataset Path (Scene Level)", value="/Volumes/HDD/MTMMC/train/train/s01/")
                    mode_dropdown = gr.Dropdown(["Model Detection", "Ground Truth"], label="Mode", value=self.mode)
                    start_btn = gr.Button("Start / Load Data")
                    pause_checkbox = gr.Checkbox(label="Pause", value=self.paused)
                    frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Frame")
                    next_frame_btn = gr.Button("Next Frame")
                    clear_btn = gr.Button("Clear Selection")
                    refresh_btn = gr.Button("Refresh Display")
                    with gr.Column(visible=False): # Hidden buttons for gallery interaction
                        track_buttons = {i: gr.Button(f"Track {i}", elem_id=f"track_button_{i}") for i in range(1, 500)} # Range for potential track IDs
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Tracking View"): image_output = gr.Image(label="Combined Camera View", type="numpy", image_mode="RGB")
                        with gr.TabItem("Map View"): map_output = gr.Image(label="Combined Top-Down Map", type="numpy", image_mode="RGB")
            gr.Markdown("## Detected People (Model Mode Only)")
            gallery_output = gr.HTML()
            status_output = gr.Textbox(label="Status", interactive=False)

            # --- Event Handlers ---
            start_btn.click(self._on_start, inputs=[dataset_path, mode_dropdown], outputs=[status_output, frame_slider, pause_checkbox, image_output, map_output, gallery_output])
            mode_dropdown.change(self._on_mode_change, inputs=[mode_dropdown], outputs=[status_output]).then(self._on_frame_change, inputs=[frame_slider], outputs=[image_output, map_output, gallery_output])
            pause_checkbox.change(self._toggle_playback, inputs=[pause_checkbox], outputs=[status_output])
            frame_slider.release(self._on_frame_change, inputs=[frame_slider], outputs=[image_output, map_output, gallery_output])
            next_frame_btn.click(self._on_next_frame, inputs=[frame_slider], outputs=[frame_slider, status_output]).then(self._on_frame_change, inputs=[frame_slider], outputs=[image_output, map_output, gallery_output])
            clear_btn.click(self._on_clear_selection, outputs=[status_output]).then(self._on_frame_change, inputs=[frame_slider], outputs=[image_output, map_output, gallery_output])
            refresh_btn.click(self._on_refresh, inputs=[frame_slider], outputs=[frame_slider, image_output, map_output, gallery_output])
            for i, btn in track_buttons.items():
                btn.click(self._on_track_person, inputs=gr.Number(value=i, visible=False), outputs=[status_output]).then(self._on_frame_change, inputs=[frame_slider], outputs=[image_output, map_output, gallery_output])
        return demo


    def _on_mode_change(self, mode):
        self.mode = mode
        print(f"Mode changed to: {self.mode}")
        if self.mode != "Model Detection" and self.tracker:
             self.tracker.selected_track_id = None
             self.tracker.person_crops.clear()
             self.tracker.track_history.clear()
             self.tracker.current_boxes.clear()
             self.tracker.current_track_ids.clear()
             self.tracker.current_confidences.clear()
        return f"Mode changed to {mode}. Display updates on frame change/refresh."


    def _on_start(self, dataset_path, mode):
        self.dataset_path = dataset_path
        self.mode = mode
        self.gt_data.clear(); self.gt_track_history.clear()
        self.camera_dirs = []; self.camera_ids = []
        self.H = None; self.src_points = None; self.dst_points = None
        self.frame_width = None; self.frame_height = None
        self.current_frame_index = 0; self.paused = True

        print(f"Re-initializing tracker. Model: {self.model_type}, Path: {self.model_path}")
        try:
            self.tracker = PersonTracker(self.model_path, self.model_type, self.map_width, self.map_height)
        except Exception as e:
             error_msg = f"FATAL ERROR initializing tracker/strategy: {e}."
             print(error_msg)
             self.tracker = None
             return error_msg, gr.update(), gr.update(value=True), None, None, ""

        # --- Validate Dataset Path & Find Cameras ---
        if not dataset_path or not os.path.isdir(dataset_path):
            errmsg = f"Error: Dataset path not found or invalid: '{dataset_path}'"
            return errmsg, gr.update(), gr.update(value=True), None, None, ""
        try:
            all_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            self.camera_dirs = sorted([os.path.join(dataset_path, d) for d in all_dirs if d.startswith('c') and d[1:].isdigit()])
            self.camera_ids = [os.path.basename(cam_dir) for cam_dir in self.camera_dirs]
        except Exception as e:
            errmsg = f"Error listing camera directories in '{dataset_path}': {e}"
            return errmsg, gr.update(), gr.update(value=True), None, None, ""
        if not self.camera_dirs:
            errmsg = f"Error: No camera directories (cXX) found in path: {dataset_path}"
            return errmsg, gr.update(), gr.update(value=True), None, None, ""
        print(f"Found {len(self.camera_dirs)} camera directories: {self.camera_ids}")

        # --- REMOVED strategy initialization call ---
        # The single strategy is created in PersonTracker.__init__ now.

        # --- Load GT Data ---
        found_gt = False; gt_load_errors = []
        for cam_dir in self.camera_dirs:
            cam_id = os.path.basename(cam_dir)
            try:
                gt_for_cam = self.load_gt_data_for_camera(cam_dir)
                if gt_for_cam: self.gt_data[cam_id] = gt_for_cam; found_gt = True
            except Exception as e: gt_load_errors.append(f"GT {cam_id}: {e}")
        status_msg = f"Dataset loaded ({len(self.camera_ids)} cams). Mode: {self.mode}."
        if gt_load_errors: status_msg += f" GT Load Warnings: {'; '.join(gt_load_errors)}"
        if self.mode == "Ground Truth" and not found_gt and not gt_load_errors: status_msg += " Warning: No GT data found."


        # --- Determine Max Frames & Initial Homography ---
        max_frames = 0; initial_frames = self._load_frames(0)
        if not initial_frames:
             return "Error: Could not load initial frames.", gr.update(), gr.update(value=True), None, None, ""

        first_valid_frame = next((initial_frames[cid] for cid in self.camera_ids if initial_frames.get(cid) is not None and initial_frames[cid].size > 0), None)
        if first_valid_frame is None:
             return "Error: All initial frames failed.", gr.update(), gr.update(value=True), None, None, ""

        self._ensure_homography(first_valid_frame) # Calculate H based on first frame

        # Get max frames from first camera directory
        if self.camera_dirs:
            first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
            if os.path.isdir(first_cam_rgb_dir):
                try: max_frames = len([f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')])
                except Exception as e: status_msg += f" Warn: Max frames read error ({e})."; max_frames = 100
            else: status_msg += " Warn: RGB dir missing."; max_frames = 100
        else: max_frames = 100

        # Process frame 0 for initial display
        output_image, map_img, gallery_html = self._process_and_get_outputs(0, initial_frames)

        return (status_msg + " Ready.",
                gr.update(maximum=max_frames - 1 if max_frames > 0 else 0, value=0),
                gr.update(value=self.paused),
                output_image, map_img, gallery_html)


    def _toggle_playback(self, paused):
        self.paused = paused
        return f"Playback {'paused' if paused else 'resumed (manual stepping only)'}"


    def _process_and_get_outputs(self, frame_index: int, preloaded_frames: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Loads frames, processes based on mode, returns display outputs (RGB images, HTML gallery)."""
        frames_bgr = preloaded_frames if preloaded_frames is not None else self._load_frames(frame_index)
        output_image_rgb: Optional[np.ndarray] = None
        map_img_rgb: Optional[np.ndarray] = None
        gallery_html: str = ""

        if not frames_bgr:
            # Return blank placeholders on load error
            blank_h=self.frame_height or 480; blank_w=self.frame_width or 640
            placeholder=np.zeros((blank_h, blank_w, 3), dtype=np.uint8); cv2.putText(placeholder, f"Error Loading Frame {frame_index}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            map_placeholder_rgb = cv2.cvtColor(np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            return placeholder_rgb, map_placeholder_rgb, "<p>Error loading frames.</p>"

        # Ensure homography exists
        first_valid_frame = next((f for f in frames_bgr.values() if f is not None and f.size > 0), None)
        if first_valid_frame is not None: self._ensure_homography(first_valid_frame)

        # --- Processing ---
        if self.mode == "Model Detection":
            if not self.tracker: # Check if tracker failed initialization
                 gallery_html = "<p>Tracker Error.</p>"
                 # Combine raw frames for display
                 raw_frames_rgb = {cid: cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for cid, f in frames_bgr.items() if f is not None}
                 output_image_rgb = self._combine_frames(raw_frames_rgb)
                 map_placeholder_rgb = cv2.cvtColor(np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8), cv2.COLOR_BGR2RGB)
                 map_img_rgb = map_placeholder_rgb
            else:
                try:
                    # process_multiple_frames uses the single shared strategy now
                    annotated_frames_rgb, map_img_model_rgb = self.tracker.process_multiple_frames(frames_bgr, self.paused)
                    output_image_rgb = self._combine_frames(annotated_frames_rgb) # Combine annotated views
                    map_img_rgb = map_img_model_rgb # Use the combined map returned
                    gallery_html = create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)
                except Exception as e:
                    print(f"Error during Model Detection processing step: {e}")
                    # Fallback display on processing error
                    raw_frames_rgb = {cid: cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for cid, f in frames_bgr.items() if f is not None}
                    output_image_rgb = self._combine_frames(raw_frames_rgb)
                    map_placeholder_rgb = cv2.cvtColor(np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8), cv2.COLOR_BGR2RGB)
                    map_img_rgb = map_placeholder_rgb
                    gallery_html = f"<p>Processing Error: {e}</p>"

        elif self.mode == "Ground Truth":
            annotated_frames_bgr = {}
            all_gt_boxes_for_map = []; all_gt_ids_for_map = []
            current_gt_history_for_map = defaultdict(list)
            frame_id_to_check = frame_index + 1

            for cam_id, frame_bgr in frames_bgr.items():
                annotated_bgr = self.draw_gt_boxes(frame_bgr.copy() if frame_bgr is not None else None, frame_index, cam_id)
                annotated_frames_bgr[cam_id] = annotated_bgr
                # Collect data for combined GT map
                if cam_id in self.gt_data and frame_id_to_check in self.gt_data[cam_id]:
                    for person_id, x, y, w, h in self.gt_data[cam_id][frame_id_to_check]:
                        if w > 0 and h > 0:
                            cx, cy, by = x + w / 2, y + h / 2, y + h
                            all_gt_boxes_for_map.append([cx, cy, w, h])
                            all_gt_ids_for_map.append(person_id)
                            current_gt_history_for_map[person_id].append((cx, by))
                            if len(current_gt_history_for_map[person_id]) > 30: current_gt_history_for_map[person_id] = current_gt_history_for_map[person_id][-30:]

            # Create combined GT map
            if self.H is not None and self.dst_points is not None:
                map_img_gt_bgr = create_map_visualization(self.map_width, self.map_height, self.dst_points, all_gt_boxes_for_map, all_gt_ids_for_map, current_gt_history_for_map, self.H, None)
                map_img_rgb = cv2.cvtColor(map_img_gt_bgr, cv2.COLOR_BGR2RGB) if map_img_gt_bgr is not None else None
            else:
                placeholder_map = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8)
                cv2.putText(placeholder_map, "No Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                map_img_rgb = cv2.cvtColor(placeholder_map, cv2.COLOR_BGR2RGB)

            output_image_rgb = self._combine_frames(annotated_frames_bgr) # Combine annotated GT views
            if output_image_rgb is not None: output_image_rgb = cv2.cvtColor(output_image_rgb, cv2.COLOR_BGR2RGB) # Ensure final is RGB
            gallery_html = "" # No gallery in GT mode

        else: # Unknown mode
             raw_frames_rgb = {cid: cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for cid, f in frames_bgr.items() if f is not None}
             output_image_rgb = self._combine_frames(raw_frames_rgb)
             map_placeholder_rgb = cv2.cvtColor(np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8), cv2.COLOR_BGR2RGB)
             map_img_rgb = map_placeholder_rgb
             gallery_html = ""

        return output_image_rgb, map_img_rgb, gallery_html


    def _on_frame_change(self, frame_index):
        if self.dataset_path is None or not self.camera_dirs: return None, None, "<p>Dataset not loaded.</p>"
        try: self.current_frame_index = int(frame_index)
        except (ValueError, TypeError): self.current_frame_index = 0
        return self._process_and_get_outputs(self.current_frame_index, None)


    def _on_next_frame(self, current_slider_value):
        if self.dataset_path is None or not self.camera_dirs: return current_slider_value, "Dataset not loaded."
        max_frame_index = 0
        if self.camera_dirs:
            first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
            if os.path.isdir(first_cam_rgb_dir):
                try: max_frame_index = len([f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')]) - 1
                except Exception: pass # Keep max_frame_index 0
        try: current_idx = int(current_slider_value)
        except (ValueError, TypeError): current_idx = 0
        next_frame_index = current_idx + 1
        if max_frame_index > 0: next_frame_index = min(next_frame_index, max_frame_index)
        self.current_frame_index = next_frame_index
        status = f"Advanced to frame {self.current_frame_index}" + (" (End Reached)" if max_frame_index > 0 and next_frame_index == max_frame_index else "")
        return gr.update(value=self.current_frame_index), status


    def _on_clear_selection(self):
        if self.mode == "Model Detection" and self.tracker: return self.tracker.select_person(None)
        else: return "Clear selection only applicable in Model Detection mode (or tracker error)."


    def _on_refresh(self, frame_slider_value):
        print("Refreshing display...")
        try: frame_index = int(frame_slider_value)
        except (ValueError, TypeError): frame_index = 0
        self.current_frame_index = frame_index
        output_image_rgb, map_img_rgb, gallery_html = self._process_and_get_outputs(frame_index, None)
        return gr.update(value=frame_index), output_image_rgb, map_img_rgb, gallery_html


    def _on_track_person(self, track_id):
        """Callback for gallery button click -> calls tracker.select_person."""
        if self.mode == "Model Detection" and self.tracker:
             try:
                 # Gradio sends value from gr.Number() which should be float/int
                 track_id_int = int(track_id)
                 return self.tracker.select_person(track_id_int) # select_person handles logic
             except (ValueError, TypeError): return f"Error: Invalid track ID {track_id}"
             except Exception as e: return f"Error selecting person: {e}"
        else: return "Tracking selection only available in Model Detection mode (or tracker error)."


    def _load_frames(self, frame_index: int) -> Dict[str, Optional[np.ndarray]]:
        """Loads BGR frames for the given index from all camera directories."""
        frames_bgr = {}
        if not self.camera_dirs: return frames_bgr
        max_frames = 0
        if self.camera_dirs: # Get max frames estimate
            try: max_frames = len([f for f in os.listdir(os.path.join(self.camera_dirs[0], 'rgb')) if f.lower().endswith('.jpg')])
            except Exception: max_frames = 0
        # Bounds check frame_index
        if frame_index < 0: frame_index = 0
        if max_frames > 0 and frame_index >= max_frames: frame_index = max_frames - 1
        frame_file_name = f'{frame_index + 1:06d}.jpg' # Assumes 1-based naming
        h_placeholder = self.frame_height or 1080; w_placeholder = self.frame_width or 1920

        for cam_dir in self.camera_dirs:
            cam_id = os.path.basename(cam_dir)
            image_path = os.path.join(cam_dir, 'rgb', frame_file_name)
            img = None
            try:
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is not None and (self.frame_height is None or self.frame_width is None):
                        self.frame_height, self.frame_width = img.shape[:2]; h_placeholder, w_placeholder = self.frame_height, self.frame_width
            except Exception as e: print(f"Error reading image {image_path}: {e}"); img = None

            if img is not None: frames_bgr[cam_id] = img
            else: # Create placeholder
                 placeholder_img = np.zeros((h_placeholder, w_placeholder, 3), dtype=np.uint8)
                 cv2.putText(placeholder_img, f"No Image ({cam_id} F:{frame_index+1})", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                 frames_bgr[cam_id] = placeholder_img
        return frames_bgr


    def _combine_frames(self, frames_dict: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Combines multiple frames/maps into a grid, resizing if necessary."""
        if not frames_dict: return None
        valid_frames_data={cid:f for cid,f in frames_dict.items() if f is not None and f.size>0}
        sorted_cam_ids = sorted(valid_frames_data.keys())
        valid_frames = [valid_frames_data[cid] for cid in sorted_cam_ids]
        if not valid_frames: return np.zeros((100, 100, 3), dtype=np.uint8)
        num_items = len(valid_frames)
        if num_items == 1: return valid_frames[0].copy()
        rows = int(np.ceil(np.sqrt(num_items))); cols = int(np.ceil(num_items / rows))
        try: target_h, target_w, chans = valid_frames[0].shape; dtype = valid_frames[0].dtype
        except Exception: return np.zeros((100, 100, 3), dtype=np.uint8)
        combined_image = np.zeros((rows * target_h, cols * target_w, chans), dtype=dtype)
        frame_idx = 0
        for i in range(rows):
            for j in range(cols):
                if frame_idx < num_items:
                    current_frame = valid_frames[frame_idx]
                    try:
                        if current_frame.shape != (target_h, target_w, chans):
                             current_frame = cv2.resize(current_frame, (target_w, target_h))
                        combined_image[i*target_h:(i+1)*target_h, j*target_w:(j+1)*target_w, :] = current_frame
                    except Exception as e: print(f"Error placing frame {frame_idx} (Cam:{sorted_cam_ids[frame_idx]}) in grid: {e}")
                    frame_idx += 1
        return combined_image
