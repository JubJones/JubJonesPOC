import os
import cv2
import gradio as gr
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple # Added Optional, Tuple

# Make sure PersonTracker is imported correctly
from simple_poc.tracking.tracker import PersonTracker
from simple_poc.ui.gallery import create_gallery_html
from simple_poc.tracking.map import compute_homography, create_map_visualization


class MTMMCTrackerApp:
    # Added model_type and adjusted default model_path
    def __init__(self, model_path="yolov8n.pt", model_type='yolo', map_width=400, map_height=600):
        self.model_path = model_path
        self.model_type = model_type # Store model type
        self.map_width = map_width
        self.map_height = map_height
        # Pass model_type to PersonTracker, but strategies are not created yet
        self.tracker = PersonTracker(model_path, model_type=model_type, map_width=map_width, map_height=map_height)
        self.dataset_path = None
        self.camera_dirs = []
        self.camera_ids = [] # Store camera IDs
        self.current_frame_index = 0
        self.paused = True
        self.gt_data = {} # Key: camera_id, Value: dict {frame_id: [(person_id, x, y, w, h), ...]}
        self.mode = "Model Detection" # Default mode

        # --- Attributes for GT map (remain mostly the same) ---
        self.gt_track_history = defaultdict(lambda: defaultdict(list)) # Key: camera_id, Key: person_id, Value: list[(cx, cy_bottom)]
        self.H = None # Homography matrix (shared between GT and Model mode)
        self.src_points = None # Source points for homography
        self.dst_points = None # Destination points for homography
        self.frame_width = None # Store frame dimensions
        self.frame_height = None # Store frame dimensions
        # --- End GT attributes ---

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
                    try:
                        frame_id, person_id, x, y, w, h = map(float, parts[:6])
                        frame_id = int(frame_id)
                        person_id = int(person_id)
                        if frame_id not in gt_data_cam:
                            gt_data_cam[frame_id] = []
                        # Store as (person_id, x, y, w, h) - using original top-left coords
                        gt_data_cam[frame_id].append((person_id, x, y, w, h))
                    except ValueError:
                         print(f"Warning: Skipping line with invalid number format in {gt_path}: {line.strip()}")
                         continue
        except FileNotFoundError:
            # print(f"Info: Ground truth file not found at {gt_path}. No GT boxes or map points for this camera.")
            return {} # Return empty dict, not None
        except Exception as e: # Catch other potential errors like permissions
            print(f"Error reading GT file {gt_path}: {e}")
            return {}
        return gt_data_cam


    def draw_gt_boxes(self, frame, frame_index, camera_id):
        """Draws ground truth bounding boxes on the frame using loaded gt_data."""
        if frame is None: # Check if frame is valid
             print(f"Warning: Cannot draw GT boxes on None frame for camera {camera_id}")
             return None

        if camera_id not in self.gt_data:
            return frame # Return frame unmodified if no GT data for this camera

        camera_gt = self.gt_data[camera_id]
        frame_id_to_check = frame_index + 1 # gt.txt frame IDs are 1-based

        if frame_id_to_check in camera_gt:
            for person_id, x, y, w, h in camera_gt[frame_id_to_check]:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                # Check if box is valid before drawing
                if w > 0 and h > 0: # Check width and height directly
                    try:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green GT boxes
                        # Adjust text position if near top edge
                        text_y = y1 - 10 if y1 > 15 else y1 + 15
                        cv2.putText(frame, f"ID: {person_id}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception as e:
                         print(f"Error drawing GT box ID {person_id} on frame {frame_index} for camera {camera_id}: {e}")
        return frame

    def _ensure_homography(self, frame):
        """Calculates and stores homography if not already done, using the first valid frame."""
        # Only calculate if H is None and we have a valid frame
        if self.H is None and frame is not None and frame.size > 0:
            # Check if frame dimensions are already set, otherwise get them
            if self.frame_height is None or self.frame_width is None:
                 self.frame_height, self.frame_width = frame.shape[:2]

            # Ensure dimensions are valid before proceeding
            if self.frame_height and self.frame_width:
                print(f"Calculating homography for frame size: {self.frame_width}x{self.frame_height}")
                try:
                    self.H, self.src_points, self.dst_points = compute_homography(
                        self.frame_width, self.frame_height, self.map_width, self.map_height
                    )
                    # Also store essential geometry in the tracker instance
                    # This is needed if process_frame itself calculates maps, and for consistency
                    self.tracker.H = self.H
                    self.tracker.src_points = self.src_points
                    self.tracker.dst_points = self.dst_points
                    self.tracker.frame_width = self.frame_width
                    self.tracker.frame_height = self.frame_height
                    print("Homography calculated and shared with tracker.")

                    # Defensive check if compute_homography somehow returned None without exception
                    if self.H is None:
                        print("Error: Homography computation returned None unexpectedly.")
                        self.dst_points = None # Ensure dependent attributes are also reset
                        self.src_points = None
                        # Don't reset tracker's H here, let it remain None

                except Exception as e:
                    print(f"Error computing homography: {e}")
                    self.H = None # Ensure it remains None if calculation fails
                    self.src_points = None
                    self.dst_points = None
                    # Optionally clear from tracker too, though it should check for None anyway
                    self.tracker.H = None
                    self.tracker.src_points = None
                    self.tracker.dst_points = None

            else:
                 print("Warning: Cannot calculate homography, frame dimensions are invalid.")


    def build_ui(self):
        # --- UI Definition (Mostly unchanged, ensure outputs match return values) ---
        with gr.Blocks() as demo:
            gr.Markdown("# MTMMC Person Tracking")

            with gr.Row():
                with gr.Column(scale=1):
                    # Rely on initial app creation `model_type` for now.
                    gr.Markdown(f"**Model:** {self.model_type.upper()} (`{self.model_path}`)")

                    dataset_path = gr.Textbox(label="Dataset Path (Scene Level, e.g., /path/to/train/s01/)", value="/Volumes/HDD/MTMMC/train/train/s01/")
                    mode_dropdown = gr.Dropdown(
                        ["Model Detection", "Ground Truth"], label="Mode", value=self.mode
                    )
                    start_btn = gr.Button("Start Tracking / Load Data")
                    pause_checkbox = gr.Checkbox(label="Pause", value=self.paused)
                    frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Frame Position")
                    next_frame_btn = gr.Button("Next Frame")
                    clear_btn = gr.Button("Clear Selection (Model Mode)")
                    refresh_btn = gr.Button("Refresh Display")

                    # Hidden buttons for gallery interaction (important for linking clicks)
                    with gr.Column(visible=False) as hidden_buttons_col:
                        # Create enough potential buttons, they are hidden anyway
                        # Max track ID can be large, adjust range if needed
                        track_buttons = {i: gr.Button(f"Track {i}", elem_id=f"track_button_{i}")
                                         for i in range(1, 500)} # Increased range substantially

                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Tracking View"):
                            # Expects RGB numpy array
                            image_output = gr.Image(label="Combined Camera View", type="numpy", image_mode="RGB")
                        with gr.TabItem("Map View(s)"):
                             # Expects RGB numpy array (now shows combined map)
                            map_output = gr.Image(label="Combined Top-Down Map", type="numpy", image_mode="RGB")

            gr.Markdown("## Detected People (Model Mode Only)")
            gallery_output = gr.HTML() # Gallery uses base64 encoded RGB images
            status_output = gr.Textbox(label="Status", interactive=False)

            # --- Event Handlers ---
            # Store necessary initial config to pass to _on_start
            # No longer needed as self holds the config, tracker re-created in _on_start
            # initial_state = gr.State(...) # REMOVED

            start_btn.click(
                self._on_start,
                # Pass UI elements needed for initialization
                inputs=[dataset_path, mode_dropdown],
                outputs=[status_output, frame_slider, pause_checkbox, image_output, map_output, gallery_output]
            )
            mode_dropdown.change(
                self._on_mode_change,
                inputs=[mode_dropdown],
                outputs=[status_output]
                # Chain the update to redraw the frame in the new mode
            ).then(
                self._on_frame_change,
                inputs=[frame_slider], # Trigger based on current slider value
                outputs=[image_output, map_output, gallery_output]
            )
            pause_checkbox.change(
                self._toggle_playback,
                inputs=[pause_checkbox],
                outputs=[status_output]
            )
            # Use `release` for slider for better performance than `change`
            frame_slider.release(
                self._on_frame_change,
                inputs=[frame_slider],
                outputs=[image_output, map_output, gallery_output]
            )
            next_frame_btn.click(
                self._on_next_frame,
                inputs=[frame_slider],
                outputs=[frame_slider, status_output] # Update slider value and status
                # Chain the update to redraw the new frame
            ).then(
                 self._on_frame_change,
                 inputs=[frame_slider], # Trigger based on the *updated* slider value
                 outputs=[image_output, map_output, gallery_output]
            )
            clear_btn.click(
                self._on_clear_selection,
                outputs=[status_output]
                # Chain the update to redraw frame with cleared selection
            ).then(
                 self._on_frame_change,
                 inputs=[frame_slider],
                 outputs=[image_output, map_output, gallery_output]
            )
            refresh_btn.click(
                self._on_refresh,
                inputs=[frame_slider], # Pass current frame index
                # Outputs: update slider (redundant?), images, gallery
                outputs=[frame_slider, image_output, map_output, gallery_output]
            )

            # Link gallery buttons (JS onclick) to hidden Gradio buttons
            for i, btn in track_buttons.items():
                # Pass track_id as a number input
                btn.click(
                    self._on_track_person,
                    # Use gr.Number for type hint, value is the track ID 'i'
                    # Make number input invisible
                    inputs=gr.Number(value=i, visible=False),
                    outputs=[status_output] # Update status on click
                    # Chain the update to redraw with the new selection highlight
                ).then(
                    self._on_frame_change,
                    inputs=[frame_slider], # Redraw based on current slider
                    outputs=[image_output, map_output, gallery_output]
                )

        return demo

    def _on_mode_change(self, mode):
        self.mode = mode
        print(f"Mode changed to: {self.mode}")
        # Reset states that are mode-specific
        if self.mode != "Model Detection":
             # Clear model-specific state if switching away
             self.tracker.selected_track_id = None
             self.tracker.person_crops.clear() # Clear gallery crops
             self.tracker.track_history.clear() # Clear model track history
             # Potentially clear current boxes/ids too?
             self.tracker.current_boxes.clear()
             self.tracker.current_track_ids.clear()
             self.tracker.current_confidences.clear()
        # else: # Switching to Model Detection
             # Optionally clear GT history? Let's keep it for now.
             # self.gt_track_history.clear()
             # Important: Ensure tracker state is clean if switching back
             # This is handled by the re-initialization in _on_start mostly,
             # but clearing history here helps if _on_start isn't called again.
             # self.tracker.track_history.clear() # Already cleared above if switching away


        return f"Mode changed to {mode}. Display will update on next frame change/refresh."

    # Modified _on_start to use self attributes for tracker config
    def _on_start(self, dataset_path, mode):
        self.dataset_path = dataset_path
        self.mode = mode
        self.gt_data = {} # Clear previous GT data
        self.gt_track_history.clear() # Clear GT history on new start
        self.camera_dirs = [] # Clear previous camera list
        self.camera_ids = [] # Clear previous camera IDs

        # --- Re-initialize Tracker ---
        # Use attributes stored in self (model_path, model_type, etc.)
        print(f"Re-initializing tracker. Model: {self.model_type}, Path: {self.model_path}")
        try:
            # Create a fresh tracker instance
            self.tracker = PersonTracker(
                model_path=self.model_path,
                model_type=self.model_type,
                map_width=self.map_width,
                map_height=self.map_height
            )
            # Strategies will be initialized after finding cameras
        except Exception as e:
             # If tracker base init fails (unlikely now, but possible), report error
             error_msg = f"FATAL ERROR initializing tracker base class: {e}."
             print(error_msg)
             return (
                 error_msg,
                 gr.update(), # No change to slider max
                 gr.update(value=True), # Keep paused
                 None, None, "" # No image, map, gallery
             )

        # Reset homography and frame dimensions (will be recalculated)
        self.H = None
        self.src_points = None
        self.dst_points = None
        self.frame_width = None
        self.frame_height = None
        # Tracker's internal geometry state is reset by creating a new instance above

        # --- Validate Dataset Path ---
        if not dataset_path or not os.path.isdir(dataset_path):
            errmsg = f"Error: Dataset path not found or invalid: '{dataset_path}'"
            return errmsg, gr.update(), gr.update(value=True), None, None, ""

        try:
            # Find camera directories (c01, c02, etc.)
            all_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            # Filter more strictly for c<number> pattern
            cam_dirs_found = sorted([
                os.path.join(dataset_path, d) for d in all_dirs
                if d.startswith('c') and d[1:].isdigit()
            ])

            # # Removed the deeper path check - assume path is correct scene level for simplicity.
            # # Add back if needed:
            # if not cam_dirs_found:
            #      # Check one level deeper logic...

            self.camera_dirs = cam_dirs_found
            self.camera_ids = [os.path.basename(cam_dir) for cam_dir in self.camera_dirs] # Store IDs

        except Exception as e:
            errmsg = f"Error listing camera directories in '{dataset_path}': {e}"
            return errmsg, gr.update(), gr.update(value=True), None, None, ""

        if not self.camera_dirs:
            errmsg = f"Error: No camera directories (cXX) found in path: {dataset_path}"
            return errmsg, gr.update(), gr.update(value=True), None, None, ""

        print(f"Found {len(self.camera_dirs)} camera directories: {self.camera_ids}")

        # --- Initialize Tracker Strategies (Crucial Step) ---
        if self.mode == "Model Detection": # Only init strategies if in model mode
            if not self.camera_ids:
                 errmsg = "Error: Cannot initialize model strategies, no camera IDs found."
                 return errmsg, gr.update(), gr.update(value=True), None, None, ""
            try:
                 # Call the new method to create a strategy instance per camera
                 self.tracker.initialize_strategies(self.camera_ids)
            except Exception as e:
                 error_msg = f"FATAL ERROR initializing camera strategies: {e}. Check model path/type and dependencies."
                 print(error_msg)
                 return (
                     error_msg, gr.update(), gr.update(value=True), None, None, ""
                 )

        # --- Load GT Data (if exists) ---
        found_gt = False
        gt_load_errors = []
        for cam_dir in self.camera_dirs:
            cam_id = os.path.basename(cam_dir)
            try:
                gt_for_cam = self.load_gt_data_for_camera(cam_dir)
                if gt_for_cam: # Only add if data was actually loaded
                    self.gt_data[cam_id] = gt_for_cam
                    found_gt = True
            except Exception as e:
                gt_load_errors.append(f"Error loading GT for {cam_id}: {e}")

        status_msg = f"Dataset loaded ({len(self.camera_dirs)} cameras). Mode: {self.mode}."
        if gt_load_errors:
             status_msg += f" GT Load Warnings: {'; '.join(gt_load_errors)}"
        if self.mode == "Ground Truth" and not found_gt and not gt_load_errors:
             status_msg += " Warning: No gt.txt files found for Ground Truth mode."
             print("Warning: Could not load any gt.txt files for Ground Truth mode.")


        # --- Determine max frames & Calculate Initial Homography ---
        self.current_frame_index = 0
        self.paused = True # Start paused
        max_frames = 0

        # Load frame 0 to get dimensions and calculate homography
        initial_frames = self._load_frames(0)
        if not initial_frames:
             errmsg = "Error: Could not load initial frames (frame 0) from camera directories."
             return errmsg, gr.update(), gr.update(value=True), None, None, ""

        # Use the first available frame to calculate homography
        # Iterate through camera IDs in sorted order for consistency
        first_valid_frame = None
        for cam_id in self.camera_ids: # Use the stored, sorted camera IDs
            frame = initial_frames.get(cam_id)
            if frame is not None and frame.size > 0:
                 first_valid_frame = frame
                 break # Found a valid frame

        if first_valid_frame is None:
             errmsg = "Error: All initial frames (frame 0) failed to load or were empty. Cannot determine dimensions or calculate homography."
             return errmsg, gr.update(), gr.update(value=True), None, None, ""

        # Ensure homography is calculated *before* processing the first frame display
        # This also sets self.frame_width/height if needed
        self._ensure_homography(first_valid_frame)

        # Determine max frames based on first camera dir (more reliable way)
        if self.camera_dirs: # Check if list is not empty
            first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
            if os.path.isdir(first_cam_rgb_dir):
                try:
                    # Filter for jpg files and sort numerically if possible (though simple count is enough)
                    frame_files = sorted([f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')])
                    max_frames = len(frame_files)
                except Exception as e:
                    errmsg = f"Error reading frames from {first_cam_rgb_dir}: {e}"
                    status_msg += f" Warning: Could not determine max frames ({e})."
                    max_frames = 100 # Default fallback max
            else:
                 # errmsg = f"Warning: RGB directory not found for first camera: {first_cam_rgb_dir}. Cannot determine frame count."
                 status_msg += f" Warning: RGB dir missing for first camera ({first_cam_rgb_dir}). Assuming 100 frames."
                 max_frames = 100 # Default fallback
        else:
             status_msg += " Warning: No camera directories found to determine max frames. Assuming 100."
             max_frames = 100 # Default if no cameras were found earlier (shouldn't happen here)


        # --- Process initial frame (frame 0) for initial display ---
        # Pass the already loaded initial frames to avoid loading again
        # This call uses the homography calculated above
        # It also now uses the per-camera strategies if in Model Mode
        output_image, map_img, gallery_html = self._process_and_get_outputs(0, initial_frames)


        return (
            status_msg + " Ready.",
            gr.update(maximum=max_frames - 1 if max_frames > 0 else 0, value=0), # Update slider range and value
            gr.update(value=self.paused), # Update pause state
            output_image,
            map_img,
            gallery_html
        )

    def _toggle_playback(self, paused):
        self.paused = paused
        # Note: Actual playback thread removed for simplicity, only manual stepping now.
        return f"Playback {'paused' if paused else 'resumed (manual frame stepping only)'}"

    def _process_and_get_outputs(self, frame_index: int, preloaded_frames: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Loads frames (if not provided), processes based on mode (using per-camera strategies if applicable),
        calculates maps, and returns display outputs (RGB images for Gradio, HTML gallery).
        """
        # Load frames only if not provided (e.g., during slider change)
        # Frames are loaded as BGR by default
        frames_bgr = preloaded_frames if preloaded_frames is not None else self._load_frames(frame_index)

        output_image_rgb: Optional[np.ndarray] = None
        map_img_rgb: Optional[np.ndarray] = None
        gallery_html: str = ""

        if not frames_bgr:
            print(f"Error: Failed to load any frames for index {frame_index}.")
            # Return blank placeholders
            blank_h = self.frame_height or 480
            blank_w = self.frame_width or 640
            placeholder = np.zeros((blank_h, blank_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Error Loading Frame {frame_index}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Red text
            placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            map_placeholder_rgb = cv2.cvtColor(np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8), cv2.COLOR_BGR2RGB) # Grey map
            return placeholder_rgb, map_placeholder_rgb, "<p>Error loading frames.</p>"

        # Ensure homography is calculated using the first valid frame if needed
        first_valid_frame = next((f for f in frames_bgr.values() if f is not None and f.size > 0), None)
        if first_valid_frame is not None:
             self._ensure_homography(first_valid_frame) # Calculate if not already done

        # --- Processing based on mode ---
        if self.mode == "Model Detection":
            # Process with tracker (expects BGR frames, uses per-camera strategies)
            # process_multiple_frames returns RGB annotated frames dict and combined RGB map
            try:
                # Check if strategies were initialized
                if not self.tracker.strategies_per_camera and self.camera_ids:
                     # Attempt to initialize strategies if they are missing (e.g., if mode changed after start)
                     print("Warning: Strategies not initialized. Attempting initialization now...")
                     try:
                         self.tracker.initialize_strategies(self.camera_ids)
                     except Exception as init_e:
                         raise RuntimeError("Failed to initialize strategies on demand") from init_e # Re-raise as runtime error


                annotated_frames_rgb, map_img_model_rgb = self.tracker.process_multiple_frames(frames_bgr, self.paused)
                # Combine the RGB annotated frames from the tracker into a grid
                output_image_rgb = self._combine_frames(annotated_frames_rgb) # Expects RGB, returns RGB
                map_img_rgb = map_img_model_rgb # Already combined RGB map
                # Generate gallery HTML (expects RGB crops from tracker)
                # Needs to handle per-camera crops dict now
                gallery_html = create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)
            except Exception as e:
                 print(f"Error during Model Detection processing for frame {frame_index}: {e}")
                 # Fallback to showing raw frames on error
                 raw_frames_rgb = {cam_id: cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for cam_id, f in frames_bgr.items() if f is not None}
                 output_image_rgb = self._combine_frames(raw_frames_rgb)
                 map_placeholder_rgb = cv2.cvtColor(np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8), cv2.COLOR_BGR2RGB) # Grey map
                 map_img_rgb = map_placeholder_rgb
                 gallery_html = f"<p>Error during detection: {e}</p>"


        elif self.mode == "Ground Truth":
            # Process GT data (doesn't involve the model strategies)
            map_images_bgr_per_cam = {} # Store individual BGR maps per camera (if needed, not currently used)
            annotated_frames_bgr = {} # Store annotated BGR frames
            frame_id_to_check = frame_index + 1 # GT uses 1-based index

            all_gt_boxes_for_map = [] # Collect boxes from all cameras for combined map
            all_gt_ids_for_map = [] # Collect IDs from all cameras for combined map
            current_gt_history_for_map = defaultdict(list) # Use simple list for combined map history points

            for cam_id, frame_bgr in frames_bgr.items():
                # Annotate frame with GT boxes (expects BGR, returns annotated BGR)
                annotated_bgr = self.draw_gt_boxes(frame_bgr.copy() if frame_bgr is not None else None, frame_index, cam_id)
                annotated_frames_bgr[cam_id] = annotated_bgr # Store annotated BGR

                # Prepare data for the combined map
                if cam_id in self.gt_data and frame_id_to_check in self.gt_data[cam_id]:
                    for person_id, x, y, w, h in self.gt_data[cam_id][frame_id_to_check]:
                        if w > 0 and h > 0:
                            center_x = x + w / 2
                            center_y = y + h / 2
                            bottom_y = y + h # Point for perspective transform

                            # Store in format expected by create_map_visualization
                            all_gt_boxes_for_map.append([center_x, center_y, w, h])
                            all_gt_ids_for_map.append(person_id)

                            # Update history (simple global list for GT points)
                            # Note: GT IDs *should* be unique across cameras in most datasets, unlike model IDs
                            current_gt_history_for_map[person_id].append((center_x, bottom_y))
                            # Limit history length if desired (e.g., keep last 30 points)
                            if len(current_gt_history_for_map[person_id]) > 30:
                                 current_gt_history_for_map[person_id] = current_gt_history_for_map[person_id][-30:]


            # Create the combined GT map if homography is available
            if self.H is not None and self.dst_points is not None:
                # create_map_visualization expects BGR map output
                map_img_gt_bgr = create_map_visualization(
                    self.map_width, self.map_height, self.dst_points,
                    all_gt_boxes_for_map,   # Use combined GT boxes
                    all_gt_ids_for_map,     # Use combined GT IDs
                    current_gt_history_for_map, # Use the GT history built for this frame
                    self.H,
                    selected_track_id=None # No selection highlight for GT maps
                )
                map_img_rgb = cv2.cvtColor(map_img_gt_bgr, cv2.COLOR_BGR2RGB) if map_img_gt_bgr is not None else None
            else:
                # If no homography, create a blank placeholder map (BGR then convert)
                placeholder_map = np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8) # Light gray BGR
                cv2.putText(placeholder_map, "No Homography", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                map_img_rgb = cv2.cvtColor(placeholder_map, cv2.COLOR_BGR2RGB)


            # Combine annotated BGR frames into a grid
            combined_annotated_bgr = self._combine_frames(annotated_frames_bgr)
            # Convert final combined tracking view to RGB
            output_image_rgb = cv2.cvtColor(combined_annotated_bgr, cv2.COLOR_BGR2RGB) if combined_annotated_bgr is not None else None

            gallery_html = "" # No gallery for GT mode

        else: # Fallback for unknown mode
            print(f"Warning: Unknown mode '{self.mode}'")
            # Just show raw frames combined
            raw_frames_rgb = {cam_id: cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for cam_id, f in frames_bgr.items() if f is not None}
            combined_raw_rgb = self._combine_frames(raw_frames_rgb)
            output_image_rgb = combined_raw_rgb
            map_placeholder_rgb = cv2.cvtColor(np.full((self.map_height, self.map_width, 3), 240, dtype=np.uint8), cv2.COLOR_BGR2RGB) # Grey map
            map_img_rgb = map_placeholder_rgb
            gallery_html = ""

        return output_image_rgb, map_img_rgb, gallery_html


    def _on_frame_change(self, frame_index):
        """Callback when the frame slider value changes (on release)."""
        if self.dataset_path is None or not self.camera_dirs:
            return None, None, "<p>Dataset not loaded. Click Start.</p>"

        try:
            self.current_frame_index = int(frame_index)
        except (ValueError, TypeError):
            print(f"Warning: Invalid frame index value received: {frame_index}. Using 0.")
            self.current_frame_index = 0
            # Optionally update slider to 0? gr.update(value=0)? Needs slider as output.

        # Pass None for preloaded_frames, so it loads them inside
        # This call will load BGR frames, process them according to mode, and return RGB images/HTML
        output_image_rgb, map_img_rgb, gallery_html = self._process_and_get_outputs(self.current_frame_index, None)

        # Return the results to update the UI components
        return output_image_rgb, map_img_rgb, gallery_html


    def _on_next_frame(self, current_slider_value):
        """Callback for the 'Next Frame' button."""
        if self.dataset_path is None or not self.camera_dirs:
             return current_slider_value, "Dataset not loaded."

        max_frame_index = 0
        # Recalculate max frames
        if self.camera_dirs:
            first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
            if os.path.isdir(first_cam_rgb_dir):
                try:
                    frame_files = [f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')]
                    max_frame_index = len(frame_files) - 1 if frame_files else 0
                except Exception as e:
                    print(f"Warning: Error reading frame count on next frame: {e}")
                    # Try to get max from slider if possible? Or assume 100?
                    # Need slider input for that. Let's assume slider max is correct.
                    # We'll just cap based on current value + 1 vs itself.
                    pass # Keep max_frame_index as 0 or its previous value

        try:
            current_idx = int(current_slider_value)
        except (ValueError, TypeError):
            print(f"Warning: Invalid slider value '{current_slider_value}' on next frame. Using 0.")
            current_idx = 0

        next_frame_index = current_idx + 1
        if max_frame_index > 0: # Only cap if we have a valid max > 0
            next_frame_index = min(next_frame_index, max_frame_index)
        elif next_frame_index < 0: # Prevent going below 0
            next_frame_index = 0

        self.current_frame_index = next_frame_index # Update internal state

        status = f"Advanced to frame {self.current_frame_index}"
        if max_frame_index > 0 and next_frame_index == max_frame_index:
            status += " (End Reached)"

        # Use gr.update to change the slider's value in the UI
        return gr.update(value=self.current_frame_index), status


    def _on_clear_selection(self):
        """Callback to clear the selected track ID in Model Detection mode."""
        if self.mode == "Model Detection":
            # Call select_person with None to clear selection
            status = self.tracker.select_person(None) # Pass None explicitly
            return status
        else:
            return "Clear selection only applicable in Model Detection mode."

    def _on_refresh(self, frame_slider_value):
        """Callback for the 'Refresh Display' button."""
        print("Refreshing display...")
        # Re-process the current frame index obtained from the slider
        try:
            frame_index = int(frame_slider_value)
        except (ValueError, TypeError):
            print(f"Warning: Invalid frame index value on refresh: {frame_slider_value}. Using 0.")
            frame_index = 0

        self.current_frame_index = frame_index # Update internal state

        # Pass None for preloaded_frames, so it loads them inside
        output_image_rgb, map_img_rgb, gallery_html = self._process_and_get_outputs(frame_index, None)

        # Return update for slider value (to keep it consistent) plus the new outputs
        return gr.update(value=frame_index), output_image_rgb, map_img_rgb, gallery_html


    def _on_track_person(self, track_id):
        """Callback when a gallery button (linked to hidden button) is clicked."""
        if self.mode == "Model Detection":
             try:
                 # track_id comes from the hidden button's value (or gr.Number input)
                 track_id_int = int(track_id)
                 # select_person handles the logic for selecting/deselecting/blocking -1
                 status = self.tracker.select_person(track_id_int)
                 return status
             except (ValueError, TypeError):
                 return f"Error: Invalid track ID received: {track_id}"
             except Exception as e:
                  return f"Error selecting person: {e}"
        else:
             return "Tracking selection only applicable in Model Detection mode."

    def _load_frames(self, frame_index: int) -> Dict[str, Optional[np.ndarray]]:
        """Loads BGR frames for the given index from all camera directories."""
        frames_bgr = {}
        if not self.camera_dirs:
            print("Warning: No camera directories loaded.")
            return frames_bgr # Return empty dict

        max_frames = 0
        # Use first camera to estimate max frames
        if self.camera_dirs:
            first_cam_rgb_dir = os.path.join(self.camera_dirs[0], 'rgb')
            if os.path.isdir(first_cam_rgb_dir):
                try:
                    frame_files = [f for f in os.listdir(first_cam_rgb_dir) if f.lower().endswith('.jpg')]
                    max_frames = len(frame_files)
                except Exception as e:
                    print(f"Warning: Could not determine max frames in _load_frames: {e}")
                    max_frames = 0 # Indicate unknown max

        # Basic bounds check
        if frame_index < 0:
            # print(f"Warning: Requested frame index {frame_index} is negative. Clamping to 0.")
            frame_index = 0
        if max_frames > 0 and frame_index >= max_frames:
            # print(f"Warning: Requested frame index {frame_index} exceeds max ({max_frames-1}). Clamping to max.")
            frame_index = max_frames - 1

        # Format filename (assuming 6 digits, zero-padded) -> Matches MTMMC format
        frame_file_name = f'{frame_index + 1:06d}.jpg' # MTMMC uses 1-based naming usually

        # Determine placeholder dimensions
        h_placeholder = self.frame_height if self.frame_height else 1080 # Use known or default HD
        w_placeholder = self.frame_width if self.frame_width else 1920

        for cam_dir in self.camera_dirs:
            cam_id = os.path.basename(cam_dir)
            image_path = os.path.join(cam_dir, 'rgb', frame_file_name)
            img = None # Initialize img to None
            try:
                if os.path.exists(image_path):
                    img = cv2.imread(image_path) # Reads in BGR format
                    if img is None:
                        print(f"Warning: Failed to read image (cv2.imread returned None) {image_path}")
                    else:
                        # Update frame dimensions if not set yet (using first valid loaded frame)
                        if self.frame_height is None or self.frame_width is None:
                            self.frame_height, self.frame_width = img.shape[:2]
                            print(f"Frame dimensions set from {cam_id}: {self.frame_width}x{self.frame_height}")
                            # Update placeholder dimensions now that we know the real ones
                            h_placeholder = self.frame_height
                            w_placeholder = self.frame_width
                # else: # File not found - expected if cameras have different lengths, suppress msg
                #     pass

            except Exception as e:
                 print(f"Error reading image {image_path}: {e}")
                 img = None # Ensure img is None on error


            # Assign the loaded image or a placeholder
            if img is not None:
                 frames_bgr[cam_id] = img
            else:
                 # Create a black placeholder BGR image
                 placeholder_img = np.zeros((h_placeholder, w_placeholder, 3), dtype=np.uint8)
                 # Add text to placeholder
                 cv2.putText(placeholder_img, f"No Image ({cam_id} - F:{frame_index+1})", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                 frames_bgr[cam_id] = placeholder_img

        return frames_bgr


    def _combine_frames(self, frames_dict: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """
        Combines multiple frames/maps (assumed same color space, e.g., all BGR or all RGB)
        into a grid. Handles None values and potentially different sizes by resizing to the first valid frame's size.
        Returns the combined image in the same color space.
        """
        if not frames_dict:
            return None

        # Filter out None values and get camera IDs in sorted order for consistent layout
        valid_frames_data = {
            cam_id: f for cam_id, f in frames_dict.items()
            if f is not None and f.size > 0
        }
        sorted_cam_ids = sorted(valid_frames_data.keys())
        valid_frames = [valid_frames_data[cam_id] for cam_id in sorted_cam_ids]


        if not valid_frames:
            print("Warning: No valid frames to combine.")
            return np.zeros((100, 100, 3), dtype=np.uint8) # Small black placeholder

        num_items = len(valid_frames)

        if num_items == 1:
            return valid_frames[0].copy()

        # Determine grid layout
        rows = int(np.ceil(np.sqrt(num_items)))
        cols = int(np.ceil(num_items / rows))

        # Get dimensions and dtype from the *first* valid frame. Assume others should match.
        try:
            target_height, target_width, channels = valid_frames[0].shape
            dtype = valid_frames[0].dtype
        except Exception as e:
            print(f"Error getting shape/dtype from first valid frame: {e}. Cannot combine.")
            return np.zeros((100, 100, 3), dtype=np.uint8) # Placeholder on error

        # Create the combined image canvas
        combined_image = np.zeros((rows * target_height, cols * target_width, channels), dtype=dtype)

        # Fill the grid
        frame_idx = 0
        for i in range(rows):
            for j in range(cols):
                if frame_idx < num_items:
                    current_frame = valid_frames[frame_idx]
                    try:
                        # Resize frame if it doesn't match the target dimensions
                        if current_frame.shape != (target_height, target_width, channels):
                             print(f"Warning: Frame {frame_idx} (CamID: {sorted_cam_ids[frame_idx]}) has mismatched shape ({current_frame.shape}), expected ({target_height, target_width, channels}). Resizing.")
                             current_frame = cv2.resize(current_frame, (target_width, target_height))

                        # Place the (potentially resized) frame into the grid
                        combined_image[i * target_height:(i + 1) * target_height, j * target_width:(j + 1) * target_width, :] = current_frame

                    except Exception as e:
                        print(f"Error placing frame {frame_idx} (CamID: {sorted_cam_ids[frame_idx]}) into combined grid: {e}")
                        # Optionally draw an error placeholder in the slot
                        error_slot = np.zeros((target_height, target_width, channels), dtype=dtype)
                        cv2.putText(error_slot, "ERR", (target_width//3, target_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        combined_image[i * target_height:(i + 1) * target_height, j * target_width:(j + 1) * target_width, :] = error_slot


                    frame_idx += 1
                # else: Leave remaining grid cells black (already initialized)

        return combined_image