import os
import json

import cv2
import gradio as gr
import numpy as np

from simple_poc.tracking.tracker import PersonTracker
from simple_poc.ui.gallery import create_gallery_html


class MTMMCTrackerApp:
    def __init__(self, model_path="yolo11n.pt"):
        self.tracker = PersonTracker(model_path)
        self.dataset_path = None
        self.camera_dirs = []
        self.gt_files = None
        self.current_frame_index = 0
        self.paused = True
        self.gt_data = {}
        self.mode = "Model Detection"  # Default mode
        self.json_data = None  # Store loaded JSON data

    def load_gt_data(self, gt_path):
        """Loads ground truth data from a gt.txt file."""
        gt_data = {}
        try:
            with open(gt_path, 'r') as f:
                for line in f:
                    frame_id, person_id, x, y, w, h = map(float, line.strip().split(','))
                    frame_id = int(frame_id)
                    person_id = int(person_id)
                    if frame_id not in gt_data:
                        gt_data[frame_id] = []
                    gt_data[frame_id].append((person_id, x, y, w, h))
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found at {gt_path}")
            return {}  # Return empty dict if file not found
        return gt_data

    def load_json_data(self, json_path):
        """Loads data from the JSON file."""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON file at {json_path}")
            return None

    def draw_gt_boxes(self, frame, frame_index, camera_id):
        """Draws ground truth bounding boxes on the frame."""
        if not self.json_data:
            return frame

        frame_str = f"{frame_index:06d}"  # Format frame index to match file names
        frame_index_int = int(frame_str)

        # Efficient lookup of annotations for the current frame
        for annotation in self.json_data['annotations']:
            if annotation['image_id'] in [img['id'] for img in self.json_data['images'] if
                                          img['file_name'].startswith(
                                              f"train/{self.dataset_path.split('/')[-2]}/{camera_id}/rgb/") and
                                          img['frame_id'] == frame_index_int]:
                bbox = annotation['bbox']
                x, y, w, h = [int(coord) for coord in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes
                instance_id = annotation.get('instance_id', -1)
                cv2.putText(frame, f"ID: {instance_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def build_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# MTMMC Person Tracking")

            with gr.Row():
                with gr.Column(scale=1):
                    dataset_path = gr.Textbox(label="Dataset Path", value="/Volumes/HDD/MTMMC/train/train/s01/")
                    mode_dropdown = gr.Dropdown(
                        ["Model Detection", "Ground Truth"], label="Mode", value="Model Detection"
                    )
                    start_btn = gr.Button("Start Tracking")
                    pause_checkbox = gr.Checkbox(label="Pause", value=True)
                    frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Frame Position")
                    next_frame_btn = gr.Button("Next Frame")
                    clear_btn = gr.Button("Clear Selection")
                    refresh_btn = gr.Button("Refresh Display")

                    with gr.Column(visible=False):
                        track_buttons = {i: gr.Button(f"Track {i}", elem_id=f"track_button_{i}")
                                         for i in range(1, 50)}

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Tracking View"):
                            image_output = gr.Image(label="Tracking")
                        with gr.TabItem("Map View"):
                            map_output = gr.Image(label="Map")

            gr.Markdown("## Detected People")
            gallery_output = gr.HTML()
            status_output = gr.Textbox(label="Status")

            start_btn.click(
                self._on_start,
                inputs=[dataset_path, mode_dropdown],
                outputs=[status_output, frame_slider, pause_checkbox, image_output, map_output, gallery_output]
            )

            mode_dropdown.change(
                self._on_mode_change,
                inputs=[mode_dropdown],
                outputs=[status_output]
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
        return f"Mode changed to {mode}"

    def _on_start(self, dataset_path, mode):
        self.dataset_path = dataset_path
        self.camera_dirs = sorted([os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if
                                   os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('c')])

        self.json_data = self.load_json_data("/Volumes/HDD/MTMMC/kaist_mtmdc_train.json")

        # gt_path = os.path.join(self.camera_dirs[0].replace('rgb', 'gt'), "gt", 'gt.txt')
        # self.gt_files = gt_path
        self.current_frame_index = 0
        self.paused = False
        self.mode = mode

        if not self.camera_dirs:
            return "Error: No camera directories found", gr.update(), gr.update(value=True), None, None, ""

        frames = self._load_frames(0)
        if self.mode == "Model Detection":
            annotated_frames, map_img = self._process_multiple_frames(frames)

        elif self.mode == "Ground Truth":
            annotated_frames = {}
            for cam_id, frame in frames.items():
                annotated_frames[cam_id] = self.draw_gt_boxes(frame.copy(), 0, cam_id)  # Pass camera ID here
            map_img = None
        else:
            annotated_frames = frames
            map_img = None

        output_image = self._combine_frames(annotated_frames)

        max_frames = 0
        for cam_dir in self.camera_dirs:
            rgb_dir = os.path.join(cam_dir, 'rgb')
            if os.path.exists(rgb_dir):
                num_frames = len([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
                max_frames = max(max_frames, num_frames)

        return (
            "Dataset loaded and tracking started",
            gr.update(maximum=max_frames - 1, value=0),
            gr.update(value=False),
            output_image,
            map_img,
            create_gallery_html(self.tracker.person_crops,
                                self.tracker.selected_track_id) if self.mode == "Model Detection" else ""
        )

    def _toggle_playback(self, paused):
        self.paused = paused
        return f"Video {'paused' if paused else 'playing'}"

    def _on_frame_change(self, frame_index):
        self.current_frame_index = int(frame_index)  # Ensure it's an integer
        frames = self._load_frames(self.current_frame_index)

        if self.mode == "Model Detection":
            annotated_frames, map_img = self._process_multiple_frames(frames)
            gallery_html = create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)

        elif self.mode == "Ground Truth":
            annotated_frames = {}
            for cam_id, frame in frames.items():
                annotated_frames[cam_id] = self.draw_gt_boxes(frame.copy(), self.current_frame_index,
                                                              cam_id)  # Pass cam_id
            map_img = None
            gallery_html = ""
        else:
            annotated_frames = frames
            map_img = None
            gallery_html = ""

        output_image = self._combine_frames(annotated_frames)
        return output_image, map_img, gallery_html

    def _on_next_frame(self, frame_slider):
        # Determine max frame index based on available images in the *first* camera directory.
        max_frame_index = len(os.listdir(os.path.join(self.camera_dirs[0], 'rgb'))) - 1
        self.current_frame_index = min(self.current_frame_index + 1, max_frame_index)
        return gr.update(value=self.current_frame_index), f"Advanced to frame {self.current_frame_index}"

    def _on_clear_selection(self):
        self.tracker.selected_track_id = None
        return "Cleared selection"

    def _on_refresh(self, frame_slider):
        # Casting to int is crucial here for consistent behavior
        frame_index = int(self.current_frame_index)
        return gr.update(value=frame_index), *self._on_frame_change(frame_index)

    def _on_track_person(self, track_id):
        return self.tracker.select_person(int(track_id))

    def _process_multiple_frames(self, frames):
        person_crops, map_img = self.tracker.process_multiple_frames(frames, self.paused)
        return person_crops, map_img

    def _load_frames(self, frame_index):
        frames = {}
        for cam_dir in self.camera_dirs:
            image_path = os.path.join(cam_dir, 'rgb', f'{frame_index:06d}.jpg')
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                frames[cam_dir.split('/')[-1]] = img  # Use camera ID as key
            else:
                # Return black image if frame doesn't exist.  Important for synchronization.
                print(f"Warning: Image not found at {image_path}, using a black image as placeholder.")
                frames[cam_dir.split('/')[-1]] = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Correct dimensions
        return frames

    def _combine_frames(self, frames):
        if not frames:
            return None

        # Combine frames into a single image (e.g., grid layout)
        num_cameras = len(frames)
        if num_cameras == 1:
            return list(frames.values())[0]

        # Basic grid layout (adjust as needed)
        rows = int(np.ceil(np.sqrt(num_cameras)))
        cols = int(np.ceil(num_cameras / rows))

        # Get the dimensions from the *first* frame.  Assume all are same size.
        height, width, _ = list(frames.values())[0].shape
        combined_image = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

        camera_index = 0
        for i in range(rows):
            for j in range(cols):
                if camera_index < num_cameras:
                    camera_id = list(frames.keys())[camera_index]
                    # Ensure we don't try to paste a None frame (if image was missing)
                    if frames[camera_id] is not None:
                        combined_image[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = frames[camera_id]
                    camera_index += 1

        return combined_image


if __name__ == "__main__":
    app = MTMMCTrackerApp()
    demo = app.build_ui()
    demo.launch()