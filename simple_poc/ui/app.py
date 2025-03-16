import os

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

    def build_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# MTMMC Person Tracking")

            with gr.Row():
                with gr.Column(scale=1):
                    dataset_path = gr.Textbox(label="Dataset Path", value="/Volumes/One Touch/MTMMC/train/train/s01/")
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
                inputs=[dataset_path],
                outputs=[status_output, frame_slider, pause_checkbox, image_output, map_output, gallery_output]
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

    def _on_start(self, dataset_path):
        self.dataset_path = dataset_path
        self.camera_dirs = sorted([os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if
                                   os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('c')])
        gt_path = os.path.join(self.camera_dirs[0].replace('rgb', 'gt'), "gt", 'gt.txt')
        self.gt_files = gt_path
        self.current_frame_index = 0
        self.paused = False

        if not self.camera_dirs:
            return "Error: No camera directories found", gr.update(), gr.update(value=True), None, None, ""

        frames = self._load_frames(0)
        annotated_frames, map_img = self._process_multiple_frames(frames)
        output_image = self._combine_frames(annotated_frames)

        return (
            "Dataset loaded and tracking started",
            gr.update(maximum=len(os.listdir(os.path.join(self.camera_dirs[0], 'rgb'))) - 1, value=0),
            gr.update(value=False),
            output_image,
            map_img,
            create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)
        )

    def _toggle_playback(self, paused):
        self.paused = paused
        return f"Video {'paused' if paused else 'playing'}"

    def _on_frame_change(self, frame_index):
        self.current_frame_index = frame_index
        frames = self._load_frames(frame_index)
        annotated_frames, map_img = self._process_multiple_frames(frames)
        output_image = self._combine_frames(annotated_frames)
        return output_image, map_img, create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)

    def _on_next_frame(self, frame_slider):
        self.current_frame_index = min(self.current_frame_index + 1,
                                       len(os.listdir(os.path.join(self.camera_dirs[0], 'rgb'))) - 1)
        return gr.update(value=self.current_frame_index), f"Advanced to frame {self.current_frame_index}"

    def _on_clear_selection(self):
        self.tracker.selected_track_id = None
        return "Cleared selection"

    def _on_refresh(self, frame_slider):
        return gr.update(value=self.current_frame_index), *self._on_frame_change(self.current_frame_index)

    def _on_track_person(self, track_id):
        return self.tracker.select_person(int(track_id))

    def _process_multiple_frames(self, frames):
        person_crops, map_img = self.tracker.process_multiple_frames(frames, self.paused)
        print(f"Type of person_crops: {type(person_crops)}")
        if isinstance(person_crops, dict):
            for key, value in person_crops.items():
                print(f"Type of person_crops[{key}]: {type(value)}")
        return person_crops, map_img

    def _load_frames(self, frame_index):
        frames = {}
        for cam_dir in self.camera_dirs:
            image_path = os.path.join(cam_dir, 'rgb', f'{frame_index:06d}.jpg')
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                print(f"Type of loaded image from {image_path}: {type(img)}")
                frames[cam_dir.split('/')[-1]] = img
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

        height, width, _ = list(frames.values())[0].shape
        combined_image = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
        camera_index = 0
        for i in range(rows):
            for j in range(cols):
                if camera_index < num_cameras:
                    camera_id = list(frames.keys())[camera_index]
                    combined_image[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = frames[camera_id]
                    camera_index += 1

        return combined_image

