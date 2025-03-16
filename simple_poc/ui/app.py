import gradio as gr
import os
import cv2
import numpy as np
from simple_poc.tracking.tracker import PersonTracker
from simple_poc.ui.gallery import create_gallery_html


class MTMMCTrackerApp:
    def __init__(self, model_path="yolo11n.pt"):
        self.tracker = PersonTracker(model_path)
        self.dataset_path = None
        self.image_files = []
        self.gt_files = None  # Changed to a single file path
        self.current_frame_index = 0
        self.paused = True
        self.gt_data = {}  # Store ground truth data from gt.txt

    def build_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# MTMMC Person Tracking")

            with gr.Row():
                with gr.Column(scale=1):
                    dataset_path = gr.Textbox(label="Dataset Path", value="/path/to/MTMMC/train/train/s01/c01/rgb")
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
        self.dataset_path = f"{dataset_path}"
        self.image_files = sorted(
            [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')])
        gt_path = dataset_path.replace('rgb', 'gt')
        self.gt_files = gt_path + "/gt.txt"
        self.current_frame_index = 0
        self.paused = False
        self._load_all_ground_truth()

        if not self.image_files:
            return "Error: No images found", gr.update(), gr.update(value=True), None, None, ""

        image = cv2.imread(self.image_files[0])
        annotated_frame, map_img = self._process_frame(image)

        return (
            "Dataset loaded and tracking started",
            gr.update(maximum=len(self.image_files) - 1, value=0),
            gr.update(value=False),
            annotated_frame,
            map_img,
            create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)
        )

    def _toggle_playback(self, paused):
        self.paused = paused
        return f"Video {'paused' if paused else 'playing'}"

    def _on_frame_change(self, frame_index):
        self.current_frame_index = frame_index
        image = cv2.imread(self.image_files[frame_index])
        annotated_frame, map_img = self._process_frame(image)
        return annotated_frame, map_img, create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)

    def _on_next_frame(self, frame_slider):
        self.current_frame_index = min(self.current_frame_index + 1, len(self.image_files) - 1)
        return gr.update(value=self.current_frame_index), f"Advanced to frame {self.current_frame_index}"

    def _on_clear_selection(self):
        self.tracker.selected_track_id = None
        return "Cleared selection"

    def _on_refresh(self, frame_slider):
        return gr.update(value=self.current_frame_index), *self._on_frame_change(self.current_frame_index)

    def _on_track_person(self, track_id):
        return self.tracker.select_person(int(track_id))

    def _process_frame(self, frame):
        gt_boxes, track_ids = self._get_ground_truth(self.current_frame_index)
        self.tracker.current_boxes = gt_boxes
        self.tracker.current_track_ids = track_ids
        return self.tracker.process_frame(frame, self.paused)

    def _load_all_ground_truth(self):
        self.gt_data = {}
        with open(self.gt_files, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id, obj_id, x1, y1, w, h = int(parts[0]), int(parts[1]), float(parts[2]), float(
                    parts[3]), float(parts[4]), float(parts[5])
                if frame_id not in self.gt_data:
                    self.gt_data[frame_id] = []
                self.gt_data[frame_id].append((obj_id, np.array([x1, y1, w, h])))

    def _get_ground_truth(self, frame_id):
        boxes = []
        track_ids = []
        if frame_id in self.gt_data:
            for obj_id, box in self.gt_data[frame_id]:
                boxes.append(box)
                track_ids.append(obj_id)
        return boxes, track_ids


if __name__ == "__main__":
    app = MTMMCTrackerApp(model_path="yolo11n.pt")
    demo = app.build_ui()
    demo.launch(share=True)
