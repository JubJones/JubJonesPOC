import gradio as gr

from simple_poc.tracking.tracker import PersonTracker
from simple_poc.ui.gallery import create_gallery_html
from simple_poc.ui.video_player import VideoPlayer


class PersonTrackerApp:
    def __init__(self, model_path="yolo11n.pt"):
        self.tracker = PersonTracker(model_path)
        self.video_player = VideoPlayer()

    def build_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Person Tracking with YOLO")

            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Input Video")
                    video_path = gr.Textbox(label="Or enter video path", value="test.avi")

                    with gr.Row():
                        start_btn = gr.Button("Start Tracking")
                        pause_checkbox = gr.Checkbox(label="Pause", value=True)

                    frame_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="Frame Position"
                    )
                    next_frame_btn = gr.Button("Next Frame")
                    clear_btn = gr.Button("Clear Selection")
                    refresh_btn = gr.Button("Refresh Display")

                    # Hidden buttons for person tracking
                    with gr.Column(visible=False):
                        track_buttons = {i: gr.Button(f"Track {i}", elem_id=f"track_button_{i}")
                                         for i in range(1, 50)}

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
                self._on_start,
                inputs=[video_input, video_path],
                outputs=[status_output, frame_slider, pause_checkbox,
                         video_output, map_output, gallery_output]
            )

            pause_checkbox.change(
                self._toggle_playback,
                inputs=[pause_checkbox],
                outputs=[status_output]
            )

            frame_slider.change(
                self._on_frame_change,
                inputs=[frame_slider, video_path, pause_checkbox],
                outputs=[video_output, map_output, gallery_output]
            )

            next_frame_btn.click(
                self._on_next_frame,
                inputs=[video_path, frame_slider, pause_checkbox],
                outputs=[frame_slider, status_output]
            ).then(
                self._on_frame_change,
                inputs=[frame_slider, video_path, pause_checkbox],
                outputs=[video_output, map_output, gallery_output]
            )

            clear_btn.click(
                self._on_clear_selection,
                outputs=[status_output]
            ).then(
                self._on_frame_change,
                inputs=[frame_slider, video_path, pause_checkbox],
                outputs=[video_output, map_output, gallery_output]
            )

            refresh_btn.click(
                self._on_refresh,
                inputs=[video_path, frame_slider, pause_checkbox],
                outputs=[frame_slider, video_output, map_output, gallery_output]
            )

            # Connect track buttons
            for i, btn in track_buttons.items():
                btn.click(
                    self._on_track_person,
                    inputs=gr.Number(value=i, visible=False),
                    outputs=[status_output]
                ).then(
                    self._on_frame_change,
                    inputs=[frame_slider, video_path, pause_checkbox],
                    outputs=[video_output, map_output, gallery_output]
                )

        return demo

    def _on_start(self, video_input, video_path):
        path = video_input if video_input else video_path

        if not self.video_player.open_video(path):
            return "Error: Could not open video", gr.update(), gr.update(value=True), None, None, ""

        frame = self.video_player.get_frame(0)
        if frame is None:
            return "Error: Could not read frame", gr.update(), gr.update(value=True), None, None, ""

        annotated_frame, map_img = self.tracker.process_frame(frame)
        self.video_player.state["playing"] = True
        self.video_player.state["current_frame"] = 0
        self.video_player.toggle_playback(False, self._frame_callback)

        return (
            "Video loaded and playing",
            gr.update(maximum=self.video_player.state["total_frames"] - 1, value=0),
            gr.update(value=False),
            annotated_frame,
            map_img,
            create_gallery_html(self.tracker.person_crops, self.tracker.selected_track_id)
        )

    def _frame_callback(self, frame_index):
        # This would typically update shared state that UI can check
        pass

    def _toggle_playback(self, paused):
        return self.video_player.toggle_playback(paused, self._frame_callback)

    def _on_frame_change(self, frame_index, video_path, paused):
        frame = self.video_player.get_frame(frame_index, video_path)
        if frame is None:
            return None, None, "<p>Error reading frame</p>"

        annotated_frame, map_img = self.tracker.process_frame(frame, paused)
        gallery_html = create_gallery_html(
            self.tracker.person_crops,
            self.tracker.selected_track_id
        )

        return annotated_frame, map_img, gallery_html

    def _on_next_frame(self, video_path, frame_slider, paused):
        next_frame, message = self.video_player.advance_frame()
        return gr.update(value=next_frame), message

    def _on_clear_selection(self):
        self.tracker.selected_track_id = None
        return "Cleared selection"

    def _on_refresh(self, video_path, frame_slider, paused):
        if not self.video_player.state["playing"] or paused:
            return gr.update(), gr.update(), gr.update(), gr.update()

        current = self.video_player.state["current_frame"]
        if current != frame_slider and current < self.video_player.state["total_frames"]:
            frame = self.video_player.get_frame(current, video_path)
            if frame is not None:
                annotated_frame, map_img = self.tracker.process_frame(frame, False)
                gallery_html = create_gallery_html(
                    self.tracker.person_crops,
                    self.tracker.selected_track_id
                )
                return gr.update(value=current), annotated_frame, map_img, gallery_html

        return gr.update(), gr.update(), gr.update(), gr.update()

    def _on_track_person(self, track_id):
        return self.tracker.select_person(int(track_id))
