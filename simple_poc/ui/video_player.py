import cv2
import time
import threading


class VideoPlayer:
    def __init__(self):
        self.state = {
            "cap": None,
            "total_frames": 0,
            "current_frame": 0,
            "playing": False,
            "video_path": None,
            "play_thread": None,
            "stop_thread": False,
        }
        self.lock = threading.Lock()  # Add lock for thread safety

    def open_video(self, video_path):
        with self.lock:
            if self.state["cap"] is not None:
                self.state["cap"].release()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            self.state["cap"] = cap
            self.state["video_path"] = video_path
            self.state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.state["current_frame"] = 0
            return True

    def get_frame(self, frame_index, video_path=None):
        with self.lock:
            # Handle video switching
            if self.state["cap"] is None or (
                video_path and video_path != self.state["video_path"]
            ):
                if not video_path:
                    return None

                # Stop playback before opening new video
                self._pause_if_playing()

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None

                self.state["cap"] = cap
                self.state["video_path"] = video_path
                self.state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Pause playback while seeking
            was_playing = self._pause_if_playing()

            # Read the frame
            self.state["cap"].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = self.state["cap"].read()

            # Resume playback if needed
            if was_playing:
                self._resume_playback()

            return frame if success else None

    def _pause_if_playing(self):
        """Temporarily pause playback and return previous state"""
        was_playing = self.state["playing"]
        if was_playing:
            self.state["playing"] = False
            self.state["stop_thread"] = True
            if self.state["play_thread"] and self.state["play_thread"].is_alive():
                self.state["play_thread"].join(1.0)
        return was_playing

    def _resume_playback(self):
        """Resume video playback"""
        self.state["playing"] = True
        self.state["stop_thread"] = False
        self.state["play_thread"] = threading.Thread(
            target=self._play_thread_function, args=(self._frame_callback,), daemon=True
        )
        self.state["play_thread"].start()

    def toggle_playback(self, paused, frame_callback):
        with self.lock:
            self.state["playing"] = not paused
            self._frame_callback = frame_callback

            if self.state["playing"]:
                self._pause_if_playing()  # Clear any existing thread
                self._resume_playback()

            return f"Video {'paused' if paused else 'playing'}"

    def _play_thread_function(self, frame_callback):
        fps = 15
        frame_time = 1.0 / fps

        while not self.state["stop_thread"] and self.state["playing"]:
            start_time = time.time()

            with self.lock:
                if self.state["current_frame"] >= self.state["total_frames"] - 1:
                    self.state["playing"] = False
                    break

                self.state["current_frame"] += 1
                current_frame = self.state["current_frame"]

            if frame_callback:
                frame_callback(current_frame)

            elapsed = time.time() - start_time
            time.sleep(max(0, frame_time - elapsed))

    def advance_frame(self):
        with self.lock:
            if not self.state["video_path"]:
                return self.state["current_frame"], "No video loaded"

            # Pause playback while advancing
            was_playing = self._pause_if_playing()

            # Advance frame
            next_frame = min(
                self.state["current_frame"] + 1, self.state["total_frames"] - 1
            )
            self.state["current_frame"] = next_frame

            # Resume if needed
            if was_playing and next_frame < self.state["total_frames"] - 1:
                self._resume_playback()
            elif next_frame == self.state["total_frames"] - 1:
                return next_frame, "End of video reached"

            return next_frame, f"Advanced to frame {next_frame}"
