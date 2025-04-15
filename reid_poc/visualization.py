"""Functions for drawing annotations on frames and displaying combined views."""

import logging
from typing import Dict, List
import cv2
import numpy as np

from reid_poc.alias_types import CameraID, FrameData, TrackData # Use relative import

logger = logging.getLogger(__name__)

def draw_annotations(
    frames: Dict[CameraID, FrameData],
    processed_results: Dict[CameraID, List[TrackData]],
    draw_bboxes: bool = True,
    show_track_id: bool = True,
    show_global_id: bool = True
) -> Dict[CameraID, FrameData]:
    """Draws bounding boxes and IDs onto the provided frames based on processed results."""
    annotated_frames: Dict[CameraID, FrameData] = {}
    default_frame_h, default_frame_w = 1080, 1920 # Default placeholder size
    first_valid_frame_found = False

    # Determine a consistent frame size for placeholders
    for frame in frames.values():
        if frame is not None and frame.size > 0:
            default_frame_h, default_frame_w = frame.shape[:2]
            first_valid_frame_found = True
            break

    for cam_id, frame in frames.items():
        current_h, current_w = default_frame_h, default_frame_w
        # Create a copy to draw on or a placeholder if frame is missing
        if frame is None or frame.size == 0:
            placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            annotated_frames[cam_id] = placeholder
            continue
        else:
            # Work on a copy to avoid modifying the original frame data
            annotated_frame = frame.copy()
            current_h, current_w = frame.shape[:2]

        results_for_cam = processed_results.get(cam_id, [])

        for track_info in results_for_cam:
            bbox = track_info.get('bbox_xyxy')
            track_id = track_info.get('track_id')
            global_id = track_info.get('global_id') # Can be None

            if bbox is None:
                continue

            # Ensure coordinates are valid integers within frame bounds
            try:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(current_w - 1, x2)
                y2 = min(current_h - 1, y2)
                # Skip drawing if the box has no area after clamping
                if x1 >= x2 or y1 >= y2:
                    continue
            except (ValueError, TypeError):
                logger.warning(f"[{cam_id}] Invalid bbox coordinates received: {bbox}")
                continue

            # --- Determine Color ---
            color = (200, 200, 200) # Default color (e.g., gray) for tracks without global ID
            if global_id is not None:
                # Generate a pseudo-random color based on the global ID
                seed = int(global_id) * 3 + 5 # Simple seeding
                # Ensure color components are within valid range [0, 255]
                # Use modulo and add offset for variability, avoiding pure black/white
                color = (
                    (seed * 41) % 200 + 55,
                    (seed * 17) % 200 + 55,
                    (seed * 29) % 200 + 55
                )

            # --- Draw Bounding Box ---
            if draw_bboxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # --- Prepare and Draw Label ---
            label_parts = []
            if show_track_id and track_id is not None:
                label_parts.append(f"T:{track_id}")
            if show_global_id:
                # Display Global ID or '?' if None
                label_parts.append(f"G:{global_id if global_id is not None else '?'}")

            label = " ".join(label_parts)

            if label:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1

                # Calculate text size and position
                (text_w, text_h), baseline = cv2.getTextSize(label, font_face, font_scale, thickness + 1)

                # Position label above the box, move below if it goes off-screen top
                label_y_pos = y1 - baseline - 5
                if label_y_pos < text_h: # Check if top of text is offscreen
                    label_y_pos = y2 + text_h + baseline + 5 # Place below box

                # Clamp label position to be within frame vertically
                label_y_pos = max(text_h + baseline, min(label_y_pos, current_h - baseline - 1))
                 # Clamp label start horizontally
                label_x_pos = max(0, x1)

                # Draw a filled rectangle background for the label
                bg_x1 = label_x_pos
                bg_y1 = label_y_pos - text_h - baseline
                bg_x2 = label_x_pos + text_w
                bg_y2 = label_y_pos + baseline

                # Clamp background rectangle coordinates
                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(current_w - 1, bg_x2)
                bg_y2 = min(current_h - 1, bg_y2)

                # Ensure background box has valid dimensions before drawing
                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                     cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)
                     # Draw the text on top of the background (use black for contrast)
                     cv2.putText(annotated_frame, label, (label_x_pos, label_y_pos), font_face, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        annotated_frames[cam_id] = annotated_frame
    return annotated_frames


def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines multiple annotated frames into a grid and displays them in a single window."""
    valid_annotated = [f for f in annotated_frames.values() if f is not None and f.size > 0]

    if not valid_annotated:
        # Display a placeholder if no valid frames are available
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames to Display", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated)
        # Determine grid layout (try to make it squarish)
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))

        # Assume all valid frames have the same shape, use the first one
        target_h, target_w = valid_annotated[0].shape[:2]

        combined_h = rows * target_h
        combined_w = cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        frame_idx_display = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx_display < num_cams:
                    frame = valid_annotated[frame_idx_display]
                    # Resize frame only if necessary (e.g., if input frames had different resolutions)
                    if frame.shape[:2] != (target_h, target_w):
                        try:
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err:
                            logger.warning(f"Error resizing frame for grid display: {resize_err}. Using black placeholder.")
                            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                    # Place the frame into the combined display grid
                    try:
                        combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = frame
                    except ValueError as slice_err:
                        logger.error(f"Error placing frame into grid (shape mismatch?): {slice_err}. Target slice: H={target_h},W={target_w}. Frame shape: {frame.shape}")
                        # Optionally fill the slot with a black rectangle
                        combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = 0

                    frame_idx_display += 1

        # --- Resize the final combined image if it exceeds max_width ---
        disp_h, disp_w = combined_display.shape[:2]
        if disp_w > max_width:
            try:
                scale = max_width / disp_w
                disp_h_new = int(disp_h * scale)
                disp_w_new = max_width
                combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err:
                logger.error(f"Failed to resize final combined display: {final_resize_err}")

    # --- Show the image ---
    try:
        # Check if window still exists before showing
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(window_name, combined_display)
        else:
            # logger.info("Display window closed, cannot show image.") # Reduce log noise
            pass
    except Exception as e:
        logger.error(f"Error during cv2.imshow: {e}")