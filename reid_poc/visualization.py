"""Functions for drawing annotations on frames and displaying combined views."""

import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from reid_poc.alias_types import CameraID, FrameData, TrackData, TrackKey, ExitRule, HandoffTriggerInfo # Use relative import

logger = logging.getLogger(__name__)

def draw_annotations(
    frames: Dict[CameraID, FrameData],
    processed_results: Dict[CameraID, List[TrackData]],
    handoff_triggers: List[HandoffTriggerInfo], # Added parameter
    frame_shapes: Dict[CameraID, Optional[Tuple[int, int]]], # Needed for quadrant lines
    draw_bboxes: bool = True,
    show_track_id: bool = True,
    show_global_id: bool = True,
    draw_quadrants: bool = True, # Configurable drawing
    highlight_triggers: bool = True # Configurable drawing
) -> Dict[CameraID, FrameData]:
    """Draws bounding boxes, IDs, quadrant lines, and handoff triggers onto frames."""
    annotated_frames: Dict[CameraID, FrameData] = {}
    default_frame_h, default_frame_w = 1080, 1920 # Default placeholder size

    # Try to get a size from the provided shapes map first
    first_valid_shape = next((shape for shape in frame_shapes.values() if shape), None)
    if first_valid_shape:
        default_frame_h, default_frame_w = first_valid_shape
    else: # Fallback: check actual frames if shapes weren't available
        first_valid_frame = next((f for f in frames.values() if f is not None and f.size > 0), None)
        if first_valid_frame is not None:
            default_frame_h, default_frame_w = first_valid_frame.shape[:2]

    # --- Prepare Trigger Map for quick lookup ---
    # Maps source_track_key -> rule for easy checking during drawing
    trigger_map: Dict[TrackKey, ExitRule] = {
        trigger.source_track_key: trigger.rule
        for trigger in handoff_triggers
    }

    for cam_id, frame in frames.items():
        current_h, current_w = default_frame_h, default_frame_w
        shape_from_map = frame_shapes.get(cam_id)
        if shape_from_map:
            current_h, current_w = shape_from_map
        elif frame is not None:
            current_h, current_w = frame.shape[:2] # Use actual frame shape if available

        # Create a copy to draw on or a placeholder if frame is missing/invalid
        if frame is None or frame.size == 0 or current_h <= 0 or current_w <= 0:
            placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            annotated_frames[cam_id] = placeholder
            continue
        else:
            annotated_frame = frame.copy()

        # --- Draw Quadrant Lines ---
        if draw_quadrants:
            mid_x, mid_y = current_w // 2, current_h // 2
            line_color = (100, 100, 100) # Dark Gray
            cv2.line(annotated_frame, (mid_x, 0), (mid_x, current_h), line_color, 1)
            cv2.line(annotated_frame, (0, mid_y), (current_w, mid_y), line_color, 1)

        # --- Draw Tracks and Potential Triggers ---
        results_for_cam = processed_results.get(cam_id, [])
        for track_info in results_for_cam:
            bbox = track_info.get('bbox_xyxy')
            track_id = track_info.get('track_id')
            global_id = track_info.get('global_id') # Can be None

            if bbox is None or track_id is None: # Need bbox and track_id
                continue

            current_track_key: TrackKey = (cam_id, track_id)

            # Check if this track triggered a handoff
            triggered_rule = trigger_map.get(current_track_key) if highlight_triggers else None

            # Ensure coordinates are valid integers within frame bounds
            try:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(current_w - 1, x2)
                y2 = min(current_h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue # Skip zero-area boxes
            except (ValueError, TypeError):
                logger.warning(f"[{cam_id}] Invalid bbox coordinates for T:{track_id}: {bbox}")
                continue

            # --- Determine Color and Thickness ---
            color = (200, 200, 200) # Default: Light Gray (track without GID)
            thickness = 2
            if global_id is not None:
                # Generate pseudo-random color based on Global ID
                seed = int(global_id) * 3 + 5
                color = ((seed * 41) % 200 + 55, (seed * 17) % 200 + 55, (seed * 29) % 200 + 55)

            if triggered_rule:
                color = (0, 0, 255) # Bright Red for trigger
                thickness = 3

            # --- Draw Bounding Box ---
            if draw_bboxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # --- Prepare and Draw Label ---
            label_parts = []
            if show_track_id:
                label_parts.append(f"T:{track_id}")
            if show_global_id:
                label_parts.append(f"G:{global_id if global_id is not None else '?'}")

            label = " ".join(label_parts)
            trigger_label = ""
            if triggered_rule:
                trigger_label = f"{cam_id} -> {triggered_rule.target_cam_id}"

            if label or trigger_label:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_base = 0.6
                thickness_text = 1

                # Position calculation needs to account for potentially two lines of text
                text_y_pos = y1 - 10 # Start position for the top line
                line_height = 20 # Approx height of a line

                # Adjust if starting position is too high
                if text_y_pos < line_height * (1 if not trigger_label else 2):
                     text_y_pos = y2 + line_height # Move below box

                # Draw Trigger Label (if exists) - Cyan color
                if trigger_label:
                    trigger_color = (255, 255, 0) # Cyan
                    (tw, th), bl = cv2.getTextSize(trigger_label, font_face, font_scale_base, thickness_text + 1)
                    # Simple background rectangle (optional, can clutter)
                    # cv2.rectangle(annotated_frame, (x1, text_y_pos - th - bl), (x1 + tw, text_y_pos + bl), (50, 50, 50), cv2.FILLED)
                    cv2.putText(annotated_frame, trigger_label, (x1, text_y_pos),
                                font_face, font_scale_base, trigger_color, thickness_text + 1, cv2.LINE_AA) # Bold trigger text
                    text_y_pos += line_height # Move down for the next label

                # Draw Main Label (Track/Global ID) - Use box color
                if label:
                    (tw, th), bl = cv2.getTextSize(label, font_face, font_scale_base, thickness_text)
                     # Simple background rectangle (optional, can clutter)
                     # cv2.rectangle(annotated_frame, (x1, text_y_pos - th - bl), (x1 + tw, text_y_pos + bl), color, cv2.FILLED)
                    cv2.putText(annotated_frame, label, (x1, text_y_pos),
                                font_face, font_scale_base, color, thickness_text, cv2.LINE_AA)


        annotated_frames[cam_id] = annotated_frame
    return annotated_frames


def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines multiple annotated frames into a grid and displays them."""
    # --- Function remains mostly the same, ensure error handling is robust ---
    valid_annotated_items = sorted([ (cid, f) for cid, f in annotated_frames.items() if f is not None and f.size > 0])
    valid_annotated = [item[1] for item in valid_annotated_items] # Get frames in sorted order

    if not valid_annotated:
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated)
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))

        # Use shape of the first valid frame as target (assuming consistency or resize needed)
        target_h, target_w = valid_annotated[0].shape[:2]
        if target_h <= 0 or target_w <= 0:
            logger.error("First valid frame has invalid shape for tiling.")
            return # Cannot create grid

        combined_h = rows * target_h
        combined_w = cols * target_w
        combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        frame_idx_display = 0
        for r in range(rows):
            for c in range(cols):
                if frame_idx_display < num_cams:
                    frame = valid_annotated[frame_idx_display]
                    # Resize if necessary (should ideally match target_h, target_w)
                    if frame.shape[0] != target_h or frame.shape[1] != target_w:
                        try:
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        except Exception as resize_err:
                            logger.warning(f"Error resizing frame {frame_idx_display} for grid: {resize_err}. Using black.")
                            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                    try:
                        combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = frame
                    except ValueError as slice_err:
                        logger.error(f"Error placing frame {frame_idx_display} into grid (shape mismatch?): {slice_err}. Frame shape: {frame.shape}")
                        combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = 0 # Fill with black

                    frame_idx_display += 1

        # --- Resize final combined image if it exceeds max_width ---
        disp_h, disp_w = combined_display.shape[:2]
        if disp_w > max_width:
            try:
                scale = max_width / disp_w
                disp_h_new = int(disp_h * scale)
                disp_w_new = max_width
                if disp_h_new > 0 and disp_w_new > 0:
                    combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
            except Exception as final_resize_err:
                logger.error(f"Failed to resize final combined display: {final_resize_err}")

    # --- Show the image ---
    try:
        # Check if window still exists before showing
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(window_name, combined_display)
        # else: # Reduce log noise if window is closed
            # logger.debug("Display window closed, cannot show image.")
            # pass
    except cv2.error as cv_err:
         # Can happen if window is closed between check and imshow in rare cases
         logger.debug(f"OpenCV error during imshow (window likely closed): {cv_err}")
    except Exception as e:
        logger.error(f"Unexpected error during cv2.imshow: {e}", exc_info=False)