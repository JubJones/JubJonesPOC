"""Functions for drawing annotations on frames and displaying combined views."""

import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from collections import Counter # Import Counter

from reid_poc.alias_types import CameraID, FrameData, TrackData, TrackKey, ExitRule, HandoffTriggerInfo # Use relative import

logger = logging.getLogger(__name__)

# Define default color (Green)
DEFAULT_COLOR = (0, 255, 0) # Green

def generate_unique_color(global_id: int) -> Tuple[int, int, int]:
    """Generates a unique, deterministic color for a global ID, avoiding pure green."""
    # Generate pseudo-random color based on Global ID
    seed = int(global_id) * 3 + 5
    # Generate values between 55 and 255 to avoid very dark colors
    r = (seed * 41) % 200 + 55
    g = (seed * 17) % 200 + 55
    b = (seed * 29) % 200 + 55
    color = (b, g, r) # OpenCV uses BGR

    # Ensure the generated color is not exactly the default green
    if color == DEFAULT_COLOR:
        # Slightly adjust if it happens to be green
        color = (b, g - 10 if g >= 10 else g + 10, r)
        # Clamp just in case
        color = tuple(max(0, min(255, c)) for c in color)

    return color

def draw_annotations(
    frames: Dict[CameraID, FrameData],
    processed_results: Dict[CameraID, List[TrackData]],
    handoff_triggers: List[HandoffTriggerInfo],
    frame_shapes: Dict[CameraID, Optional[Tuple[int, int]]],
    draw_bboxes: bool = True,
    show_track_id: bool = True,
    show_global_id: bool = True,
    draw_quadrants: bool = True,
    highlight_triggers: bool = True
) -> Dict[CameraID, FrameData]:
    """Draws bounding boxes, IDs, quadrant lines, and handoff triggers onto frames.
       Unique colors are only applied if a global_id is shared by >= 2 tracks this frame.
    """
    annotated_frames: Dict[CameraID, FrameData] = {}
    default_frame_h, default_frame_w = 1080, 1920

    # --- Pre-calculate Global ID counts for this frame ---
    # Count how many times each non-None global_id appears across all cameras
    global_id_counts = Counter()
    all_global_ids_in_frame = []
    for cam_results in processed_results.values():
        for track_info in cam_results:
            gid = track_info.get('global_id')
            if gid is not None:
                all_global_ids_in_frame.append(gid)
    global_id_counts.update(all_global_ids_in_frame)
    # Example: global_id_counts might be {1: 2, 3: 1, 5: 3}
    # GID 1 appears twice, GID 3 once, GID 5 thrice.

    # Try to get a size from the provided shapes map first
    first_valid_shape = next((shape for shape in frame_shapes.values() if shape), None)
    if first_valid_shape:
        default_frame_h, default_frame_w = first_valid_shape
    else: # Fallback: check actual frames if shapes weren't available
        first_valid_frame = next((f for f in frames.values() if f is not None and f.size > 0), None)
        if first_valid_frame is not None:
            default_frame_h, default_frame_w = first_valid_frame.shape[:2]

    # --- Prepare Trigger Map for quick lookup ---
    trigger_map: Dict[TrackKey, ExitRule] = {
        trigger.source_track_key: trigger.rule
        for trigger in handoff_triggers
    }

    # --- Main Drawing Loop ---
    for cam_id, frame in frames.items():
        current_h, current_w = default_frame_h, default_frame_w
        shape_from_map = frame_shapes.get(cam_id)
        if shape_from_map:
            current_h, current_w = shape_from_map
        elif frame is not None:
            current_h, current_w = frame.shape[:2]

        # Create a copy to draw on or a placeholder
        if frame is None or frame.size == 0 or current_h <= 0 or current_w <= 0:
            placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            annotated_frames[cam_id] = placeholder
            continue
        else:
            annotated_frame = frame.copy()

        # --- Draw Quadrant Lines ---
        if draw_quadrants:
            mid_x, mid_y = current_w // 2, current_h // 2
            line_color = (100, 100, 100)
            cv2.line(annotated_frame, (mid_x, 0), (mid_x, current_h), line_color, 1)
            cv2.line(annotated_frame, (0, mid_y), (current_w, mid_y), line_color, 1)

        # --- Draw Tracks ---
        results_for_cam = processed_results.get(cam_id, [])
        for track_info in results_for_cam:
            bbox = track_info.get('bbox_xyxy')
            track_id = track_info.get('track_id')
            global_id = track_info.get('global_id') # Can be None

            if bbox is None or track_id is None:
                continue

            current_track_key: TrackKey = (cam_id, track_id)

            # Validate coordinates
            try:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(current_w - 1, x2), min(current_h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue
            except (ValueError, TypeError):
                logger.warning(f"[{cam_id}] Invalid bbox coordinates for T:{track_id}: {bbox}")
                continue

            # --- Determine Color and Thickness (MODIFIED LOGIC) ---
            color = DEFAULT_COLOR # Start with default green
            thickness = 2

            if global_id is not None:
                # Check if this global_id is shared by multiple active tracks in this frame
                if global_id_counts.get(global_id, 0) > 1:
                    # Only generate unique color if it's currently part of a confirmed pair/group
                    color = generate_unique_color(global_id)
                # else: color remains DEFAULT_COLOR (green) for tracks with a global_id
                #       that is currently unique (only appears once in this frame's results)

            # Check for handoff trigger AFTER color determination
            triggered_rule = trigger_map.get(current_track_key) if highlight_triggers else None
            if triggered_rule:
                # Thicken the box, color is already determined (green or unique shared color)
                thickness = 3

            # --- Draw Bounding Box ---
            if draw_bboxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # --- Prepare and Draw Label ---
            label_parts = []
            if show_track_id:
                label_parts.append(f"T:{track_id}")
            if show_global_id:
                # Display '?' if global_id is None, otherwise display the ID
                label_parts.append(f"G:{global_id if global_id is not None else '?'}")

            label = " ".join(label_parts)
            trigger_label = ""
            if triggered_rule:
                 trigger_label = f"{cam_id} -> {triggered_rule.target_cam_id}"

            if label or trigger_label:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_base = 0.8
                thickness_text = 2
                line_height = 28
                text_y_pos = y1 - 15

                num_lines = (1 if label else 0) + (1 if trigger_label else 0)
                if text_y_pos < line_height * num_lines:
                    text_y_pos = y2 + line_height

                # Draw Trigger Label (if exists) - Cyan color, larger text
                if trigger_label:
                    trigger_color = (255, 255, 0) # Cyan
                    cv2.putText(annotated_frame, trigger_label, (x1, text_y_pos),
                                font_face, font_scale_base, trigger_color, thickness_text, cv2.LINE_AA)
                    text_y_pos += line_height

                # Draw Main Label (Track/Global ID) - Use box color, larger text
                if label:
                    text_color = (255, 255, 255) if sum(color) < 384 else (0,0,0) # White on dark, black on light
                    cv2.putText(annotated_frame, label, (x1, text_y_pos),
                                font_face, font_scale_base, text_color, thickness_text, cv2.LINE_AA)


        annotated_frames[cam_id] = annotated_frame
    return annotated_frames


# display_combined_frames function remains the same as before
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
            # Create a placeholder instead of returning
            combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(combined_display, "Invalid Frame Shape", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
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