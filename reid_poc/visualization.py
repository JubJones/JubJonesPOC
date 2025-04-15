# FILE: reid_poc/visualization.py
"""Functions for drawing annotations on frames and displaying combined views."""

import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from collections import Counter

from reid_poc.alias_types import (
    CameraID, FrameData, TrackData, TrackKey, ExitRule,
    HandoffTriggerInfo, MapCoordinate # Added MapCoordinate
)

logger = logging.getLogger(__name__)

DEFAULT_COLOR = (0, 255, 0) # Green

def generate_unique_color(global_id: int) -> Tuple[int, int, int]:
    """Generates a unique, deterministic color for a global ID, avoiding pure green."""
    # --- Function remains the same ---
    seed = int(global_id) * 3 + 5
    r = (seed * 41) % 200 + 55
    g = (seed * 17) % 200 + 55
    b = (seed * 29) % 200 + 55
    color = (b, g, r) # OpenCV uses BGR
    if color == DEFAULT_COLOR: color = (b, g - 10 if g >= 10 else g + 10, r)
    return tuple(max(0, min(255, c)) for c in color)

def draw_annotations(
    frames: Dict[CameraID, FrameData],
    processed_results: Dict[CameraID, List[TrackData]],
    handoff_triggers: List[HandoffTriggerInfo],
    frame_shapes: Dict[CameraID, Optional[Tuple[int, int]]],
    draw_bboxes: bool = True,
    show_track_id: bool = True,
    show_global_id: bool = True,
    draw_quadrants: bool = True,
    highlight_triggers: bool = True,
    show_map_coords: bool = False # NEW flag
) -> Dict[CameraID, FrameData]:
    """Draws bounding boxes, IDs, quadrant lines, handoff triggers, and optionally map coords."""
    annotated_frames: Dict[CameraID, FrameData] = {}
    default_frame_h, default_frame_w = 1080, 1920

    # Pre-calculate Global ID counts
    global_id_counts = Counter()
    all_global_ids_in_frame = [track_info.get('global_id')
                               for cam_results in processed_results.values()
                               for track_info in cam_results
                               if track_info.get('global_id') is not None]
    global_id_counts.update(all_global_ids_in_frame)

    # Determine default frame size
    first_valid_shape = next((shape for shape in frame_shapes.values() if shape), None)
    if first_valid_shape: default_frame_h, default_frame_w = first_valid_shape
    else:
        first_valid_frame = next((f for f in frames.values() if f is not None and f.size > 0), None)
        if first_valid_frame is not None: default_frame_h, default_frame_w = first_valid_frame.shape[:2]

    # Prepare Trigger Map
    trigger_map: Dict[TrackKey, ExitRule] = {
        trigger.source_track_key: trigger.rule for trigger in handoff_triggers
    }

    # Main Drawing Loop
    for cam_id, frame in frames.items():
        current_h, current_w = default_frame_h, default_frame_w
        shape_from_map = frame_shapes.get(cam_id)
        if shape_from_map: current_h, current_w = shape_from_map
        elif frame is not None: current_h, current_w = frame.shape[:2]

        # Handle missing frames
        if frame is None or frame.size == 0 or current_h <= 0 or current_w <= 0:
            placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            annotated_frames[cam_id] = placeholder
            continue
        else:
            annotated_frame = frame.copy()

        # Draw Quadrant Lines
        if draw_quadrants:
            mid_x, mid_y = current_w // 2, current_h // 2
            line_color = (100, 100, 100)
            cv2.line(annotated_frame, (mid_x, 0), (mid_x, current_h), line_color, 1)
            cv2.line(annotated_frame, (0, mid_y), (current_w, mid_y), line_color, 1)

        # Draw Tracks
        results_for_cam = processed_results.get(cam_id, [])
        for track_info in results_for_cam:
            bbox = track_info.get('bbox_xyxy')
            track_id = track_info.get('track_id')
            global_id = track_info.get('global_id')
            map_coord = track_info.get('map_coord') # Get map coordinate

            if bbox is None or track_id is None: continue
            current_track_key: TrackKey = (cam_id, track_id)

            try: # Validate coordinates
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(current_w - 1, x2), min(current_h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue
            except (ValueError, TypeError): continue

            # Determine Color and Thickness
            color = DEFAULT_COLOR
            thickness = 2
            if global_id is not None and global_id_counts.get(global_id, 0) > 1:
                color = generate_unique_color(global_id)

            triggered_rule = trigger_map.get(current_track_key) if highlight_triggers else None
            if triggered_rule: thickness = 3

            # Draw Bounding Box
            if draw_bboxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Prepare and Draw Labels
            label_parts = []
            if show_track_id: label_parts.append(f"T:{track_id}")
            if show_global_id: label_parts.append(f"G:{global_id if global_id is not None else '?'}")
            # --- NEW: Add Map Coordinates to label if available and enabled ---
            map_coord_label = ""
            if show_map_coords and map_coord is not None:
                 map_coord_label = f"Map:({map_coord[0]:.1f},{map_coord[1]:.1f})"

            main_label = " ".join(label_parts)
            trigger_label = ""
            if triggered_rule: trigger_label = f"{cam_id} -> {triggered_rule.target_cam_id}"

            if main_label or trigger_label or map_coord_label:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_base = 0.6 # Slightly smaller default
                thickness_text = 1
                line_height = 20 # Adjusted for smaller font
                text_y_pos = y1 - 5 # Start closer to the box top

                # Adjust starting Y pos if it goes off screen top
                num_lines = (1 if trigger_label else 0) + (1 if main_label else 0) + (1 if map_coord_label else 0)
                if text_y_pos < line_height * num_lines:
                    text_y_pos = y2 + line_height

                # Draw Trigger Label (if exists) - Cyan color
                if trigger_label:
                    cv2.putText(annotated_frame, trigger_label, (x1, text_y_pos),
                                font_face, font_scale_base, (255, 255, 0), thickness_text, cv2.LINE_AA)
                    text_y_pos += line_height

                # Draw Main Label (Track/Global ID) - Use box color background contrast
                if main_label:
                    text_color = (255, 255, 255) if sum(color) < 384 else (0,0,0) # White on dark, black on light
                    cv2.putText(annotated_frame, main_label, (x1, text_y_pos),
                                font_face, font_scale_base, text_color, thickness_text, cv2.LINE_AA)
                    text_y_pos += line_height

                # Draw Map Coordinate Label (if exists) - White color
                if map_coord_label:
                     cv2.putText(annotated_frame, map_coord_label, (x1, text_y_pos),
                                font_face, font_scale_base, (255, 255, 255), thickness_text, cv2.LINE_AA)
                    # text_y_pos += line_height # No need to increment Y further if it's the last line


        annotated_frames[cam_id] = annotated_frame
    return annotated_frames


def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines multiple annotated frames into a grid and displays them."""
    # --- Function remains the same ---
    valid_annotated_items = sorted([ (cid, f) for cid, f in annotated_frames.items() if f is not None and f.size > 0])
    if not valid_annotated_items:
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        valid_annotated = [item[1] for item in valid_annotated_items]
        num_cams = len(valid_annotated)
        cols = int(np.ceil(np.sqrt(num_cams))); rows = int(np.ceil(num_cams / cols))
        target_h, target_w = valid_annotated[0].shape[:2]
        if target_h <= 0 or target_w <= 0:
             combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
             cv2.putText(combined_display, "Invalid Frame Shape", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            combined_h = rows * target_h; combined_w = cols * target_w
            combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            frame_idx_display = 0
            for r in range(rows):
                 for c in range(cols):
                     if frame_idx_display < num_cams:
                         frame = valid_annotated[frame_idx_display]
                         if frame.shape[0] != target_h or frame.shape[1] != target_w:
                             try: frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                             except Exception as resize_err: logger.warning(f"Error resizing frame {frame_idx_display} for grid: {resize_err}. Using black."); frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                         try: combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = frame
                         except ValueError as slice_err: logger.error(f"Error placing frame {frame_idx_display} into grid: {slice_err}. Frame shape: {frame.shape}"); combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = 0
                         frame_idx_display += 1
            disp_h, disp_w = combined_display.shape[:2]
            if disp_w > max_width:
                 try:
                     scale = max_width / disp_w; disp_h_new = int(disp_h * scale); disp_w_new = max_width
                     if disp_h_new > 0 and disp_w_new > 0: combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
                 except Exception as final_resize_err: logger.error(f"Failed to resize final combined display: {final_resize_err}")

    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(window_name, combined_display)
    except cv2.error as cv_err: logger.debug(f"OpenCV error during imshow: {cv_err}")
    except Exception as e: logger.error(f"Unexpected error during cv2.imshow: {e}", exc_info=False)


# --- NEW (Optional): Function to display a separate BEV map ---
def display_bev_map(window_name: str, bev_map_image: Optional[FrameData]):
     """Displays the generated BEV map image in a separate window."""
     if bev_map_image is None:
         # Create a placeholder if no map generated or available
         bev_map_image = np.zeros((600, 800, 3), dtype=np.uint8)
         cv2.putText(bev_map_image, "BEV Map Not Available", (30, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

     try:
          # Ensure window exists before showing (create if not?)
          # For simplicity, assume it's created in main.py if needed
          if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) >= 0: # Check if window exists
               cv2.imshow(window_name, bev_map_image)
          # else: # Window might have been closed
          #      logger.debug(f"BEV Map window '{window_name}' not found or closed.")

     except cv2.error as cv_err:
          logger.debug(f"OpenCV error during BEV imshow '{window_name}': {cv_err}")
     except Exception as e:
          logger.error(f"Unexpected error during BEV cv2.imshow '{window_name}': {e}", exc_info=False)