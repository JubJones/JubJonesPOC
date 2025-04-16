# FILE: reid_poc/visualization.py
# -*- coding: utf-8 -*-
"""Functions for drawing annotations on frames and displaying combined views."""

import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
# Removed Counter import as it's no longer needed here
from pathlib import Path

# Use relative imports
from reid_poc.alias_types import CameraID, FrameData, TrackData, TrackKey, ExitRule, HandoffTriggerInfo, GlobalID
from reid_poc.config import PipelineConfig # Import PipelineConfig for BEV settings

logger = logging.getLogger(__name__)

# Define default color (Green) - Used for all dots now
DEFAULT_COLOR = (0, 255, 0) # Green
# Define color for Track ID text (Red) - BBox label
ID_TEXT_COLOR = (0, 0, 255) # Red in BGR
# Define fallback color for Global ID text if not found in map
DEFAULT_GID_TEXT_COLOR = (255, 255, 255) # White

def generate_unique_color(global_id: int) -> Tuple[int, int, int]:
    """Generates a unique, deterministic color for a global ID, avoiding pure green."""
    # --- Function remains the same ---
    seed = int(global_id) * 3 + 5
    # Make colors generally lighter/brighter for better visibility on dark maps/backgrounds
    r = (seed * 41) % 155 + 100 # Range 100-254
    g = (seed * 17) % 155 + 100 # Range 100-254
    b = (seed * 29) % 155 + 100 # Range 100-254
    color = (b, g, r) # OpenCV uses BGR
    # Ensure not too close to pure green or pure white
    if (abs(g - 255) < 30 and abs(r) < 30 and abs(b) < 30): # Avoid green screen color
         g = 200
    if (r > 230 and g > 230 and b > 230): # Avoid near white
        r -= 30
        g -= 30
    color = tuple(max(0, min(255, c)) for c in color)
    # Ensure it's not exactly the default green
    if color == DEFAULT_COLOR:
        color = (b, g - 10 if g >= 10 else g + 10, r)
        color = tuple(max(0, min(255, c)) for c in color)

    return color


# Accepts pre-calculated global_id_color_map
def draw_annotations(
    frames: Dict[CameraID, FrameData],
    processed_results: Dict[CameraID, List[TrackData]],
    handoff_triggers: List[HandoffTriggerInfo],
    frame_shapes: Dict[CameraID, Optional[Tuple[int, int]]], # Use shapes from config
    global_id_color_map: Dict[GlobalID, Tuple[int, int, int]], # Passed parameter
    draw_bboxes: bool = True,
    show_track_id: bool = True,
    show_global_id: bool = True,
    draw_quadrants: bool = True,
    highlight_triggers: bool = True
) -> Dict[CameraID, FrameData]:
    """Draws bounding boxes, IDs, quadrant lines, and handoff triggers onto frames."""
    # This function remains unchanged from the previous correct version,
    # still using the passed map for bbox colors.
    annotated_frames: Dict[CameraID, FrameData] = {}
    default_frame_h, default_frame_w = 1080, 1920

    first_valid_shape = next((shape for shape in frame_shapes.values() if shape), None)
    if first_valid_shape: default_frame_h, default_frame_w = first_valid_shape
    else:
        first_valid_frame = next((f for f in frames.values() if f is not None and f.size > 0), None)
        if first_valid_frame is not None: default_frame_h, default_frame_w = first_valid_frame.shape[:2]

    trigger_map: Dict[TrackKey, ExitRule] = {trigger.source_track_key: trigger.rule for trigger in handoff_triggers}

    for cam_id, frame in frames.items():
        current_h, current_w = default_frame_h, default_frame_w
        shape_from_config = frame_shapes.get(cam_id)
        if shape_from_config: current_h, current_w = shape_from_config
        elif frame is not None: current_h, current_w = frame.shape[:2]

        if frame is None or frame.size == 0 or current_h <= 0 or current_w <= 0:
            placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            annotated_frames[cam_id] = placeholder
            continue
        else:
            annotated_frame = frame.copy()

        if draw_quadrants:
            mid_x, mid_y = current_w // 2, current_h // 2
            line_color = (100, 100, 100)
            cv2.line(annotated_frame, (mid_x, 0), (mid_x, current_h), line_color, 1)
            cv2.line(annotated_frame, (0, mid_y), (current_w, mid_y), line_color, 1)

        results_for_cam = processed_results.get(cam_id, [])
        for track_info in results_for_cam:
            bbox = track_info.get('bbox_xyxy'); track_id = track_info.get('track_id'); global_id = track_info.get('global_id')
            if bbox is None or track_id is None: continue
            current_track_key: TrackKey = (cam_id, track_id)

            try:
                x1, y1, x2, y2 = map(int, bbox); x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(current_w - 1, x2), min(current_h - 1, y2)
                if x1 >= x2 or y1 >= y2: continue
            except (ValueError, TypeError): logger.warning(f"[{cam_id}] Invalid bbox coordinates for T:{track_id}: {bbox}"); continue

            bbox_color = global_id_color_map.get(global_id, DEFAULT_COLOR) if global_id is not None else DEFAULT_COLOR
            thickness = 2
            triggered_rule = trigger_map.get(current_track_key) if highlight_triggers else None
            if triggered_rule: thickness = 3

            if draw_bboxes: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, thickness)

            label_parts = []; trigger_label = ""
            if show_track_id: label_parts.append(f"T:{track_id}")
            if show_global_id: label_parts.append(f"G:{global_id if global_id is not None else '?'}")
            id_label = " ".join(label_parts)
            if triggered_rule: trigger_label = f"{cam_id} -> {triggered_rule.target_cam_id}"

            if id_label or trigger_label:
                font_face = cv2.FONT_HERSHEY_SIMPLEX; font_scale_id = 0.8; font_scale_trigger = 0.7; thickness_text = 2;
                text_y_pos = y1 - 10; line_height = 25
                num_lines = (1 if id_label else 0) + (1 if trigger_label else 0)
                if text_y_pos < line_height * num_lines: text_y_pos = y2 + line_height

                if trigger_label:
                    trigger_color = (255, 255, 0) # Cyan
                    cv2.putText(annotated_frame, trigger_label, (x1 + 2, text_y_pos), font_face, font_scale_trigger, trigger_color, thickness_text, cv2.LINE_AA)
                    text_y_pos += line_height

                if id_label:
                    cv2.putText(annotated_frame, id_label, (x1 + 2, text_y_pos), font_face, font_scale_id, ID_TEXT_COLOR, thickness_text, cv2.LINE_AA)

        annotated_frames[cam_id] = annotated_frame
    return annotated_frames


# --- MODIFIED: Draws green dots, but uses unique color for GID text ---
def create_single_camera_bev_map(
    camera_id: CameraID,
    results_for_cam: List[TrackData],
    config: PipelineConfig,
    preloaded_background: Optional[np.ndarray],
    global_id_color_map: Dict[GlobalID, Tuple[int, int, int]] # Passed parameter
) -> FrameData:
    """
    Creates the Bird's Eye View map visualization for a single camera.
    Uses preloaded background if provided, otherwise uses black.
    Draws GREEN dots for all points, but uses the unique color from
    global_id_color_map for the Global ID text label.
    """
    bev_h, bev_w = config.bev_map_display_size
    map_scale = config.bev_map_world_scale
    offset_x, offset_y = config.bev_map_world_origin_offset_px

    logger.debug(f"[{camera_id}] Creating BEV Map Cell. TargetSize:{bev_w}x{bev_h}, Scale:{map_scale}, Offset:({offset_x},{offset_y})")

    # --- Background handling (unchanged) ---
    if preloaded_background is not None:
        bev_map = preloaded_background.copy()
        logger.debug(f"[{camera_id}] Using preloaded BEV background (shape: {bev_map.shape}).")
        if bev_map.shape[0] != bev_h or bev_map.shape[1] != bev_w:
            logger.warning(f"[{camera_id}] Preloaded BEV background shape {bev_map.shape[:2]} doesn't match target {config.bev_map_display_size}. Resizing again.")
            try:
                bev_map = cv2.resize(bev_map, (bev_w, bev_h), interpolation=cv2.INTER_AREA)
            except Exception as resize_err:
                logger.error(f"[{camera_id}] Failed to resize preloaded BEV background: {resize_err}. Using black fallback.")
                bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    else:
        logger.debug(f"[{camera_id}] No preloaded BEV background, using black.")
        bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # --- Add Camera ID Text (unchanged) ---
    cam_text = f"BEV - {camera_id}"
    text_color_cam_id = (200, 200, 200); text_pos = (15, 30)
    cv2.putText(bev_map, cam_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color_cam_id, 2, cv2.LINE_AA)

    # --- Plot points ---
    points_drawn_count = 0
    for track_info in results_for_cam:
        global_id = track_info.get('global_id'); map_coords = track_info.get('map_coords')
        if map_coords is not None: # Plot if map coords exist
            map_x, map_y = map_coords
            display_x = int(map_x * map_scale + offset_x); display_y = int(map_y * map_scale + offset_y)

            if 0 <= display_x < bev_w and 0 <= display_y < bev_h:
                # --- MODIFIED: Always draw dot with DEFAULT_COLOR ---
                cv2.circle(bev_map, (display_x, display_y), radius=5, color=DEFAULT_COLOR, thickness=-1)

                # Draw Global ID Text if available, using the unique color
                if global_id is not None:
                    font_face = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.5;
                    # --- MODIFIED: Get unique color for text, fallback to white ---
                    text_color_gid = global_id_color_map.get(global_id, DEFAULT_GID_TEXT_COLOR)
                    cv2.putText(bev_map, str(global_id), (display_x + 8, display_y + 4), font_face, font_scale, text_color_gid, 1, cv2.LINE_AA)

                points_drawn_count += 1

    logger.debug(f"[{camera_id}] BEV Map Cell: Finished plotting. Drawn {points_drawn_count} points.")
    return bev_map


# --- display_combined_bev_maps function remains the same ---
def display_combined_bev_maps(
    window_name: str,
    bev_map_images: Dict[CameraID, FrameData],
    layout_order: List[CameraID], # e.g., ['c09', 'c12', 'c13', 'c16']
    target_cell_shape: Tuple[int, int] # (height, width) for each cell
    ):
    """Combines multiple individual BEV maps into a grid and displays them."""
    if not bev_map_images:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                placeholder = np.zeros((target_cell_shape[0]*2, target_cell_shape[1]*2, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No BEV Data", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow(window_name, placeholder)
        except: pass # Ignore errors if window closed
        return

    target_h, target_w = target_cell_shape
    num_cams = len(layout_order)
    if num_cams == 0: return

    # Determine grid shape
    if num_cams <= 2: cols, rows = num_cams, 1
    elif num_cams <= 4: cols, rows = 2, 2
    elif num_cams <= 6: cols, rows = 3, 2
    elif num_cams <= 9: cols, rows = 3, 3
    else: # Fallback for more
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))

    combined_h = rows * target_h
    combined_w = cols * target_w
    combined_grid_image = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    logger.debug(f"Creating combined BEV grid: {cols}x{rows} cells, total size {combined_w}x{combined_h}")

    for idx, cam_id in enumerate(layout_order):
        row_idx = idx // cols; col_idx = idx % cols
        img_to_place = bev_map_images.get(cam_id)

        if img_to_place is None or img_to_place.size == 0:
            img_to_place = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(img_to_place, f"No Data ({cam_id})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)

        # Ensure correct size
        if img_to_place.shape[0] != target_h or img_to_place.shape[1] != target_w:
            logger.warning(f"Resizing BEV map for {cam_id} from {img_to_place.shape[:2]} to {target_cell_shape} for grid display.")
            try: img_to_place = cv2.resize(img_to_place, (target_w, target_h), interpolation=cv2.INTER_AREA)
            except Exception as resize_err:
                logger.error(f"Error resizing BEV map for {cam_id} display: {resize_err}. Using black cell.")
                img_to_place = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        start_y, end_y = row_idx * target_h, (row_idx + 1) * target_h
        start_x, end_x = col_idx * target_w, (col_idx + 1) * target_w

        try: combined_grid_image[start_y:end_y, start_x:end_x] = img_to_place
        except ValueError as slice_err: logger.error(f"Error placing BEV map {cam_id} into grid: {slice_err}. Frame shape: {img_to_place.shape}")

    # Show combined image
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(window_name, combined_grid_image)
            logger.debug(f"Displayed combined BEV map in window '{window_name}'")
    except cv2.error as cv_err: logger.debug(f"OpenCV error during combined BEV imshow (window likely closed): {cv_err}")
    except Exception as e: logger.error(f"Unexpected error during combined BEV cv2.imshow: {e}", exc_info=False)


# --- display_combined_frames function remains the same ---
def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines multiple annotated frames into a grid and displays them."""
    # Sort frames by camera ID for consistent display order
    valid_annotated_items = sorted([ (cid, f) for cid, f in annotated_frames.items() if f is not None and f.size > 0])
    valid_annotated = [item[1] for item in valid_annotated_items]

    if not valid_annotated:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, placeholder)
        except: pass
        return

    num_cams = len(valid_annotated)
    # Determine grid shape (same logic as BEV grid)
    if num_cams <= 2: cols, rows = num_cams, 1
    elif num_cams <= 4: cols, rows = 2, 2
    elif num_cams <= 6: cols, rows = 3, 2
    elif num_cams <= 9: cols, rows = 3, 3
    else: cols = int(np.ceil(np.sqrt(num_cams))); rows = int(np.ceil(num_cams / cols))

    # Use shape of the first valid frame as target, assume others are the same
    target_h, target_w = valid_annotated[0].shape[:2]
    if target_h <= 0 or target_w <= 0: # Safety check
        logger.error("First valid frame has invalid shape for tiling.")
        return # Cannot create grid

    combined_h = rows * target_h; combined_w = cols * target_w
    combined_display = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

    frame_idx_display = 0
    for r in range(rows):
        for c in range(cols):
            if frame_idx_display < num_cams:
                frame = valid_annotated[frame_idx_display]
                # Ensure consistent shape (resize if needed, though ideally they match)
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    try: frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    except Exception as resize_err: frame = np.zeros((target_h, target_w, 3), dtype=np.uint8) # Black on error

                try: combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = frame
                except ValueError as slice_err: logger.error(f"Error placing frame {frame_idx_display} into grid: {slice_err}"); combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = 0 # Black on error
                frame_idx_display += 1

    # Resize final combined image if it exceeds max_width
    disp_h, disp_w = combined_display.shape[:2]
    if disp_w > max_width:
        try:
            scale = max_width / disp_w; disp_h_new = int(disp_h * scale); disp_w_new = max_width
            if disp_h_new > 0 and disp_w_new > 0:
                combined_display = cv2.resize(combined_display, (disp_w_new, disp_h_new), interpolation=cv2.INTER_AREA)
        except Exception as final_resize_err: logger.error(f"Failed to resize final combined display: {final_resize_err}")

    # Show final image
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(window_name, combined_display)
    except cv2.error as cv_err: logger.debug(f"OpenCV error during camera view imshow (window likely closed): {cv_err}")
    except Exception as e: logger.error(f"Unexpected error during camera view cv2.imshow: {e}", exc_info=False)