# -*- coding: utf-8 -*-
"""Functions for drawing annotations on frames and displaying combined views."""

import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from collections import Counter
from pathlib import Path

# Use relative imports
from reid_poc.alias_types import CameraID, FrameData, TrackData, TrackKey, ExitRule, HandoffTriggerInfo, GlobalID
from reid_poc.config import PipelineConfig # Import PipelineConfig for BEV settings

logger = logging.getLogger(__name__)

# Define default color (Green)
DEFAULT_COLOR = (0, 255, 0) # Green
# Define color for ID text (Red)
ID_TEXT_COLOR = (0, 0, 255) # Red in BGR

# --- Global dictionary to store BEV background image ---
_bev_background_cache: Optional[np.ndarray] = None
_bev_background_path_cache: Optional[Path] = None

def generate_unique_color(global_id: int) -> Tuple[int, int, int]:
    """Generates a unique, deterministic color for a global ID, avoiding pure green."""
    seed = int(global_id) * 3 + 5
    r = (seed * 41) % 200 + 55
    g = (seed * 17) % 200 + 55
    b = (seed * 29) % 200 + 55
    color = (b, g, r) # OpenCV uses BGR
    if color == DEFAULT_COLOR:
        color = (b, g - 10 if g >= 10 else g + 10, r)
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
    """Draws bounding boxes, IDs, quadrant lines, and handoff triggers onto frames."""
    # --- Function remains the same as previous version ---
    annotated_frames: Dict[CameraID, FrameData] = {}
    default_frame_h, default_frame_w = 1080, 1920

    global_id_counts = Counter()
    all_global_ids_in_frame = []
    for cam_results in processed_results.values():
        for track_info in cam_results:
            gid = track_info.get('global_id')
            if gid is not None:
                all_global_ids_in_frame.append(gid)
    global_id_counts.update(all_global_ids_in_frame)

    global_id_bbox_colors: Dict[GlobalID, Tuple[int, int, int]] = {}
    for gid, count in global_id_counts.items():
        if count > 1: global_id_bbox_colors[gid] = generate_unique_color(gid)
        else: global_id_bbox_colors[gid] = DEFAULT_COLOR

    first_valid_shape = next((shape for shape in frame_shapes.values() if shape), None)
    if first_valid_shape: default_frame_h, default_frame_w = first_valid_shape
    else:
        first_valid_frame = next((f for f in frames.values() if f is not None and f.size > 0), None)
        if first_valid_frame is not None: default_frame_h, default_frame_w = first_valid_frame.shape[:2]

    trigger_map: Dict[TrackKey, ExitRule] = {trigger.source_track_key: trigger.rule for trigger in handoff_triggers}

    for cam_id, frame in frames.items():
        current_h, current_w = default_frame_h, default_frame_w
        shape_from_map = frame_shapes.get(cam_id)
        if shape_from_map: current_h, current_w = shape_from_map
        elif frame is not None: current_h, current_w = frame.shape[:2]

        if frame is None or frame.size == 0 or current_h <= 0 or current_w <= 0:
            placeholder = np.zeros((default_frame_h, default_frame_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No Frame ({cam_id})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            annotated_frames[cam_id] = placeholder
            continue
        else: annotated_frame = frame.copy()

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

            bbox_color = DEFAULT_COLOR; thickness = 2
            if global_id is not None and global_id in global_id_bbox_colors: bbox_color = global_id_bbox_colors[global_id]
            triggered_rule = trigger_map.get(current_track_key) if highlight_triggers else None
            if triggered_rule: thickness = 3

            if draw_bboxes: cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, thickness)

            label_parts = []; trigger_label = ""
            if show_track_id: label_parts.append(f"T:{track_id}")
            if show_global_id: label_parts.append(f"G:{global_id if global_id is not None else '?'}")
            id_label = " ".join(label_parts)
            if triggered_rule: trigger_label = f"{cam_id} -> {triggered_rule.target_cam_id}"

            if id_label or trigger_label:
                font_face = cv2.FONT_HERSHEY_SIMPLEX; font_scale_id = 0.8; font_scale_trigger = 0.7; thickness_text = 2; text_y_pos = y1 - 10; line_height = 25
                num_lines = (1 if id_label else 0) + (1 if trigger_label else 0)
                if text_y_pos < line_height * num_lines: text_y_pos = y2 + line_height
                if trigger_label: trigger_color = (255, 255, 0); cv2.putText(annotated_frame, trigger_label, (x1 + 2, text_y_pos), font_face, font_scale_trigger, trigger_color, thickness_text, cv2.LINE_AA); text_y_pos += line_height
                if id_label: cv2.putText(annotated_frame, id_label, (x1 + 2, text_y_pos), font_face, font_scale_id, ID_TEXT_COLOR, thickness_text, cv2.LINE_AA)

        annotated_frames[cam_id] = annotated_frame
    return annotated_frames


def create_single_camera_bev_map(
    camera_id: CameraID,
    results_for_cam: List[TrackData],
    config: PipelineConfig,
) -> FrameData:
    """Creates the Bird's Eye View map visualization for a single camera."""
    # --- Function remains the same as previous version ---
    global _bev_background_cache, _bev_background_path_cache

    bev_h, bev_w = config.bev_map_display_size # Size of this individual map cell
    map_scale = config.bev_map_world_scale
    offset_x, offset_y = config.bev_map_world_origin_offset_px
    background_path = config.bev_map_background_path

    logger.debug(f"[{camera_id}] Creating BEV Map Cell. TargetSize:{bev_w}x{bev_h}, Scale:{map_scale}, Offset:({offset_x},{offset_y})")

    bev_map: Optional[np.ndarray] = None
    if background_path:
        if background_path == _bev_background_path_cache and _bev_background_cache is not None:
            bev_map = _bev_background_cache.copy(); logger.debug(f"[{camera_id}] Using cached BEV background.")
        else:
            try: img = cv2.imread(str(background_path));
            except Exception as e: logger.error(f"[{camera_id}] Error loading BEV background {background_path}: {e}")
            if img is not None:
                 logger.debug(f"[{camera_id}] Loaded BEV background (original shape: {img.shape})")
                 try: bev_map = cv2.resize(img, (bev_w, bev_h), interpolation=cv2.INTER_AREA)
                 except Exception as e_resize: logger.error(f"[{camera_id}] Failed to resize background {img.shape} to {(bev_w, bev_h)}: {e_resize}")
                 if bev_map is not None: _bev_background_cache = bev_map.copy(); _bev_background_path_cache = background_path; logger.debug(f"[{camera_id}] Resized/cached BEV background (shape: {bev_map.shape})")
                 else: logger.warning(f"[{camera_id}] Resizing BEV background failed.")
            else: logger.warning(f"[{camera_id}] Failed to read BEV background: {background_path}")
        if bev_map is None: _bev_background_cache = None; _bev_background_path_cache = None; bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8); logger.info(f"[{camera_id}] Using black background for BEV map (fallback).")
    else: logger.debug(f"[{camera_id}] No BEV background path, using black."); bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    if bev_map is None: bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8) # Final safety net

    # --- Add Camera ID Text ---
    cam_text = f"BEV - {camera_id}"
    text_color = (200, 200, 200); text_pos = (15, 30)
    cv2.putText(bev_map, cam_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA) # Adjusted scale slightly

    # --- Pre-calculate Global ID colors ---
    global_id_counts = Counter([t.get('global_id') for t in results_for_cam if t.get('global_id') is not None])
    global_id_colors: Dict[GlobalID, Tuple[int, int, int]] = {}
    for gid, count in global_id_counts.items(): global_id_colors[gid] = generate_unique_color(gid) if count > 1 else DEFAULT_COLOR # Use default color for GIDs unique *within this camera view*

    # --- Plot points ---
    points_drawn_count = 0
    for track_info in results_for_cam:
        global_id = track_info.get('global_id'); map_coords = track_info.get('map_coords'); track_id = track_info.get('track_id')
        if global_id is not None and map_coords is not None:
            map_x, map_y = map_coords; color = global_id_colors.get(global_id, DEFAULT_COLOR)
            display_x = int(map_x * map_scale + offset_x); display_y = int(map_y * map_scale + offset_y)
            if 0 <= display_x < bev_w and 0 <= display_y < bev_h:
                cv2.circle(bev_map, (display_x, display_y), radius=5, color=color, thickness=-1)
                font_face = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.5; text_color_gid = (255, 255, 255) if sum(color) < 384 else (0,0,0)
                cv2.putText(bev_map, str(global_id), (display_x + 8, display_y + 4), font_face, font_scale, text_color_gid, 1, cv2.LINE_AA)
                points_drawn_count += 1
            # else: logger.debug(f"    [{camera_id}] Skipping GID {global_id}: Coords ({display_x}, {display_y}) out of bounds ({bev_w}x{bev_h}).") # Can reduce log noise

    logger.debug(f"[{camera_id}] BEV Map Cell: Finished plotting. Drawn {points_drawn_count} points.")
    return bev_map

# --- NEW FUNCTION TO TILE BEV MAPS ---
def display_combined_bev_maps(
    window_name: str,
    bev_map_images: Dict[CameraID, FrameData],
    layout_order: List[CameraID], # e.g., ['c09', 'c12', 'c13', 'c16']
    target_cell_shape: Tuple[int, int] # (height, width) for each cell
    ):
    """Combines multiple individual BEV maps into a grid and displays them."""
    if not bev_map_images:
        logger.warning("display_combined_bev_maps called with no images.")
        # Optionally display a placeholder?
        # placeholder = np.zeros((target_cell_shape[0]*2, target_cell_shape[1]*2, 3), dtype=np.uint8)
        # cv2.putText(placeholder, "No BEV Data", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # cv2.imshow(window_name, placeholder)
        return

    target_h, target_w = target_cell_shape
    num_cams = len(layout_order) # Base grid on layout order
    if num_cams == 0: return # Nothing to display

    # Determine grid shape (assuming 2x2 for 4 cams, otherwise adapt)
    if num_cams <= 2:
        cols = num_cams
        rows = 1
    elif num_cams <= 4:
        cols = 2
        rows = 2
    else: # Fallback for > 4 cameras
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))

    combined_h = rows * target_h
    combined_w = cols * target_w
    combined_grid_image = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    logger.debug(f"Creating combined BEV grid: {cols}x{rows} cells, total size {combined_w}x{combined_h}")

    for idx, cam_id in enumerate(layout_order):
        row_idx = idx // cols
        col_idx = idx % cols

        img_to_place = bev_map_images.get(cam_id)

        if img_to_place is None or img_to_place.size == 0:
            logger.warning(f"No BEV image found for {cam_id} in combined grid.")
            # Create a placeholder for this cell
            img_to_place = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(img_to_place, f"No Data ({cam_id})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)

        # Resize if necessary
        if img_to_place.shape[0] != target_h or img_to_place.shape[1] != target_w:
            logger.warning(f"Resizing BEV map for {cam_id} from {img_to_place.shape[:2]} to {target_cell_shape} for grid.")
            try:
                img_to_place = cv2.resize(img_to_place, (target_w, target_h), interpolation=cv2.INTER_AREA)
            except Exception as resize_err:
                 logger.error(f"Error resizing BEV map for {cam_id}: {resize_err}. Using black cell.")
                 img_to_place = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Calculate placement coordinates
        start_y = row_idx * target_h
        end_y = start_y + target_h
        start_x = col_idx * target_w
        end_x = start_x + target_w

        # Place the image onto the grid
        try:
            combined_grid_image[start_y:end_y, start_x:end_x] = img_to_place
        except ValueError as slice_err:
             logger.error(f"Error placing BEV map for {cam_id} into grid (shape mismatch?): {slice_err}. Frame shape: {img_to_place.shape}, Target slice: {start_y}:{end_y}, {start_x}:{end_x}")
             # Optionally fill with black or error color
             combined_grid_image[start_y:end_y, start_x:end_x] = (0, 0, 50) # Dark red error indicator

    # --- Show the combined image ---
    try:
        # Check if window still exists before showing
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(window_name, combined_grid_image)
            logger.debug(f"Displayed combined BEV map in window '{window_name}'")
    except cv2.error as cv_err:
        logger.debug(f"OpenCV error during combined BEV imshow (window likely closed): {cv_err}")
    except Exception as e:
        logger.error(f"Unexpected error during combined BEV cv2.imshow: {e}", exc_info=False)


# display_combined_frames function remains the same as before
def display_combined_frames(window_name: str, annotated_frames: Dict[CameraID, FrameData], max_width: int):
    """Combines multiple annotated frames into a grid and displays them."""
    # --- Function remains the same ---
    valid_annotated_items = sorted([ (cid, f) for cid, f in annotated_frames.items() if f is not None and f.size > 0])
    valid_annotated = [item[1] for item in valid_annotated_items] # Get frames in sorted order

    if not valid_annotated:
        combined_display = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(combined_display, "No Valid Frames", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        num_cams = len(valid_annotated); cols = int(np.ceil(np.sqrt(num_cams))); rows = int(np.ceil(num_cams / cols))
        target_h, target_w = valid_annotated[0].shape[:2]
        if target_h <= 0 or target_w <= 0:
            logger.error("First valid frame has invalid shape for tiling.")
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
                        except ValueError as slice_err: logger.error(f"Error placing frame {frame_idx_display} into grid (shape mismatch?): {slice_err}. Frame shape: {frame.shape}"); combined_display[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = 0
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
    except cv2.error as cv_err:
        logger.debug(f"OpenCV error during imshow (window likely closed): {cv_err}")
    except Exception as e:
        logger.error(f"Unexpected error during cv2.imshow: {e}", exc_info=False)