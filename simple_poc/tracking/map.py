from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def compute_homography(
    frame_width: int, frame_height: int, map_width: int, map_height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the homography matrix for perspective transformation from frame
    coordinates to map coordinates based on predefined corner points.

    Args:
        frame_width: Width of the input video frame.
        frame_height: Height of the input video frame.
        map_width: Width of the target map visualization.
        map_height: Height of the target map visualization.

    Returns:
        A tuple containing:
        - The computed homography matrix (H).
        - The source points used in the frame.
        - The destination points used on the map.
    """
    # Define source points in the frame (approximating the ground plane ROI)
    src_points = np.float32(
        [
            [frame_width * 0.1, frame_height * 0.1],
            [frame_width * 0.9, frame_height * 0.1],
            [frame_width * 0.9, frame_height * 0.95],
            [frame_width * 0.1, frame_height * 0.95],
        ]
    )

    # Define corresponding destination points on the map
    dst_points = np.float32(
        [[0, 0], [map_width, 0], [map_width, map_height], [0, map_height]]
    )

    homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return homography_matrix, src_points, dst_points


def create_map_visualization(
    map_width: int,
    map_height: int,
    dst_points: np.ndarray,
    current_boxes_xywh: List[List[float]],  # Renamed for clarity
    current_track_ids: List[int],
    track_history: Dict[int, List[Tuple[float, float]]],
    homography_matrix: np.ndarray,
    selected_track_id: Optional[int] = None,
) -> np.ndarray:
    """
    Creates a top-down map view visualizing tracked object positions and trajectories.

    Args:
        map_width: Width of the map image.
        map_height: Height of the map image.
        dst_points: Destination points defining the ROI boundary on the map.
        current_boxes_xywh: List of bounding boxes [cx, cy, w, h] for the current frame.
        current_track_ids: List of track IDs corresponding to the boxes.
        track_history: Dictionary storing recent historical positions (bottom-center) for each track ID.
        homography_matrix: The perspective transformation matrix.
        selected_track_id: Optional ID of the track to highlight.

    Returns:
        A NumPy array representing the map visualization image (BGR format).
    """
    map_image = np.full(
        (map_height, map_width, 3), 255, dtype=np.uint8
    )  # White background

    # Draw region of interest boundary on map
    cv2.polylines(
        map_image,
        [dst_points.astype(np.int32).reshape((-1, 1, 2))],
        isClosed=True,
        color=(255, 0, 0),
        thickness=2,
    )  # Blue ROI border

    # Draw current positions and trajectories
    _draw_tracked_points_on_map(
        map_image,
        current_boxes_xywh,
        current_track_ids,
        track_history,
        homography_matrix,
        selected_track_id,
    )

    _draw_trajectories_on_map(
        map_image, track_history, homography_matrix, selected_track_id
    )

    return map_image


def _draw_tracked_points_on_map(
    map_image: np.ndarray,
    current_boxes_xywh: List[List[float]],
    current_track_ids: List[int],
    track_history: Dict[int, List[Tuple[float, float]]],
    homography_matrix: np.ndarray,
    selected_track_id: Optional[int] = None,
) -> None:
    """Internal helper to draw current object positions on the map."""
    for box_xywh, track_id in zip(current_boxes_xywh, current_track_ids):
        # Skip if focusing on a specific track and this isn't it
        if selected_track_id is not None and track_id != selected_track_id:
            continue
        # Skip placeholder IDs or invalid boxes
        if track_id == -1 or len(box_xywh) != 4:
            continue

        center_x, center_y, width, height = box_xywh
        # Use bottom-center point for mapping to ground plane
        bottom_center_x = float(center_x)
        bottom_center_y = float(center_y + height / 2)

        is_selected = track_id == selected_track_id
        color = (
            (0, 0, 255) if is_selected else (0, 255, 0)
        )  # Red if selected, Green otherwise
        radius = 7 if is_selected else 5

        # Update position history (only done here for current frame point)
        # Note: tracker.py handles accumulating history across frames
        # track_history[track_id].append((bottom_center_x, bottom_center_y))
        # if len(track_history[track_id]) > 30:  # Limit history length (Handled in tracker.py)
        #     track_history[track_id].pop(0)

        # Transform point to map coordinates
        point_in_frame = np.array(
            [[[bottom_center_x, bottom_center_y]]], dtype=np.float32
        )
        point_transformed = cv2.perspectiveTransform(point_in_frame, homography_matrix)

        # Check if transformation was successful and point is valid
        if point_transformed is None or point_transformed.size == 0:
            continue  # Skip drawing if transform failed

        map_x = int(point_transformed[0][0][0])
        map_y = int(point_transformed[0][0][1])
        point_on_map = (map_x, map_y)

        # Draw circle representing the person
        cv2.circle(
            map_image, point_on_map, radius, color, thickness=-1
        )  # Filled circle

        # Add ID label next to the circle
        cv2.putText(
            map_image,
            str(track_id),
            (map_x + radius, map_y + radius // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )


def _draw_trajectories_on_map(
    map_image: np.ndarray,
    track_history: Dict[int, List[Tuple[float, float]]],
    homography_matrix: np.ndarray,
    selected_track_id: Optional[int] = None,
) -> None:
    """Internal helper to draw historical trajectories on the map."""
    for track_id, trajectory_points in track_history.items():
        # Skip if focusing on a specific track and this isn't it
        if selected_track_id is not None and track_id != selected_track_id:
            continue
        # Skip placeholder IDs or tracks with insufficient history
        if track_id == -1 or len(trajectory_points) < 2:
            continue

        is_selected = track_id == selected_track_id
        # Red if selected, light gray otherwise
        color = (0, 0, 255) if is_selected else (200, 200, 200)
        thickness = 3 if is_selected else 2

        # Transform all trajectory points (bottom-center in frame coords) to map coordinates
        trajectory_np = np.array(trajectory_points, dtype=np.float32).reshape(-1, 1, 2)
        trajectory_transformed = cv2.perspectiveTransform(
            trajectory_np, homography_matrix
        )

        # Check if transformation produced valid points
        if trajectory_transformed is None or trajectory_transformed.size == 0:
            continue  # Skip drawing if transform failed

        # Reshape for polylines and convert to integer coordinates
        map_points = trajectory_transformed.reshape(-1, 2).astype(np.int32)

        # Draw trajectory line
        cv2.polylines(
            map_image, [map_points], isClosed=False, color=color, thickness=thickness
        )
