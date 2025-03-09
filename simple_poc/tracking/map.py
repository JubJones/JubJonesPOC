from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def compute_homography(frame_width: int, frame_height: int, map_width: int, map_height: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Compute homography matrix for perspective transformation"""
    src_points = np.float32([
        [frame_width * 0.1, frame_height * 0.1],
        [frame_width * 0.9, frame_height * 0.1],
        [frame_width * 0.9, frame_height * 0.95],
        [frame_width * 0.1, frame_height * 0.95]
    ])

    dst_points = np.float32([
        [0, 0],
        [map_width, 0],
        [map_width, map_height],
        [0, map_height]
    ])

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H, src_points, dst_points


def create_map_visualization(
        map_width: int,
        map_height: int,
        dst_points: np.ndarray,
        current_boxes: List[np.ndarray],
        current_track_ids: List[int],
        track_history: Dict[int, List[Tuple[float, float]]],
        homography_matrix: np.ndarray,
        selected_track_id: Optional[int] = None
) -> np.ndarray:
    """Create a top-down map view showing tracked people's positions."""
    # Create blank white image for map
    img_map = np.full((map_height, map_width, 3), 255, dtype=np.uint8)

    # Draw region of interest boundary on map
    cv2.polylines(img_map, [dst_points.astype(np.int32).reshape((-1, 1, 2))],
                  True, (255, 0, 0), 2)

    # Draw people and their movement trajectories
    draw_tracked_people_on_map(
        img_map, current_boxes, current_track_ids,
        track_history, homography_matrix, selected_track_id
    )

    draw_trajectories_on_map(
        img_map, track_history, homography_matrix, selected_track_id
    )

    return img_map


def draw_tracked_people_on_map(
        img_map: np.ndarray,
        current_boxes: List[np.ndarray],
        current_track_ids: List[int],
        track_history: Dict[int, List[Tuple[float, float]]],
        homography_matrix: np.ndarray,
        selected_track_id: Optional[int] = None
) -> None:
    """Draw current position of each tracked person on the map."""
    for box, track_id in zip(current_boxes, current_track_ids):
        # Skip if we're focusing on a selected person and this isn't them
        if selected_track_id is not None and track_id != selected_track_id:
            continue

        x, cy, w, h = box
        y = cy + h / 2  # Use bottom center point for position on ground
        is_selected = (track_id == selected_track_id)
        color = (0, 0, 255) if is_selected else (0, 255, 0)  # Red if selected

        # Update position history for this person
        track_history[track_id].append((float(x), float(y)))
        if len(track_history[track_id]) > 30:  # Limit history length
            track_history[track_id].pop(0)

        # Transform camera coordinates to map coordinates using homography
        bottom_center = np.array([[[float(x), float(y)]]], dtype=np.float32)
        pt_transformed = cv2.perspectiveTransform(bottom_center, homography_matrix)
        pt_mapped = (int(pt_transformed[0][0][0]), int(pt_transformed[0][0][1]))

        # Draw person as circle on map
        radius = 7 if is_selected else 5
        cv2.circle(img_map, pt_mapped, radius, color, -1)  # Filled circle

        # Add ID label
        cv2.putText(img_map, str(track_id), (pt_mapped[0] + 5, pt_mapped[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_trajectories_on_map(
        img_map: np.ndarray,
        track_history: Dict[int, List[Tuple[float, float]]],
        homography_matrix: np.ndarray,
        selected_track_id: Optional[int] = None
) -> None:
    """Draw movement history trails for each tracked person on the map."""
    for track_id, track in track_history.items():
        # Skip if we're focusing on a selected person and this isn't them
        if selected_track_id is not None and track_id != selected_track_id:
            continue

        # Only draw trajectories with at least 2 points
        if len(track) >= 2:
            is_selected = (track_id == selected_track_id)
            # Use distinct colors for selected vs regular trajectories
            color = (0, 0, 255) if is_selected else (200, 200, 200)  # Red if selected, light gray otherwise
            thickness = 3 if is_selected else 2

            # Transform all trajectory points to map coordinates
            track_np = np.array(track, dtype=np.float32).reshape(-1, 1, 2)
            track_transformed = cv2.perspectiveTransform(track_np, homography_matrix)
            points = track_transformed.reshape(-1, 2).astype(np.int32)

            # Draw trajectory line
            cv2.polylines(img_map, [points], False, color, thickness)