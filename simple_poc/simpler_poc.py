from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


def compute_homography(frame_width, frame_height, map_width, map_height):
    """
    Compute a homography matrix given the dimensions of the source frame and the destination map.
    The source points are manually defined and represent the region of interest on the original frame (e.g. the ground).
    The destination points define the corners of the top-down map.
    You will likely need to adapt these points for your specific scene.
    """
    # Define four points on the original frame (e.g. bottom portion where people stand)
    src_points = np.float32([
        [frame_width * 0.1, frame_height * 0.1],  # top-left of ROI
        [frame_width * 0.9, frame_height * 0.1],  # top-right of ROI
        [frame_width * 0.9, frame_height * 0.95], # bottom-right of ROI
        [frame_width * 0.1, frame_height * 0.95]  # bottom-left of ROI
    ])

    # Define the corresponding points for the destination top-down map.
    dst_points = np.float32([
        [0, 0],                      # top-left
        [map_width, 0],              # top-right
        [map_width, map_height],     # bottom-right
        [0, map_height]              # bottom-left
    ])

    # Compute Homography matrix from source to destination points.
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H, src_points, dst_points

# Initialize YOLO tracking model and video capture
model = YOLO("yolo11n.pt")
video_path = "/Users/krittinsetdhavanich/Downloads/test_yolo_track/test.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Could not open video: {video_path}")

# Create a dictionary to store tracking history for drawing trajectories
track_history = defaultdict(lambda: [])

# Read an initial frame to obtain frame dimensions
ret, frame = cap.read()
if not ret:
    cap.release()
    raise Exception("Unable to read from video source")
frame_height, frame_width = frame.shape[:2]
# Reset the video pointer to the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Define dimensions for the top-down map
map_width, map_height = 600, 800
H, src_points, dst_points = compute_homography(frame_width, frame_height, map_width, map_height)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run tracking on the current frame
    results = model.track(frame, persist=True)
    # results[0].boxes.xywh is assumed to be in [x, y, w, h] format.
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Create a blank top-down map with a white background and draw the destination polygon for reference.
    img_map = np.full((map_height, map_width, 3), 255, dtype=np.uint8)
    dst_pts_int = dst_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_map, [dst_pts_int], isClosed=True, color=(255, 0, 0), thickness=2)

    # Optionally draw the source region on the original frame for visualization.
    src_pts_int = src_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [src_pts_int], isClosed=True, color=(0, 255, 255), thickness=2)

    # Get the annotated frame from the YOLO model (draws boxes and IDs)
    annotated_frame = results[0].plot()

    for box, track_id in zip(boxes, track_ids):
        # Unpack the box: here we assume box = [x, y, w, h] where (x, y) is the center.
        # If your box represents top-left, adjust the computation accordingly.
        x, y, w, h = box
        # Compute the bottom-center of the person.
        # If (x,y) is the center, then the bottom center is at (x, y + h/2)
        bottom_center = np.array([[[float(x), float(y) + float(h) / 2]]], dtype=np.float32)

        # Update track history (in the frame coordinates)
        track_history[track_id].append((float(x), float(y) + float(h) / 2))
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        # Transform the bottom-center point to the top-down map coordinates using the homography
        pt_transformed = cv2.perspectiveTransform(bottom_center, H)
        pt_mapped = (int(pt_transformed[0][0][0]), int(pt_transformed[0][0][1]))

        # Draw a circle and label the track ID on the top-down map
        cv2.circle(img_map, pt_mapped, 5, (0, 255, 0), -1)
        cv2.putText(img_map, str(track_id), (pt_mapped[0] + 10, pt_mapped[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Optionally, draw the historical trajectories on the map
    for track_id, track in track_history.items():
        if len(track) >= 2:
            track_np = np.array(track, dtype=np.float32).reshape(-1, 1, 2)
            track_transformed = cv2.perspectiveTransform(track_np, H)
            points = track_transformed.reshape(-1, 2).astype(np.int32)
            cv2.polylines(img_map, [points], isClosed=False, color=(200, 200, 200), thickness=2)

    cv2.imshow("YOLO11 Tracking", annotated_frame)
    cv2.imshow("Map", img_map)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()