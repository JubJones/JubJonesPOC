import cv2
import os
import re
import numpy as np
import math
from collections import defaultdict

# --- Predefined list of distinct BGR colors ---
DISTINCT_COLORS = [
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 0),    # Dark Green (distinct from bright green)
    (0, 0, 128),    # Maroon
    (128, 128, 0),  # Teal
    (255, 192, 203), # Pink
    (0, 255, 0),    # Lime (Using this space - Careful if needed distinction)
    (75, 0, 130),   # Indigo
    (245, 245, 220), # Beige
    (210, 105, 30), # Chocolate
    (128, 128, 128),# Grey
    (255, 255, 0),   # Cyan/Aqua
]

def sorted_alphanumeric(data):
    """Sorts a list alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def load_annotations(gt_path):
    """Loads ground truth annotations from a gt.txt file."""
    annotations = defaultdict(list)
    if not os.path.isfile(gt_path):
        # print(f"Warning: Ground truth file not found: {gt_path}. Skipping annotations for this camera.")
        return annotations

    with open(gt_path, "r") as f:
        for line in f:
            try:
                data = line.strip().split(",")
                frame_id = int(data[0])
                object_id = int(data[1])
                bbox_x = float(data[2])
                bbox_y = float(data[3])
                bbox_width = float(data[4])
                bbox_height = float(data[5])
                annotations[frame_id].append(
                    (bbox_x, bbox_y, bbox_width, bbox_height, object_id)
                )
            except (IndexError, ValueError) as e:
                print(f"Warning: Skipping malformed line in {gt_path}: {line.strip()} - Error: {e}")
    return annotations

def create_montage(frames, grid_dims, target_size):
    """Creates a single image montage from a list of frames."""
    target_h, target_w = target_size
    rows, cols = grid_dims
    montage = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        if frame is None:
            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(frame, "No Frame", (int(target_w*0.1), int(target_h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            interpolation = cv2.INTER_AREA if frame.shape[0] > target_h else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

        row_idx = i // cols
        col_idx = i % cols
        y_offset = row_idx * target_h
        x_offset = col_idx * target_w
        montage[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = frame

    num_frames = len(frames)
    for i in range(num_frames, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        y_offset = row_idx * target_h
        x_offset = col_idx * target_w
        placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Empty", (int(target_w*0.2), int(target_h*0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        montage[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = placeholder

    return montage


def display_multi_video_with_bboxes(root_dir, scene, cameras):
    """
    Displays videos from multiple cameras simultaneously in a grid,
    assigning a unique persistent color to each object ID when it appears
    in multiple views simultaneously. IDs in single views are green.

    Args:
        root_dir: The root directory of the MTMMC dataset.
        scene: The scene identifier (e.g., "s01").
        cameras: A list of camera identifiers (e.g., ["c01", "c02", "c03", "c05"]).
    """
    if not cameras:
        raise ValueError("Camera list cannot be empty.")

    all_annotations = {}
    all_image_files = {}
    camera_base_paths = {}
    found_image_files = set()

    # --- Persistent color map for multi-view IDs ---
    object_color_map = {}
    next_color_index = 0
    # ---

    print("Loading data...")
    for camera in cameras:
        # (Data loading code remains the same as previous version)
        print(f"  Processing camera: {camera}")
        camera_path = os.path.join(root_dir, "train", "train", scene, camera)
        video_path = os.path.join(camera_path, "rgb")
        gt_path = os.path.join(camera_path, "gt", "gt.txt")

        if not os.path.isdir(video_path):
            print(f"  Warning: Video directory not found: {video_path}. Skipping.")
            continue

        camera_base_paths[camera] = video_path
        all_annotations[camera] = load_annotations(gt_path) # Returns empty dict if gt missing

        try:
            cam_files = [f for f in os.listdir(video_path) if f.lower().endswith(".jpg")]
            if not cam_files:
                 print(f"  Warning: No JPG files found in {video_path}. Skipping.")
                 if camera in camera_base_paths: del camera_base_paths[camera]
                 if camera in all_annotations: del all_annotations[camera]
                 continue
            all_image_files[camera] = set(cam_files) # Store as set for faster lookup
            found_image_files.update(cam_files)
            # print(f"  Found {len(cam_files)} images and {len(all_annotations[camera])} annotated frames.") # Less verbose
        except FileNotFoundError:
             print(f"  Warning: Error accessing directory {video_path}. Skipping.")
             if camera in camera_base_paths: del camera_base_paths[camera]
             if camera in all_annotations: del all_annotations[camera]
             continue
    # (End of data loading code)

    active_cameras = list(camera_base_paths.keys())
    if not active_cameras:
        raise ValueError("No valid camera data found for the specified cameras.")
    print(f"Active cameras being processed: {active_cameras}")

    if not found_image_files:
        print("No image files found across any active cameras. Exiting.")
        return
    sorted_all_found_files = sorted_alphanumeric(list(found_image_files))
    print(f"Processing {len(sorted_all_found_files)} unique frames across active cameras.")

    num_cameras = len(active_cameras)
    cols = int(math.ceil(math.sqrt(num_cameras)))
    rows = int(math.ceil(num_cameras / cols))
    grid_dims = (rows, cols)

    target_h, target_w = None, None
    for fname in sorted_all_found_files:
        for cam in active_cameras:
             if fname in all_image_files[cam]:
                 try:
                     first_img_path = os.path.join(camera_base_paths[cam], fname)
                     first_img = cv2.imread(first_img_path)
                     if first_img is not None:
                         target_h, target_w, _ = first_img.shape
                         # print(f"Determined target frame size from {cam}/{fname}: {target_w}x{target_h}")
                         break
                     # else: print(f"Warning: Could not read {first_img_path} to determine size.")
                 except Exception as e:
                     print(f"Error reading {cam}/{fname} for size: {e}")
        if target_h is not None:
            break

    if target_h is None:
        print("Error: Could not read any image to determine target frame size. Defaulting to 640x480.")
        target_h, target_w = 480, 640
    target_size = (target_h, target_w)
    print(f"Using target frame size: {target_w}x{target_h}")


    COLOR_GREEN = (0, 255, 0) # BGR
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_INFO = 0.6
    FONT_SCALE_ID = 0.7
    THICKNESS_BOX = 2
    THICKNESS_TEXT = 1

    print("Starting video display... Press 'q' to quit.")
    frame_count = 0
    for image_file in sorted_all_found_files:
        frame_count += 1
        # if frame_count % 100 == 0: print(f"  Processing frame {frame_count}/{len(sorted_all_found_files)}: {image_file}") # Less verbose

        try:
            frame_id = int(os.path.splitext(image_file)[0])
        except ValueError:
            # print(f"Warning: Could not parse frame ID from filename '{image_file}'. Skipping file.") # Less verbose
            continue

        # Identify object IDs present across multiple cameras for THIS frame_id
        ids_this_frame = defaultdict(set)
        for camera in active_cameras:
            if camera in all_annotations and frame_id in all_annotations[camera]:
                for _, _, _, _, object_id in all_annotations[camera][frame_id]:
                    ids_this_frame[object_id].add(camera)

        # Determine which IDs are multi-view *in this frame*
        multi_view_ids_in_frame = {
            obj_id for obj_id, cams in ids_this_frame.items() if len(cams) > 1
        }

        frames_to_display = []
        for camera in active_cameras:
            image = None
            image_path = os.path.join(camera_base_paths[camera], image_file)

            if image_file in all_image_files.get(camera, set()):
                image = cv2.imread(image_path)
                if image is None:
                    # print(f"Warning: Could not read image {image_path}. Showing placeholder.") # Less verbose
                    pass # Handled by create_montage
                else:
                    # Display camera ID and filename
                    text = f"{camera}: {image_file}"
                    cv2.putText(image, text, (10, 30), FONT, FONT_SCALE_INFO, (0,0,0), THICKNESS_TEXT + 1, cv2.LINE_AA)
                    cv2.putText(image, text, (10, 30), FONT, FONT_SCALE_INFO, (255,255,255), THICKNESS_TEXT, cv2.LINE_AA)

                    # Draw bounding boxes
                    if camera in all_annotations and frame_id in all_annotations[camera]:
                        for bbox_x, bbox_y, bbox_width, bbox_height, object_id in all_annotations[camera][frame_id]:
                            pt1 = (int(bbox_x), int(bbox_y))
                            pt2 = (int(bbox_x + bbox_width), int(bbox_y + bbox_height))

                            # --- Determine box color ---
                            if object_id in multi_view_ids_in_frame:
                                # This ID is seen in multiple views in THIS frame. Assign/get its persistent color.
                                if object_id not in object_color_map:
                                    # Assign the next available color
                                    object_color_map[object_id] = DISTINCT_COLORS[next_color_index % len(DISTINCT_COLORS)]
                                    next_color_index += 1
                                    # print(f"Assigned color {object_color_map[object_id]} to new multi-view ID {object_id}") # Debug
                                box_color = object_color_map[object_id]
                            else:
                                # This ID is only in one view in THIS frame.
                                box_color = COLOR_GREEN
                            # --- End Color Determination ---

                            # Draw bounding box
                            cv2.rectangle(image, pt1, pt2, box_color, THICKNESS_BOX)
                            # Put object ID text
                            id_text_pos = (pt1[0], pt1[1] - 7)
                            cv2.putText(image, str(object_id), id_text_pos, FONT, FONT_SCALE_ID, (0,0,0), THICKNESS_TEXT + 1, cv2.LINE_AA) # Outline
                            cv2.putText(image, str(object_id), id_text_pos, FONT, FONT_SCALE_ID, box_color, THICKNESS_TEXT, cv2.LINE_AA) # Text

            frames_to_display.append(image)

        # Create and display the montage
        montage = create_montage(frames_to_display, grid_dims, target_size)
        cv2.imshow("Multi-Camera View (Unique Colors for Multi-View IDs)", montage)

        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord('p'):
             print("Paused. Press 'p' again to resume.")
             while True:
                 key2 = cv2.waitKey(0) & 0xFF
                 if key2 == ord('p'):
                     print("Resuming...")
                     break
                 elif key2 == ord('q'):
                     print("Quitting...")
                     cv2.destroyAllWindows()
                     return

    cv2.destroyAllWindows()
    print("Video display finished.")


if __name__ == "__main__":
    # --- Configuration ---
    # root_directory = "D:/MTMMC"
    root_directory = "/Volumes/HDD/MTMMC"

    # Campus
    # selected_cameras = ["c01", "c02", "c03", "c15"]
    # selected_scene = "s47"

    # Factory
    selected_scene = "s10"
    selected_cameras = ["c09", "c12", "c13", "c16"]
    # --- End Configuration ---

    print(f"Starting script for scene '{selected_scene}', cameras: {selected_cameras}")
    print(f"Dataset root: {root_directory}")

    try:
        display_multi_video_with_bboxes(root_directory, selected_scene, selected_cameras)
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except FileNotFoundError as e:
         print(f"File System Error: {e}. Check dataset path and structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()