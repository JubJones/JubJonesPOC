import cv2
import os
import re


def sorted_alphanumeric(data):
    """Sorts a list alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def display_video_with_bboxes(root_dir, scene, camera):
    """
    Displays a video from the MTMMC dataset with bounding boxes drawn
    based on the ground truth annotations, and shows the current image filename.

    Args:
        root_dir: The root directory of the MTMMC dataset.
        scene: The scene identifier (e.g., "s01").
        camera: The camera identifier (e.g., "c01").
    """

    video_path = os.path.join(
        root_dir, "train", "train", scene, camera, "rgb"
    )  # Path to the images
    gt_path = os.path.join(
        root_dir, "train", "train", scene, camera, "gt", "gt.txt"
    )  # Path to gt.txt

    # Check if paths exist
    if not os.path.isdir(video_path):
        raise ValueError(f"Video directory not found: {video_path}")
    if not os.path.isfile(gt_path):
        raise ValueError(f"Ground truth file not found: {gt_path}")

    # Read ground truth annotations
    annotations = {}
    with open(gt_path, "r") as f:
        for line in f:
            data = line.strip().split(",")
            frame_id = int(data[0])
            object_id = int(data[1])
            bbox_x = float(data[2])
            bbox_y = float(data[3])
            bbox_width = float(data[4])
            bbox_height = float(data[5])

            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append(
                (bbox_x, bbox_y, bbox_width, bbox_height, object_id)
            )

    # Get sorted list of image files
    image_files = sorted_alphanumeric(os.listdir(video_path))

    # Iterate through image files and display with bounding boxes
    for image_file in image_files:
        if not image_file.endswith(".jpg"):  # skip if not jpg files.
            continue

        frame_id = int(
            os.path.splitext(image_file)[0]
        )  # Extract frame ID from filename
        image_path = os.path.join(video_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # --- START ADDED CODE ---
        # Display the current image filename at the top-left corner
        cv2.putText(
            image,
            image_file,  # Text to display (the filename)
            (10, 30),  # Position (x, y) - near top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Font scale
            (255, 255, 255),  # Color in BGR (White)
            2,  # Thickness
        )
        # --- END ADDED CODE ---

        if frame_id in annotations:
            for bbox_x, bbox_y, bbox_width, bbox_height, object_id in annotations[
                frame_id
            ]:
                # Draw bounding box
                cv2.rectangle(
                    image,
                    (int(bbox_x), int(bbox_y)),
                    (int(bbox_x + bbox_width), int(bbox_y + bbox_height)),
                    (0, 255, 0),  # Green
                    2,
                )
                # Put object ID text
                cv2.putText(
                    image,
                    str(object_id),
                    (int(bbox_x), int(bbox_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),  # Green
                    2,
                )

        cv2.imshow("Video with Bounding Boxes", image)

        # Wait for a key press or a short delay to control the video speed. 25ms is good for ~30-40 FPS video.
        if cv2.waitKey(25) & 0xFF == ord("q"):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # root_directory = "D:\MTMMC"  # Windows
    root_directory = "/Volumes/HDD/MTMMC"  # Mac
    selected_scene = "s10"
    selected_camera = "c07"
    display_video_with_bboxes(root_directory, selected_scene, selected_camera)
