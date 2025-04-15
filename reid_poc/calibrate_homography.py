# FILE: reid_poc/calibrate_homography.py

import logging
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# --- Assuming this script is inside reid_poc directory ---
try:
    from reid_poc.config import setup_paths_and_config, PipelineConfig
    from reid_poc.data_loader import load_dataset_info, load_frames_for_batch
    from reid_poc.alias_types import FrameData
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure this script is placed inside the 'reid_poc' directory.")
    print("Or adjust the PYTHONPATH environment variable.")
    sys.exit(1)

# --- Global variables for mouse callback and drawing ---
image_points = []       # Stores (u, v) coordinates clicked on the image
map_points = []         # Stores corresponding (X, Y) coordinates entered by user
base_frame_clean: FrameData = None # Holds the original loaded frame without any drawings
points_updated = False  # Flag to trigger redraw in main loop

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("HomographyCalibrator")

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to select image points and prompts for map points."""
    global image_points, map_points, points_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Record image point (coordinates are directly from the AUTOSIZE window)
        img_pt = (float(x), float(y))
        image_points.append(img_pt)
        logger.info(f"Added image point {len(image_points)}: ({x}, {y})")

        # 2. Prompt user for corresponding map point via console
        while True:
            try:
                map_pt_str = input(f"  -> Enter corresponding MAP coordinate (X,Y) for point {len(image_points)}: ")
                map_x_str, map_y_str = map_pt_str.strip().split(',')
                map_pt = (float(map_x_str), float(map_y_str))
                map_points.append(map_pt)
                logger.info(f"  -> Added map point {len(map_points)}: {map_pt}")
                points_updated = True # Signal the main loop to redraw
                break # Exit prompt loop if input is valid
            except ValueError:
                print("    Invalid format. Please enter as X,Y (e.g., 10.5,25.0)")
            except Exception as e:
                print(f"    An error occurred: {e}")
                # If error occurred during input, remove the last image point added
                if len(image_points) > len(map_points):
                    removed_pt = image_points.pop()
                    logger.warning(f"Removed image point {removed_pt} due to map coordinate input error.")
                break # Exit prompt loop even on error

def draw_points_and_lines(frame):
    """Draws selected points and connecting lines on the frame."""
    global image_points
    point_color = (0, 0, 255) # Red points
    line_color = (0, 255, 255) # Yellow lines
    text_color = (0, 0, 255) # Red text

    # Draw points and numbers
    for i, pt in enumerate(image_points):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 5, point_color, -1)
        cv2.putText(frame, str(i + 1), (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Draw lines connecting points
    if len(image_points) >= 2:
        pts_np = np.array(image_points, dtype=np.int32)
        # Draw lines between consecutive points
        for i in range(len(pts_np) - 1):
            cv2.line(frame, tuple(pts_np[i]), tuple(pts_np[i+1]), line_color, 1, cv2.LINE_AA)
        # If exactly 4 points, draw the closing line for the quadrilateral
        if len(image_points) == 4:
             cv2.line(frame, tuple(pts_np[3]), tuple(pts_np[0]), line_color, 1, cv2.LINE_AA)

    return frame


def main():
    """Main function to handle point selection and saving."""
    global image_points, map_points, base_frame_clean, points_updated

    parser = argparse.ArgumentParser(description="Manual Homography Point Selection Tool")
    parser.add_argument("camera_id", type=str, help="The Camera ID to calibrate (e.g., 'c09').")
    parser.add_argument("-s", "--scene", type=str, default=None, help="Override the default scene selected in config.py.")
    args = parser.parse_args()

    camera_to_calibrate = args.camera_id
    logger.info(f"--- Starting Homography Calibration for Camera: {camera_to_calibrate} ---")

    try:
        # 1. Load Configuration
        config = setup_paths_and_config()
        original_scene = config.selected_scene
        if args.scene and args.scene != original_scene:
            logger.warning(f"Overriding scene from config. Using specified scene: {args.scene}")
            # Update config object AFTER initial setup maybe needed if paths depend heavily on it
            # For simplicity, assume data_loader can handle the correct scene path finding
            config.selected_scene = args.scene
            # Need to re-validate the selected camera for the NEW scene
            # Re-running parts of config setup or explicitly checking path existence is safer
            # Basic check:
            scene_path_exists = any(
                (config.dataset_base_path / p / config.selected_scene).is_dir()
                for p in ["train/train", "train", "."]
            )
            if not scene_path_exists:
                 raise FileNotFoundError(f"Overridden scene '{config.selected_scene}' not found in expected locations.")
            logger.info("Re-validating cameras for the overridden scene implicitly via data_loader.")

        # 2. Load Dataset Info
        # Create a temporary config focused only on the target camera and scene
        temp_config_for_loader = PipelineConfig(selected_scene=config.selected_scene,
                                               dataset_base_path=config.dataset_base_path)
        temp_config_for_loader.selected_cameras = [camera_to_calibrate]

        camera_dirs, image_filenames = load_dataset_info(temp_config_for_loader)

        if camera_to_calibrate not in camera_dirs:
             raise FileNotFoundError(f"Could not find valid image directory for camera '{camera_to_calibrate}' in scene '{config.selected_scene}'.")
        if not image_filenames:
             raise FileNotFoundError(f"No image files found for scene '{config.selected_scene}'. Cannot load frame.")

        # 3. Load the First Frame
        logger.info(f"Loading first frame: {image_filenames[0]}")
        frame_dict = load_frames_for_batch({camera_to_calibrate: camera_dirs[camera_to_calibrate]}, image_filenames[0])
        base_frame_loaded = frame_dict.get(camera_to_calibrate)

        if base_frame_loaded is None or base_frame_loaded.size == 0:
            logger.error(f"Failed to load the first frame for camera '{camera_to_calibrate}'. Exiting.")
            return

        base_frame_clean = base_frame_loaded.copy() # Keep an unmodified copy

        # --- Use WINDOW_AUTOSIZE to prevent resizing and ensure 1:1 coordinate mapping ---
        window_name = f"Calibrate Homography - Cam: {camera_to_calibrate} (Scene: {config.selected_scene})"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n--- Instructions ---")
        print(f"1. Click points ON THE GROUND in the '{window_name}' window.")
        print("2. For each click, enter the corresponding MAP coordinates (X,Y) in the console.")
        print("   (Ensure these map coordinates match your defined BEV map system).")
        print("3. Select at least 4 non-collinear points.")
        print("4. Press 's' to SAVE the points and quit.")
        print("5. Press 'q' to QUIT WITHOUT saving.")
        print("--- Starting Point Selection ---")

        # 4. Main Loop for Point Selection
        while True:
            # Start with a clean frame copy each time
            frame_to_display = base_frame_clean.copy()
            # Draw current points and lines
            frame_to_display = draw_points_and_lines(frame_to_display)
            points_updated = False # Reset flag after drawing

            # Display instructions and point count on the frame
            h, w = frame_to_display.shape[:2]
            text_color = (255, 255, 255) # White text
            bg_color = (0, 0, 0) # Black background for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thick = 1

            line1 = f"Points selected: {len(image_points)} (Need >= 4)"
            line2 = "Click ground points -> Enter map coords (X,Y) in console"
            line3 = "Press 's' to save & quit | 'q' to quit"

            cv2.putText(frame_to_display, line1, (10, h - 55), font, scale, bg_color, thick + 1, cv2.LINE_AA) # Background
            cv2.putText(frame_to_display, line1, (10, h - 55), font, scale, text_color, thick, cv2.LINE_AA) # Foreground
            cv2.putText(frame_to_display, line2, (10, h - 35), font, scale, bg_color, thick + 1, cv2.LINE_AA)
            cv2.putText(frame_to_display, line2, (10, h - 35), font, scale, text_color, thick, cv2.LINE_AA)
            cv2.putText(frame_to_display, line3, (10, h - 15), font, scale, bg_color, thick + 1, cv2.LINE_AA)
            cv2.putText(frame_to_display, line3, (10, h - 15), font, scale, text_color, thick, cv2.LINE_AA)


            # Display the frame
            cv2.imshow(window_name, frame_to_display)
            key = cv2.waitKey(20) & 0xFF # Increased wait time slightly

            if key == ord('q'):
                logger.info("Quit key pressed. Exiting without saving.")
                break
            elif key == ord('s'):
                if len(image_points) < 4:
                    logger.warning(f"Cannot save yet. Need at least 4 points, only have {len(image_points)}.")
                    print(f"\n*** Cannot save yet. Need at least 4 points, only have {len(image_points)}. ***")
                elif len(image_points) != len(map_points):
                     logger.error("Mismatch between image points ({len(image_points)}) and map points ({len(map_points)}) count! Cannot save.")
                     print("\n*** Error: Mismatch between image and map point counts! Cannot save. Check console for errors during input. ***")
                else:
                    logger.info(f"Save key pressed. Saving {len(image_points)} point pairs.")
                    img_pts_np = np.array(image_points, dtype=np.float32)
                    map_pts_np = np.array(map_points, dtype=np.float32)

                    output_dir = Path(".")
                    output_dir.mkdir(exist_ok=True)
                    # Use the potentially overridden scene name in the filename
                    filename = output_dir / f"homography_points_{camera_to_calibrate}_scene_{config.selected_scene}.npz"

                    try:
                        np.savez(str(filename), image_points=img_pts_np, map_points=map_pts_np)
                        logger.info(f"Successfully saved point data to: {filename}")
                        print(f"\nSuccessfully saved point data to: {filename}")
                        break # Exit after successful save
                    except Exception as e:
                        logger.error(f"Failed to save point data: {e}", exc_info=True)
                        print(f"\n*** Error saving file: {e} ***")

    except FileNotFoundError as e:
        logger.critical(f"Setup Error: A required file or directory was not found: {e}", exc_info=True)
        print(f"\n*** Setup Error: {e} ***")
    except ValueError as e:
         logger.critical(f"Configuration or Input Error: {e}", exc_info=True)
         print(f"\n*** Config/Input Error: {e} ***")
    except ImportError as e:
         # Already handled at the top, but catch again just in case
         logger.critical(f"Import Error: {e}")
         print(f"\n*** Import Error: {e} ***")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\n*** An unexpected error occurred: {e} ***")
    finally:
        cv2.destroyAllWindows()
        # Add a small delay to ensure window closes cleanly on some systems
        for _ in range(5):
             cv2.waitKey(1)
        logger.info("--- Calibration Tool Finished ---")

if __name__ == "__main__":
    main()