# FILE: reid_poc/main.py
# -*- coding: utf-8 -*-
"""Main execution script for the Multi-Camera Tracking & Re-Identification Pipeline with Handoff."""

import logging
import time
from collections import defaultdict
from typing import Optional, Dict, Tuple

import cv2
import torch
import numpy as np

# --- Local Modules ---
from reid_poc.config import setup_paths_and_config, PipelineConfig
from reid_poc.alias_types import ProcessedBatchResult, CameraID, FrameData, MapCoordinate # Added MapCoordinate
from reid_poc.data_loader import load_dataset_info, load_frames_for_batch
from reid_poc.models import load_detector, load_reid_model
from reid_poc.tracking import initialize_trackers
from reid_poc.pipeline import MultiCameraPipeline
# Updated import for visualization
from reid_poc.visualization import draw_annotations, display_combined_frames, display_bev_map

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Constants ---
BEV_WINDOW_NAME = "BEV Map View" # Name for the optional BEV display window


def main():
    """Main function to set up and run the pipeline."""
    pipeline_instance: Optional[MultiCameraPipeline] = None
    last_batch_result: Optional[ProcessedBatchResult] = None
    config: Optional[PipelineConfig] = None
    is_paused: bool = False

    detector = None; reid_model = None; trackers = None

    try:
        # 1. Configuration and Setup (Loads homographies now)
        config = setup_paths_and_config()

        if config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("DEBUG logging enabled.")

        # 2. Load Dataset Information
        camera_dirs, image_filenames = load_dataset_info(config)
        logger.info(f"Processing scene '{config.selected_scene}' with {len(image_filenames)} frames.")

        # 3. Load Models
        detector, detector_transforms = load_detector(config.device)
        reid_model = load_reid_model(config.reid_model_weights, config.device)

        # 4. Initialize Trackers
        trackers = initialize_trackers(
            config.selected_cameras,
            config.tracker_type,
            config.tracker_config_path,
            config.device
        )

        # 5. Initialize Pipeline (Passes config which includes homographies)
        pipeline_instance = MultiCameraPipeline(
            config=config,
            detector=detector,
            detector_transforms=detector_transforms,
            reid_model=reid_model,
            trackers=trackers
        )

        # 6. Setup Display Windows
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        logger.info(f"Display window '{config.window_name}' created. Press 'p' to pause/resume, 'q' to quit.")
        # --- Create BEV window if needed (optional, based on future implementation) ---
        # if config.generate_bev_visualization:
        #      cv2.namedWindow(BEV_WINDOW_NAME, cv2.WINDOW_AUTOSIZE) # Autosize might be better for map
        #      logger.info(f"BEV display window '{BEV_WINDOW_NAME}' created.")

        # 7. Frame Processing Loop
        logger.info("--- Starting Frame Processing Loop ---")
        total_frames_loaded = 0
        total_frames_processed = 0
        frame_idx = 0
        loop_start_time = time.perf_counter()

        frame_shapes_for_viz: Dict[CameraID, Optional[Tuple[int, int]]] = {
            cam_id: cfg.frame_shape for cam_id, cfg in config.cameras_handoff_config.items()
        }

        while frame_idx < len(image_filenames):
            iter_start_time = time.perf_counter()

            # Load Frames
            current_filename = image_filenames[frame_idx]
            current_frames = load_frames_for_batch(camera_dirs, current_filename)

            # Process frame *only if not paused*
            if not is_paused:
                if not any(f is not None and f.size > 0 for f in current_frames.values()):
                    if frame_idx < 10: logger.warning(f"Frame {frame_idx}: No valid images loaded for '{current_filename}'. Skipping.")
                    frame_idx += 1
                    continue

                total_frames_loaded += 1
                process_this_frame = (frame_idx % config.frame_skip_rate == 0)
                current_batch_timings = defaultdict(float)
                processed_bev_coords: Dict[GlobalID, MapCoordinate] = {} # Store BEV coords from result

                if process_this_frame:
                    if pipeline_instance is not None:
                        # pipeline processes and returns BEV coordinates
                        batch_result = pipeline_instance.process_frame_batch_full(current_frames, frame_idx)
                        last_batch_result = batch_result # Store full result for drawing
                        current_batch_timings = batch_result.timings
                        processed_bev_coords = batch_result.bev_coordinates # Get BEV coords
                        total_frames_processed += 1
                    else:
                        logger.error("Pipeline not initialized! Cannot process frame. Exiting.")
                        break
                else: # Skipped frame processing
                     current_batch_timings['skipped_frame_overhead'] = (time.perf_counter() - iter_start_time)
                     # Use BEV coords from the *last processed* frame if skipping
                     if last_batch_result:
                          processed_bev_coords = last_batch_result.bev_coordinates

                # --- Logging and Timing ---
                iter_end_time = time.perf_counter()
                frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000
                current_loop_duration = iter_end_time - loop_start_time
                avg_display_fps = total_frames_loaded / current_loop_duration if current_loop_duration > 0 else 0
                avg_processing_fps = total_frames_processed / current_loop_duration if current_loop_duration > 0 else 0

                # Periodic Logging
                if frame_idx < 10 or frame_idx % 50 == 0 or not process_this_frame:
                     track_count = 0; trigger_count = 0; bev_count = len(processed_bev_coords)
                     if last_batch_result:
                         track_count = sum(len(tracks) for tracks in last_batch_result.results_per_camera.values())
                         trigger_count = len(last_batch_result.handoff_triggers)

                     pipeline_timing_str = ""
                     if process_this_frame and current_batch_timings:
                          stages = ['preprocess', 'detection', 'postproc', 'tracking', 'handoff', 'feature', 'reid', 'bev', 'total'] # Added bev
                          pipe_times = {k: v for k, v in current_batch_timings.items() if any(k.startswith(s) for s in stages)}
                          pipeline_timing_str = " | Pipe(ms): " + " ".join([f"{k[:4]}={v*1000:.1f}" for k, v in sorted(pipe_times.items()) if v > 0.0001])

                     status = "PROC" if process_this_frame else "SKIP"
                     logger.info(
                         f"Frame {frame_idx:<4} [{status}] | Iter:{frame_proc_time_ms:>6.1f}ms "
                         f"| AvgDisp:{avg_display_fps:5.1f} AvgProc:{avg_processing_fps:5.1f} "
                         f"| Trk:{track_count:<3} Trig:{trigger_count:<2} BEV:{bev_count:<3}{pipeline_timing_str}" # Added BEV count
                     )
                     # --- Optional: Print calculated BEV coordinates ---
                     if process_this_frame and processed_bev_coords:
                          coord_str = ", ".join([f"G{gid}:({c[0]:.1f},{c[1]:.1f})" for gid, c in sorted(processed_bev_coords.items())])
                          logger.debug(f"  BEV Coords: {coord_str}")


                # Advance frame index only if not paused
                frame_idx += 1

            # --- Annotate and Display ---
            display_frames = current_frames
            results_to_draw = {}
            triggers_to_draw = []
            # bev_map_to_display = None # Initialize BEV map for display

            if last_batch_result:
                results_to_draw = last_batch_result.results_per_camera
                triggers_to_draw = last_batch_result.handoff_triggers
                # If backend generates map image, get it here:
                # bev_map_to_display = getattr(pipeline_instance, 'current_bev_map_image', None)


            # Draw annotations on camera views (optionally including map coords)
            annotated_frames = draw_annotations(
                display_frames,
                results_to_draw,
                triggers_to_draw,
                frame_shapes_for_viz,
                draw_bboxes=config.draw_bounding_boxes,
                show_track_id=config.show_track_id,
                show_global_id=config.show_global_id,
                draw_quadrants=config.draw_quadrant_lines,
                highlight_triggers=config.highlight_handoff_triggers,
                show_map_coords=config.show_map_coordinates_on_cam # Use new config flag
            )

            # Display combined camera views
            display_combined_frames(config.window_name, annotated_frames, config.max_display_width)

            # --- Display BEV Map (if generated) ---
            # if config.generate_bev_visualization:
            #      display_bev_map(BEV_WINDOW_NAME, bev_map_to_display)


            # User Input Handling (Pause/Quit)
            wait_duration = config.display_wait_ms if not is_paused else 50
            key = cv2.waitKey(wait_duration) & 0xFF

            if key == ord('q'):
                logger.info("Quit key (q) pressed. Exiting loop.")
                break
            elif key == ord('p'):
                is_paused = not is_paused
                logger.info("<<<< PAUSED >>>> Press 'p' to resume." if is_paused else ">>>> RESUMED >>>>")

            # Check if display window was closed
            try:
                if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Display window was closed. Exiting loop.")
                    break
                # Optional: Check BEV window too if it exists
                # if config.generate_bev_visualization and cv2.getWindowProperty(BEV_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                #      logger.info("BEV Map window was closed. Exiting loop.")
                #      break
            except cv2.error:
                logger.info("Display window seems closed or unavailable. Exiting loop.")
                break


        # End of Loop
        loop_end_time = time.perf_counter()
        total_time = loop_end_time - loop_start_time
        logger.info(f"--- Frame Processing Loop Finished (Processed {total_frames_processed} frames) ---")
        if total_frames_loaded > 0 and total_time > 0.01:
            final_avg_display_fps = total_frames_loaded / total_time
            final_avg_processing_fps = total_frames_processed / total_time if total_frames_processed > 0 else 0
            logger.info(f"Total loop time: {total_time:.2f}s.")
            logger.info(f"Overall Avg Display FPS: {final_avg_display_fps:.2f}")
            logger.info(f"Overall Avg Processing FPS: {final_avg_processing_fps:.2f}")
        else:
            logger.info("Not enough frames or time elapsed for meaningful average FPS.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError, ImportError) as e:
        logger.critical(f"Pipeline Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("--- Cleaning up resources ---")
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)

        del pipeline_instance, detector, reid_model, trackers, last_batch_result, config

        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache..."); torch.cuda.empty_cache(); logger.info("CUDA cache cleared.")
        elif hasattr(torch.backends, 'mps') and hasattr(torch.mps, 'empty_cache'):
            logger.info("Clearing MPS cache..."); torch.mps.empty_cache(); logger.info("MPS cache cleared.")

        logger.info("Script finished.")


if __name__ == "__main__":
    main()