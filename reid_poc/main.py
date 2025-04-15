# -*- coding: utf-8 -*-
"""Main execution script for the Multi-Camera Tracking & Re-Identification Pipeline."""

import logging
import time
from collections import defaultdict

import cv2
import torch
import numpy as np # Keep numpy import for potential future use if needed

# --- Local Modules ---
from config import setup_paths_and_config, PipelineConfig
from alias_types import ProcessedBatchResult # Use explicit type import
from data_loader import load_dataset_info, load_frames_for_batch
from models import load_detector, load_reid_model
from tracking import initialize_trackers
from pipeline import MultiCameraPipeline
from visualization import draw_annotations, display_combined_frames

# --- Setup Logging ---
# Configure logging level and format early
logging.basicConfig(
    level=logging.INFO, # Default level, can be overridden by config later if needed
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to set up and run the pipeline."""
    pipeline_instance: Optional[MultiCameraPipeline] = None
    last_batch_result: Optional[ProcessedBatchResult] = None
    config: Optional[PipelineConfig] = None # Keep track of config for cleanup/display

    # Keep track of loaded components for potential cleanup
    detector = None
    reid_model = None
    trackers = None

    try:
        # --- 1. Configuration and Setup ---
        config = setup_paths_and_config()

        # Optional: Adjust logging level based on loaded config
        # if config.enable_debug_logging:
        #    logging.getLogger().setLevel(logging.DEBUG)
        #    logger.info("DEBUG logging enabled.")

        # --- 2. Load Dataset Information ---
        camera_dirs, image_filenames = load_dataset_info(config)
        # config.selected_cameras might have been updated in load_dataset_info
        logger.info(f"Final list of cameras to process: {config.selected_cameras}")

        # --- 3. Load Models ---
        detector, detector_transforms = load_detector(config.device)
        reid_model = load_reid_model(config.reid_model_weights, config.device)
        # Note: reid_model can be None if BoxMOT/dependencies are missing

        # --- 4. Initialize Trackers ---
        trackers = initialize_trackers(
            config.selected_cameras, # Use potentially updated list
            config.tracker_type,
            config.tracker_config_path,
            config.device
        )

        # --- 5. Initialize Pipeline ---
        pipeline_instance = MultiCameraPipeline(
            config=config,
            detector=detector,
            detector_transforms=detector_transforms,
            reid_model=reid_model,
            trackers=trackers
        )

        # --- 6. Setup Display Window ---
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        logger.info(f"Display window '{config.window_name}' created.")

        # --- 7. Frame Processing Loop ---
        logger.info("--- Starting Frame Processing Loop ---")
        total_frames_loaded = 0
        total_frames_processed = 0 # Frames actually run through pipeline.process_frame_batch_full
        loop_start_time = time.perf_counter()

        for frame_idx, current_filename in enumerate(image_filenames):
            iter_start_time = time.perf_counter()

            # --- Load Frames for Current Batch ---
            current_frames = load_frames_for_batch(camera_dirs, current_filename)
            if not any(f is not None and f.size > 0 for f in current_frames.values()):
                logger.warning(f"Frame {frame_idx}: No valid images loaded for filename '{current_filename}'. Skipping this index.")
                continue # Skip to next filename if no cameras have this frame
            total_frames_loaded += 1

            # --- Frame Skipping Logic ---
            # Check if this frame index should be processed based on skip rate
            process_this_frame = (frame_idx % config.frame_skip_rate == 0)
            current_batch_timings = defaultdict(float)

            # --- Process or Skip Frame ---
            if process_this_frame:
                if pipeline_instance is not None:
                    # Run the full pipeline processing
                    batch_result = pipeline_instance.process_frame_batch_full(current_frames, frame_idx)
                    # Store the result for drawing (even if the next frame is skipped)
                    last_batch_result = batch_result
                    # Store timings from this processed frame
                    current_batch_timings = batch_result.timings
                    total_frames_processed += 1
                else:
                    # This should not happen if setup succeeded
                    logger.error("Pipeline not initialized! Cannot process frame. Exiting.")
                    break
            else: # Skipped frame
                # Minimal timing for skipped frame overhead (mainly loading)
                 current_batch_timings['skipped_frame_overhead'] = (time.perf_counter() - iter_start_time)
                 # Ensure last_batch_result persists for drawing annotations from the previous processed frame


            # --- Annotate and Display ---
            display_frames = current_frames # Start with the raw frames for annotation
            results_to_draw = {}
            if last_batch_result: # Use the most recent processing result for drawing
                results_to_draw = last_batch_result.results_per_camera

            # Draw annotations using the visualization function
            annotated_frames = draw_annotations(
                display_frames,
                results_to_draw,
                draw_bboxes=config.draw_bounding_boxes,
                show_track_id=config.show_track_id,
                show_global_id=config.show_global_id
            )

            # Display the combined annotated frames
            display_combined_frames(config.window_name, annotated_frames, config.max_display_width)

            # --- Logging and Timing ---
            iter_end_time = time.perf_counter()
            frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000 # Wall time for this iteration
            current_loop_duration = iter_end_time - loop_start_time
            # Avg FPS based on frames *loaded* (reflects display rate)
            avg_display_fps = total_frames_loaded / current_loop_duration if current_loop_duration > 0 else 0
            # Avg FPS based on frames *processed* (reflects pipeline throughput)
            avg_processing_fps = total_frames_processed / current_loop_duration if current_loop_duration > 0 else 0

            # --- Simplified Periodic Logging ---
            if frame_idx < 10 or frame_idx % 50 == 0 or not process_this_frame: # Log first few, every 50, and skipped
                track_count = 0
                if last_batch_result:
                    track_count = sum(len(tracks) for tracks in last_batch_result.results_per_camera.values())

                # Construct the detailed pipeline timing string ONLY if the frame was processed
                pipeline_timing_str = ""
                if process_this_frame and current_batch_timings:
                    # Selectively include main pipeline stage timings for brevity
                    stages_to_log = ['preprocess', 'detection_batched', 'postprocess_scale', 'tracking', 'feature_ext', 'reid', 'total']
                    pipeline_timings = {k: v for k, v in current_batch_timings.items() if k in stages_to_log}
                    # Format timings in milliseconds
                    pipeline_timing_str = " | Pipeline(ms): " + " | ".join([f"{k[:4]}={v * 1000:.1f}" for k, v in pipeline_timings.items() if v > 0.0001])

                status = "PROC" if process_this_frame else "SKIP"
                logger.info(
                    f"Frame {frame_idx:<4} [{status}] | IterTime:{frame_proc_time_ms:>6.1f}ms "
                    f"| AvgDispFPS:{avg_display_fps:5.1f} AvgProcFPS:{avg_processing_fps:5.1f} "
                    f"| Tracks:{track_count:<3}{pipeline_timing_str}"
                )

            # --- User Input Handling ---
            key = cv2.waitKey(config.display_wait_ms) & 0xFF
            if key == ord('q'):
                logger.info("Quit key (q) pressed. Exiting loop.")
                break
            elif key == ord('p'):
                logger.info("Pause key (p) pressed. Press any key in the OpenCV window to resume.")
                cv2.waitKey(0) # Wait indefinitely until a key is pressed
                logger.info("Resuming.")

            # Check if the display window was closed by the user
            try:
                 if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Display window was closed. Exiting loop.")
                    break
            except cv2.error:
                 logger.info("Display window seems to be closed or unavailable. Exiting loop.")
                 break # Exit if window property check fails (window might be destroyed)


        # --- End of Loop ---
        loop_end_time = time.perf_counter()
        total_time = loop_end_time - loop_start_time
        logger.info("--- Frame Processing Loop Finished ---")
        logger.info(f"Total Batches Loaded: {total_frames_loaded}, Total Batches Processed: {total_frames_processed}")
        if total_frames_loaded > 0 and total_time > 0.01:
            final_avg_display_fps = total_frames_loaded / total_time
            final_avg_processing_fps = total_frames_processed / total_time if total_frames_processed > 0 else 0
            logger.info(f"Total processing time: {total_time:.2f}s.")
            logger.info(f"Overall Avg Display FPS: {final_avg_display_fps:.2f}")
            logger.info(f"Overall Avg Processing FPS: {final_avg_processing_fps:.2f}")
        else:
            logger.info("Not enough frames or time elapsed to calculate meaningful average FPS.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError, ImportError) as e:
        logger.critical(f"Pipeline Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        # Catch any other unexpected errors
        logger.critical(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info("--- Cleaning up resources ---")
        cv2.destroyAllWindows()
        # Attempt to process any pending GUI events
        for _ in range(5): cv2.waitKey(1)

        # Explicitly delete models and pipeline object to help release memory, especially GPU
        del pipeline_instance
        del detector
        del reid_model
        del trackers
        del last_batch_result

        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        logger.info("Script finished.")


if __name__ == "__main__":
    main()