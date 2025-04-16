# -*- coding: utf-8 -*-
"""Main execution script for the Multi-Camera Tracking & Re-Identification Pipeline with Handoff."""

import logging
import time
from collections import defaultdict
from typing import Optional, Dict, Tuple, List # Added List

import cv2
import torch
import numpy as np

# --- Local Modules ---
from reid_poc.config import setup_paths_and_config, PipelineConfig
from reid_poc.alias_types import ProcessedBatchResult, CameraID, FrameData
from reid_poc.data_loader import load_dataset_info, load_frames_for_batch
from reid_poc.models import load_detector, load_reid_model
from reid_poc.tracking import initialize_trackers
from reid_poc.pipeline import MultiCameraPipeline
# Import visualization functions
from reid_poc.visualization import ( # Import specific functions
    draw_annotations,
    display_combined_frames,
    create_single_camera_bev_map, # Keep this for generating individual maps
    display_combined_bev_maps   # Add the new tiling function
)

# --- Setup Logging ---
logger = logging.getLogger(__name__) # Get logger for this module

def main():
    """Main function to set up and run the pipeline."""
    pipeline_instance: Optional[MultiCameraPipeline] = None
    last_batch_result: Optional[ProcessedBatchResult] = None
    config: Optional[PipelineConfig] = None
    is_paused: bool = False # Pause flag
    # --- Revert back to single BEV window setup ---
    bev_window_name: Optional[str] = None # Name for the single combined BEV window
    bev_map_images: Dict[CameraID, FrameData] = {} # Stores the latest *individual* BEV images

    detector = None; reid_model = None; trackers = None # Initialize for finally block

    try:
        # --- 1. Configuration and Setup ---
        config = setup_paths_and_config() # Logging level is set inside here now

        # --- 2. Load Dataset Info ---
        camera_dirs, image_filenames = load_dataset_info(config)
        logger.info(f"Processing scene '{config.selected_scene}' with {len(image_filenames)} frames for cameras: {config.selected_cameras}")

        # --- 3. Load Models ---
        detector, detector_transforms = load_detector(config.device)
        reid_model = load_reid_model(config.reid_model_weights, config.device)

        # --- 4. Initialize Trackers ---
        trackers = initialize_trackers(config.selected_cameras, config.tracker_type, config.tracker_config_path, config.device)

        # --- 5. Initialize Pipeline ---
        pipeline_instance = MultiCameraPipeline(config, detector, detector_transforms, reid_model, trackers)

        # --- 6. Setup Display Windows ---
        # Main camera views window
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        logger.info(f"Display window '{config.window_name}' created.")

        # Single Combined BEV map window if enabled
        logger.debug(f"BEV Map enabled in config: {config.enable_bev_map}")
        if config.enable_bev_map:
            bev_window_name = config.window_name + " - Combined BEV Map" # New name
            cv2.namedWindow(bev_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            # We don't resize here, the tiling function determines final size
            logger.info(f"Combined BEV map window '{bev_window_name}' created.")

        logger.info("Press 'p' to pause/resume, 'q' to quit.")


        # --- 7. Frame Processing Loop ---
        logger.info("--- Starting Frame Processing Loop ---")
        total_frames_loaded = 0; total_frames_processed = 0; frame_idx = 0
        loop_start_time = time.perf_counter()
        frame_shapes_for_viz: Dict[CameraID, Optional[Tuple[int, int]]] = {cam_id: cfg.frame_shape for cam_id, cfg in config.cameras_handoff_config.items()}
        # Define the desired layout order for the BEV grid
        # Using sorted order which matches c09, c12, c13, c16
        bev_layout_order: List[CameraID] = config.selected_cameras

        while frame_idx < len(image_filenames):
            iter_start_time = time.perf_counter()
            current_filename = image_filenames[frame_idx]
            current_frames = load_frames_for_batch(camera_dirs, current_filename)

            if not is_paused:
                # --- Process Frame ---
                if not any(f is not None and f.size > 0 for f in current_frames.values()):
                    if frame_idx < 10: logger.warning(f"Frame {frame_idx}: No valid images loaded for '{current_filename}'. Skipping index.")
                    frame_idx += 1; continue
                total_frames_loaded += 1
                process_this_frame = (frame_idx % config.frame_skip_rate == 0)
                current_batch_timings = defaultdict(float)
                if process_this_frame:
                    if pipeline_instance is not None:
                        batch_result = pipeline_instance.process_frame_batch_full(current_frames, frame_idx)
                        last_batch_result = batch_result
                        current_batch_timings = batch_result.timings
                        total_frames_processed += 1
                    else: logger.error("Pipeline not initialized! Exiting."); break
                else: current_batch_timings['skipped_frame_overhead'] = (time.perf_counter() - iter_start_time)

                # --- Logging ---
                iter_end_time = time.perf_counter(); frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000; current_loop_duration = iter_end_time - loop_start_time; avg_display_fps = total_frames_loaded / current_loop_duration if current_loop_duration > 0 else 0; avg_processing_fps = total_frames_processed / current_loop_duration if current_loop_duration > 0 else 0
                if frame_idx < 10 or frame_idx % 50 == 0 or not process_this_frame:
                    track_count = 0; trigger_count = 0; map_coord_count = 0
                    if last_batch_result: track_count = sum(len(tracks) for tracks in last_batch_result.results_per_camera.values()); trigger_count = len(last_batch_result.handoff_triggers); map_coord_count = sum(1 for tracks in last_batch_result.results_per_camera.values() for t in tracks if t.get('map_coords') is not None)
                    pipeline_timing_str = ""
                    if process_this_frame and current_batch_timings: stages = ['preprocess', 'detection', 'postproc', 'tracking', 'handoff', 'feature', 'reid', 'projection', 'total']; pipeline_timings = {k: v for k, v in current_batch_timings.items() if any(k.startswith(s) for s in stages)}; pipeline_timing_str = " | Pipe(ms): " + " ".join([f"{k[:5]}={v*1000:.1f}" for k, v in sorted(pipeline_timings.items()) if v > 0.0001])
                    status = "PROC" if process_this_frame else "SKIP"; logger.info(f"Frame {frame_idx:<4} [{status}] | Iter:{frame_proc_time_ms:>6.1f}ms | AvgDisp:{avg_display_fps:5.1f} AvgProc:{avg_processing_fps:5.1f} | Trk:{track_count:<3} Trig:{trigger_count:<2} Map:{map_coord_count:<3}{pipeline_timing_str}")

                frame_idx += 1 # Advance frame index only if not paused

            # --- Annotate and Display ---
            display_frames = current_frames
            results_to_draw = {}; triggers_to_draw = []
            if last_batch_result:
                results_to_draw = last_batch_result.results_per_camera
                triggers_to_draw = last_batch_result.handoff_triggers
            logger.debug(f"Frame {frame_idx-1}: Passing {sum(len(v) for v in results_to_draw.values())} track results to visualization.")

            annotated_frames = draw_annotations(display_frames, results_to_draw, triggers_to_draw, frame_shapes_for_viz, config.draw_bounding_boxes, config.show_track_id, config.show_global_id, config.draw_quadrant_lines, config.highlight_handoff_triggers)

            # --- Create and Display Combined BEV Map ---
            if config.enable_bev_map and bev_window_name:
                bev_map_images.clear() # Clear images from previous frame
                # Generate individual maps first
                for cam_id in config.selected_cameras:
                    results_this_cam = results_to_draw.get(cam_id, [])
                    bev_img = create_single_camera_bev_map(cam_id, results_this_cam, config)
                    if bev_img is not None:
                        bev_map_images[cam_id] = bev_img
                    # else: # Already logged in create_single_camera_bev_map if it fails

                # Now tile the generated maps and display in the single BEV window
                display_combined_bev_maps(
                    window_name=bev_window_name,
                    bev_map_images=bev_map_images,
                    layout_order=bev_layout_order, # Use the defined order
                    target_cell_shape=config.bev_map_display_size # Use config for cell size
                )

            # --- Display Combined Camera Views ---
            display_combined_frames(config.window_name, annotated_frames, config.max_display_width)

            # --- User Input & Window Checks ---
            wait_duration = config.display_wait_ms if not is_paused else 50
            key = cv2.waitKey(wait_duration) & 0xFF
            if key == ord('q'): logger.info("Quit key (q) pressed."); break
            elif key == ord('p'): is_paused = not is_paused; logger.info("<<<< PAUSED >>>>" if is_paused else ">>>> RESUMED >>>>")

            # Check if any display window was closed
            main_window_closed = False; bev_window_closed = False
            try: # Check main window
                if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1: main_window_closed = True; logger.info("Main display window closed.")
            except cv2.error: logger.info("Main display window check failed."); main_window_closed = True
            # Check combined BEV window only if not already flagged
            if not main_window_closed and config.enable_bev_map and bev_window_name:
                try:
                    if cv2.getWindowProperty(bev_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        bev_window_closed = True
                        logger.info(f"Combined BEV display window '{bev_window_name}' closed.")
                except cv2.error:
                     logger.info(f"Combined BEV display window check failed for '{bev_window_name}'. Assuming closed."); bev_window_closed = True

            if main_window_closed or bev_window_closed: logger.info("A display window was closed. Exiting loop."); break

        # --- End of Loop ---
        loop_end_time = time.perf_counter(); total_time = loop_end_time - loop_start_time
        logger.info(f"--- Frame Processing Loop Finished (Processed {total_frames_processed} frames) ---")
        if total_frames_loaded > 0 and total_time > 0.01: final_avg_display_fps = total_frames_loaded / total_time; final_avg_processing_fps = total_frames_processed / total_time if total_frames_processed > 0 else 0; logger.info(f"Total loop time: {total_time:.2f}s."); logger.info(f"Overall Avg Display FPS: {final_avg_display_fps:.2f}"); logger.info(f"Overall Avg Processing FPS: {final_avg_processing_fps:.2f}")
        else: logger.info("Not enough frames or time elapsed for meaningful average FPS.")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError, ImportError) as e:
        logger.critical(f"Pipeline Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info("--- Cleaning up resources ---")
        cv2.destroyAllWindows() # Destroys all OpenCV windows
        for _ in range(10): cv2.waitKey(1) # Help process GUI events

        del pipeline_instance; del detector; del reid_model; del trackers; del last_batch_result; del config;
        bev_map_images.clear()

        if torch.cuda.is_available(): logger.info("Clearing CUDA cache..."); torch.cuda.empty_cache(); logger.info("CUDA cache cleared.")
        elif hasattr(torch.backends, 'mps') and hasattr(torch.mps, 'empty_cache'): logger.info("Clearing MPS cache..."); torch.mps.empty_cache(); logger.info("MPS cache cleared.")

        logger.info("Script finished.")


if __name__ == "__main__":
    main()