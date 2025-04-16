"""Main execution script for the Multi-Camera Tracking & Re-Identification Pipeline with Handoff."""

import logging
import time
from collections import defaultdict
from typing import Optional, Dict, Tuple, List
import json
from pathlib import Path
from datetime import datetime, timezone
import cProfile # For profiling guidance
import pstats   # For displaying profiling results
import cv2
import torch
import numpy as np

# --- Local Modules ---
from reid_poc.config import setup_paths_and_config, PipelineConfig
from reid_poc.alias_types import ProcessedBatchResult, CameraID, FrameData, GlobalID # Added GlobalID
from reid_poc.data_loader import load_dataset_info, load_frames_for_batch
from reid_poc.models import load_detector, load_reid_model
from reid_poc.tracking import initialize_trackers
# --- MODIFIED IMPORT ---
from pipeline_module import MultiCameraPipeline # Import from the new package
# --- END MODIFIED IMPORT ---
from reid_poc.visualization import (
    draw_annotations,
    display_combined_frames,
    create_single_camera_bev_map,
    display_combined_bev_maps,
    generate_unique_color # MODIFIED: Import generate_unique_color
)

# --- Setup Logging ---
# Assuming logging is configured globally elsewhere or via setup_paths_and_config
logger = logging.getLogger(__name__) # Get logger for this module


# --- Function to Save Predictions JSON --- (No changes needed from original)
def save_predictions_to_json(
    batch_result: ProcessedBatchResult,
    frame_idx: int,
    config: PipelineConfig,
    image_filenames_map: Dict[CameraID, str] # Map CameraID to its specific image filename
):
    """Constructs the JSON data and saves it to a file."""
    if not config.save_predictions or not config.output_predictions_dir:
        return # Saving disabled or output directory not set

    # Ensure output directory exists
    if not config.output_predictions_dir.is_dir():
        logger.error(f"Prediction output directory does not exist: {config.output_predictions_dir}. Cannot save.")
        return

    output_data = {
        "frame_index": frame_idx,
        "scene_id": config.selected_scene,
        "timestamp_processed_utc": datetime.now(timezone.utc).isoformat(),
        "cameras": {} # Initialize cameras dictionary
    }

    for cam_id, track_data_list in batch_result.results_per_camera.items():
        # Get the specific image filename for this camera and frame index
        image_source_name = Path(image_filenames_map.get(cam_id, f"frame_{frame_idx:06d}_unknown_cam_{cam_id}.jpg")).name
        camera_output = {
            "image_source": image_source_name,
            "tracks": [] # Initialize tracks list
        }
        for track_data in track_data_list:
            # Ensure numpy arrays are converted to lists for JSON serialization
            bbox = track_data.get('bbox_xyxy')
            bbox_list = bbox.tolist() if isinstance(bbox, np.ndarray) else bbox if isinstance(bbox, list) else None

            map_coords = track_data.get('map_coords')
            map_coords_list = list(map_coords) if map_coords is not None else None

            track_json = {
                "track_id": track_data.get('track_id'),
                "global_id": track_data.get('global_id'), # Will be None if not assigned -> null
                "bbox_xyxy": bbox_list,
                "confidence": track_data.get('conf'),
                "class_id": track_data.get('class_id'),
                "map_coords": map_coords_list
            }
            # Only add track if essential info is present
            if track_json["bbox_xyxy"] is not None and track_json["track_id"] is not None:
                camera_output["tracks"].append(track_json)
            # else: logger.debug(...) # Optional debug log for skipped tracks

        output_data["cameras"][cam_id] = camera_output

    # Add entries for cameras that might exist in the config but had no tracks in this frame
    for cam_id in config.selected_cameras:
        if cam_id not in output_data["cameras"]:
            image_source_name = Path(image_filenames_map.get(cam_id, f"frame_{frame_idx:06d}_unknown_cam_{cam_id}.jpg")).name
            output_data["cameras"][cam_id] = {
                "image_source": image_source_name,
                "tracks": [] # Empty tracks list
            }

    # Define output filename
    output_filename = config.output_predictions_dir / f"scene_{config.selected_scene}_frame_{frame_idx:06d}.json"

    try:
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2) # Use indent for readability
        logger.debug(f"Saved predictions for frame {frame_idx} to {output_filename.name}")
    except IOError as e:
        logger.error(f"Failed to write predictions for frame {frame_idx} to {output_filename}: {e}")
    except TypeError as e:
        logger.error(f"JSON serialization error for frame {frame_idx}: {e}. Data sample: {str(output_data)[:500]}")


# --- Function to Save BEV Map Images --- (No changes needed)
def save_bev_map_images(
    bev_map_images: Dict[CameraID, FrameData],
    frame_idx: int,
    config: PipelineConfig
):
    """Saves the generated BEV map images to files."""
    if not config.save_bev_maps or not config.output_bev_maps_dir or not config.enable_bev_map: return
    if not config.output_bev_maps_dir.is_dir(): logger.error(f"BEV map output directory does not exist: {config.output_bev_maps_dir}. Cannot save."); return

    for cam_id, bev_image in bev_map_images.items():
        if bev_image is None or bev_image.size == 0: continue
        output_filename = config.output_bev_maps_dir / f"scene_{config.selected_scene}_frame_{frame_idx:06d}_bev_{cam_id}.png"
        try:
            success = cv2.imwrite(str(output_filename), bev_image)
            if success: logger.debug(f"Saved BEV map {cam_id} frame {frame_idx} to {output_filename.name}")
            else: logger.warning(f"Failed save BEV map {cam_id} frame {frame_idx} (imwrite returned False)")
        except Exception as e: logger.error(f"Error saving BEV map {cam_id} frame {frame_idx}: {e}", exc_info=False)

def main_logic():
    """Contains the main application logic, separated for potential profiling."""
    pipeline_instance: Optional[MultiCameraPipeline] = None
    last_batch_result: Optional[ProcessedBatchResult] = None
    config: Optional[PipelineConfig] = None
    is_paused: bool = False
    bev_window_name: Optional[str] = None
    bev_map_images_current_frame: Dict[CameraID, FrameData] = {}
    preloaded_bev_background: Optional[np.ndarray] = None

    detector = None; reid_model = None; trackers = None # Initialize for finally block

    try:
        # --- 1. Configuration and Setup ---
        config = setup_paths_and_config() # Includes YAML loading, path validation etc.

        # --- 1b. Preload BEV Background ---
        if config.enable_bev_map and config.bev_map_background_path:
            logger.info(f"Pre-loading BEV background from: {config.bev_map_background_path}")
            try:
                img = cv2.imread(str(config.bev_map_background_path))
                if img is not None and img.size > 0:
                    target_h, target_w = config.bev_map_display_size
                    preloaded_bev_background = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    logger.info(f"Successfully loaded and resized BEV background to {config.bev_map_display_size}.")
                else:
                    logger.warning(f"Failed to read BEV background image file: {config.bev_map_background_path}")
            except Exception as e:
                logger.error(f"Error loading or resizing BEV background: {e}")
            if preloaded_bev_background is None:
                logger.warning("Proceeding without preloaded BEV background (using black).")

        # --- 2. Load Dataset Info ---
        camera_dirs, image_filenames = load_dataset_info(config)
        logger.info(f"Processing scene '{config.selected_scene}' with {len(image_filenames)} frames for cameras: {config.selected_cameras}")

        # --- 3. Load Models ---
        detector, detector_transforms = load_detector(config.device)
        reid_model = load_reid_model(config.reid_model_weights, config.device)

        # --- 4. Initialize Trackers ---
        trackers = initialize_trackers(config.selected_cameras, config.tracker_type, config.tracker_config_path, config.device)

        # --- 5. Initialize Pipeline (Uses the imported class) ---
        pipeline_instance = MultiCameraPipeline(config, detector, detector_transforms, reid_model, trackers)

        # --- 6. Setup Display Windows ---
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        logger.info(f"Display window '{config.window_name}' created.")
        if config.enable_bev_map:
            bev_window_name = config.window_name + " - Combined BEV Map"
            cv2.namedWindow(bev_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            logger.info(f"Combined BEV map window '{bev_window_name}' created.")

        logger.info("Press 'p' to pause/resume, 'q' to quit.")

        # --- 7. Frame Processing Loop ---
        logger.info("--- Starting Frame Processing Loop ---")
        total_frames_loaded = 0; total_frames_processed = 0; frame_idx = 0
        loop_start_time = time.perf_counter()
        frame_shapes_for_viz: Dict[CameraID, Optional[Tuple[int, int]]] = {
            cam_id: cam_cfg.frame_shape for cam_id, cam_cfg in config.cameras_config.items()
        }
        bev_layout_order: List[CameraID] = config.selected_cameras # Use sorted order from config

        while frame_idx < len(image_filenames):
            iter_start_time = time.perf_counter()
            current_filename_base = image_filenames[frame_idx]
            current_filenames_map = {cam_id: current_filename_base for cam_id in config.selected_cameras}

            current_frames = load_frames_for_batch(camera_dirs, current_filename_base)

            if not is_paused:
                # --- Process Frame ---
                if not any(f is not None and f.size > 0 for f in current_frames.values()):
                    if frame_idx < 10: logger.warning(f"Frame {frame_idx}: No valid images loaded for '{current_filename_base}'. Skipping index.")
                    frame_idx += 1; continue
                total_frames_loaded += 1
                process_this_frame = (frame_idx % config.frame_skip_rate == 0)
                current_batch_timings = defaultdict(float)

                # Reset BEV images for the new frame (used for tiling)
                bev_map_images_current_frame.clear()

                if process_this_frame:
                    if pipeline_instance is not None:
                        # Calls the method from the imported MultiCameraPipeline class
                        batch_result = pipeline_instance.process_frame_batch_full(current_frames, frame_idx)
                        last_batch_result = batch_result
                        current_batch_timings = batch_result.timings
                        total_frames_processed += 1

                        # Save Predictions JSON (if enabled)
                        if config.save_predictions and last_batch_result:
                            save_predictions_to_json(last_batch_result, frame_idx, config, current_filenames_map)
                    else: logger.error("Pipeline not initialized! Exiting."); break
                else: # Frame skipped
                    current_batch_timings['skipped_frame_overhead'] = (time.perf_counter() - iter_start_time)
                    # Stale 'last_batch_result' will be used for display

                # --- Logging --- (Do this before advancing frame_idx)
                iter_end_time = time.perf_counter(); frame_proc_time_ms = (iter_end_time - iter_start_time) * 1000; current_loop_duration = iter_end_time - loop_start_time; avg_display_fps = total_frames_loaded / current_loop_duration if current_loop_duration > 0 else 0; avg_processing_fps = total_frames_processed / current_loop_duration if current_loop_duration > 0 else 0
                if frame_idx < 10 or frame_idx % 50 == 0 or not process_this_frame:
                    track_count = 0; trigger_count = 0; map_coord_count = 0
                    if last_batch_result: track_count = sum(len(tracks) for tracks in last_batch_result.results_per_camera.values()); trigger_count = len(last_batch_result.handoff_triggers); map_coord_count = sum(1 for tracks in last_batch_result.results_per_camera.values() for t in tracks if t.get('map_coords') is not None)
                    pipeline_timing_str = ""
                    if process_this_frame and current_batch_timings: stages = ['preprocess', 'detection', 'postproc', 'tracking', 'handoff', 'feature', 'reid', 'projection', 'total', 'state_update']; pipeline_timings = {k: v for k, v in current_batch_timings.items() if any(s in k for s in stages)}; pipeline_timing_str = " | Pipe(ms): " + " ".join([f"{k[:10]}={v*1000:.1f}" for k, v in sorted(pipeline_timings.items()) if v > 0.0001]) # Wider key display
                    status = "PROC" if process_this_frame else "SKIP"; logger.info(f"Frame {frame_idx:<4} [{status}] | Iter:{frame_proc_time_ms:>6.1f}ms | AvgDisp:{avg_display_fps:5.1f} AvgProc:{avg_processing_fps:5.1f} | Trk:{track_count:<3} Trig:{trigger_count:<2} Map:{map_coord_count:<3}{pipeline_timing_str}")

                # --- Advance frame index ---
                frame_idx += 1 # AFTER processing and logging for the current index

            # --- END OF PROCESSING BLOCK (if not is_paused) ---

            # --- Annotate and Display (Always uses last_batch_result) ---
            display_frames = current_frames # Use frames loaded for this index
            results_to_draw = {}; triggers_to_draw = []
            if last_batch_result: # Use latest available results
                results_to_draw = last_batch_result.results_per_camera
                triggers_to_draw = last_batch_result.handoff_triggers

            # --- MODIFIED: Calculate Global Color Map for this Frame ---
            global_id_color_map_for_frame: Dict[GlobalID, Tuple[int, int, int]] = {}
            if last_batch_result:
                all_gids_in_batch = set()
                for cam_results in last_batch_result.results_per_camera.values():
                    for track_data in cam_results:
                        gid = track_data.get('global_id')
                        if gid is not None:
                            all_gids_in_batch.add(gid)
                # Generate colors only for GIDs present in this frame's results
                for gid in all_gids_in_batch:
                    # generate_unique_color imported from visualization
                    global_id_color_map_for_frame[gid] = generate_unique_color(gid)


            # --- MODIFIED: Pass Color Map to draw_annotations ---
            annotated_frames = draw_annotations(
                display_frames, results_to_draw, triggers_to_draw,
                frame_shapes_for_viz,
                global_id_color_map_for_frame, # Pass the calculated map
                config.draw_bounding_boxes, config.show_track_id, config.show_global_id,
                config.draw_quadrant_lines, config.highlight_handoff_triggers
            )

            # --- Create, Save, and Display Combined BEV Map ---
            if config.enable_bev_map and bev_window_name:
                # Generate individual maps using preloaded background AND common color map
                for cam_id in config.selected_cameras:
                    results_this_cam = results_to_draw.get(cam_id, [])
                    # --- MODIFIED: Pass Color Map to create_single_camera_bev_map ---
                    bev_img = create_single_camera_bev_map(
                        cam_id,
                        results_this_cam,
                        config,
                        preloaded_bev_background, # Pass preloaded background (might be None)
                        global_id_color_map_for_frame # Pass the calculated color map
                    )
                    if bev_img is not None:
                        bev_map_images_current_frame[cam_id] = bev_img

                # Save BEV maps (if enabled) - Use frame_idx-1 because idx was incremented
                # Only save if the frame was actually processed
                if config.save_bev_maps and (frame_idx-1) % config.frame_skip_rate == 0:
                    # Use frame_idx-1 to match the processed frame number
                    save_bev_map_images(bev_map_images_current_frame, frame_idx - 1, config)

                # Tile and display
                display_combined_bev_maps(
                    window_name=bev_window_name,
                    bev_map_images=bev_map_images_current_frame,
                    layout_order=bev_layout_order,
                    target_cell_shape=config.bev_map_display_size
                )
            # --- END BEV MAP BLOCK ---

            # --- Display Combined Camera Views ---
            display_combined_frames(config.window_name, annotated_frames, config.max_display_width)

            # --- User Input & Window Checks ---
            wait_duration = config.display_wait_ms if not is_paused else 50
            key = cv2.waitKey(wait_duration) & 0xFF
            if key == ord('q'): logger.info("Quit key (q) pressed."); break
            elif key == ord('p'): is_paused = not is_paused; logger.info("<<<< PAUSED >>>>" if is_paused else ">>>> RESUMED >>>>")

            # Check if windows closed
            main_window_closed = False; bev_window_closed = False
            try: # Check main window
                if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1: main_window_closed = True; logger.info("Main display window closed.")
            except cv2.error: main_window_closed = True # Assume closed if property check fails
            if not main_window_closed and config.enable_bev_map and bev_window_name:
                try:
                    if cv2.getWindowProperty(bev_window_name, cv2.WND_PROP_VISIBLE) < 1: bev_window_closed = True; logger.info(f"Combined BEV display window '{bev_window_name}' closed.")
                except cv2.error: bev_window_closed = True # Assume closed
            if main_window_closed or bev_window_closed: logger.info("A display window was closed. Exiting loop."); break


        # --- End of Loop ---
        loop_end_time = time.perf_counter(); total_time = loop_end_time - loop_start_time
        logger.info(f"--- Frame Processing Loop Finished (Loaded {total_frames_loaded} frames, Processed {total_frames_processed} frames) ---")
        if total_frames_loaded > 0 and total_time > 0.01: final_avg_display_fps = total_frames_loaded / total_time; final_avg_processing_fps = total_frames_processed / total_time if total_frames_processed > 0 else 0; logger.info(f"Total loop time: {total_time:.2f}s."); logger.info(f"Overall Avg Display FPS: {final_avg_display_fps:.2f}"); logger.info(f"Overall Avg Processing FPS: {final_avg_processing_fps:.2f}")
        else: logger.info("Not enough frames or time elapsed for meaningful average FPS.")


    except (FileNotFoundError, RuntimeError, ModuleNotFoundError, ImportError, ValueError) as e: # Added ValueError
        logger.critical(f"Pipeline Setup/Execution Error: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info("--- Cleaning up resources ---")
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1) # Help process GUI events on some systems

        # Use del for explicit cleanup if needed, though Python's GC handles most cases
        del pipeline_instance; del detector; del reid_model; del trackers; del last_batch_result; del config;
        bev_map_images_current_frame.clear()
        del preloaded_bev_background # Delete the preloaded image

        if torch.cuda.is_available(): logger.info("Clearing CUDA cache..."); torch.cuda.empty_cache(); logger.info("CUDA cache cleared.")
        elif hasattr(torch.backends, 'mps') and hasattr(torch.mps, 'empty_cache'): logger.info("Clearing MPS cache..."); torch.mps.empty_cache(); logger.info("MPS cache cleared.")

        logger.info("Script finished.")


if __name__ == "__main__":
    # --- cProfile Integration (Optional: Uncomment to enable) ---
    enable_profiling = False # Set to True to enable profiling
    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    # -----------------------------------------

    main_logic() # Run the main application logic

    # --- cProfile Results Handling (Optional: Uncomment to enable) ---
    if enable_profiling and profiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative') # Sort by cumulative time
        # Print top 30 functions by cumulative time
        print("\n--- cProfile Cumulative Time Results (Top 30) ---")
        stats.print_stats(30)
        # Save full stats to a file for more detailed analysis (e.g., with snakeviz)
        profile_output_file = "pipeline_profile.prof"
        try:
            stats.dump_stats(profile_output_file)
            print(f"Full profiling stats saved to: {profile_output_file}")
        except Exception as e:
            print(f"Error saving profiling stats: {e}")
    # ---------------------------------------------