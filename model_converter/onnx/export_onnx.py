# FILE: export_onnx.py
# -*- coding: utf-8 -*-
import os
import sys
import logging
from pathlib import Path
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN
# Ensure torchvision.transforms.v2 is available or fallback if needed
try:
    from torchvision.transforms.v2 import Compose
except ImportError:
    from torchvision.transforms import Compose # Fallback for older torchvision
from PIL import Image
import numpy as np

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Minimal Config for Export ---
@dataclass
class ExportConfig:
    """Configuration settings relevant for ONNX export."""
    detection_input_width: Optional[int] = 640
    # Device choice moved to get_export_device

# --- Helper Functions ---
def get_export_device() -> torch.device:
    """Selects CPU for export stability, logs GPU availability."""
    logger.info("--- Determining Compute Device for Export ---")
    if torch.cuda.is_available():
        try:
            gpu_device = torch.device("cuda")
            logger.info(f"CUDA available: {torch.cuda.get_device_name(gpu_device)}")
            _ = torch.tensor([1.0], device=gpu_device) + torch.tensor([1.0], device=gpu_device)
            logger.info("CUDA confirmed functional.")
        except Exception as e: logger.warning(f"CUDA reported available, but test failed ({e}).")
    logger.info("Using CPU device for ONNX export for better compatibility.")
    return torch.device("cpu")

def _load_detector_for_export(device: torch.device) -> Tuple[FasterRCNN, Optional[Compose]]:
    """Loads the Faster R-CNN model onto the specified device."""
    logger.info(f"Loading Faster R-CNN detector onto device: {device}...")
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        model.to(device)
        model.eval()
        # Store transforms, but they are not directly used in export itself
        transforms = weights.transforms()
        logger.info("Faster R-CNN Detector loaded successfully for export.")
        return model, transforms
    except Exception as e:
        logger.critical(f"FATAL ERROR loading Faster R-CNN: {e}", exc_info=True)
        raise RuntimeError("Failed to load object detector") from e

def calculate_input_shape(config: ExportConfig) -> Tuple[int, int]:
    """Calculates a typical input H, W based on config width FOR EXAMPLE INPUT."""
    if config.detection_input_width:
        target_w = config.detection_input_width
        # Use a common aspect ratio like 16:9 for H calculation
        target_h = int(target_w * 9.0 / 16.0)
        # Ensure H is multiple of 32 (common requirement/optimization for CNNs)
        # Add 31 then integer divide by 32, then multiply by 32
        target_h = (target_h + 31) // 32 * 32 if target_h > 32 else 32
        logger.info(f"Using example input shape (H, W): ({target_h}, {target_w}) for ONNX export trace.")
        return target_h, target_w
    else:
        # Provide a default if no width is given
        logger.warning("No detection_input_width set. Using default example size (544, 960).")
        return 544, 960 # Example default H, W

# --- Main Export Function ---
def export_model_to_onnx(config: ExportConfig, output_dir: Path, onnx_filename: str = "detector.onnx"):
    """Exports the Faster R-CNN model to ONNX format with dynamic input H/W."""

    export_device = get_export_device()
    model, _ = _load_detector_for_export(device=export_device) # Transforms not needed for export logic

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_file_path = output_dir / onnx_filename

    # 1. Define Example Input (Use 4D: Batch, Channel, Height, Width)
    example_h, example_w = calculate_input_shape(config)
    batch_size = 1 # Explicit batch size
    channels = 3   # RGB
    # Create a 4D tensor directly
    example_input_tensor = torch.randn(batch_size, channels, example_h, example_w, device=export_device, dtype=torch.float32)
    # The argument to export should be a tuple containing this tensor
    example_input_tuple = (example_input_tensor,)
    logger.info(f"Created example input tensor for tracing with shape: {example_input_tensor.shape}")

    # 2. Define Dynamic Axes (CRUCIAL - Mark H and W as dynamic on the 4D input)
    input_names = ["images"] # Single input name representing the 4D tensor [B,C,H,W]
    # Standard Faster R-CNN output names
    output_names = ["boxes", "labels", "scores"]

    # Input shape is [B, C, H, W] (rank 4)
    # Mark dimension 2 (Height) and dimension 3 (Width) as dynamic
    dynamic_axes = {
        input_names[0]: {2: 'height', 3: 'width'},  # Input: H (idx 2) and W (idx 3) are dynamic
        output_names[0]: {0: 'num_detections'},     # Output boxes: num_detections is dynamic
        output_names[1]: {0: 'num_detections'},     # Output labels: num_detections is dynamic
        output_names[2]: {0: 'num_detections'}      # Output scores: num_detections is dynamic
    }

    logger.info(f"Input names: {input_names}")
    logger.info(f"Output names: {output_names}")
    logger.info(f"Dynamic axes specified for 4D input: {dynamic_axes}")

    # 3. Perform Export
    try:
        logger.info(f"Attempting to export model to {onnx_file_path} with opset 13...")
        # Pass the 4D tensor tuple directly
        torch.onnx.export(
            model,
            example_input_tuple, # Pass the tuple (Tensor(B,C,H,W),)
            str(onnx_file_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,        # Try opset 13 or 16 if available
            verbose=False,           # Set to True for extremely detailed export logs if needed
            export_params=True,      # Include model weights in the ONNX file
            #do_constant_folding=True, # Optional: can sometimes help, sometimes hinder TRT
        )
        logger.info(f"Model potentially exported successfully with dynamic H/W (4D input) to {onnx_file_path}")
        logger.info("Please verify the exported model using Netron and try building the TensorRT engine again.")

    except Exception as e:
        logger.critical(f"Error during ONNX export: {e}", exc_info=True)
        logger.error("Export FAILED. Check error message.")
        # Attempt to remove potentially corrupted ONNX file
        if onnx_file_path.exists():
            try:
                onnx_file_path.unlink()
                logger.info(f"Removed potentially incomplete ONNX file: {onnx_file_path}")
            except OSError as unlink_err:
                logger.error(f"Failed to remove incomplete ONNX file: {unlink_err}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Faster R-CNN model to ONNX with Dynamic Shapes (4D Input).")
    parser.add_argument(
        "--output-dir", type=str, default="./onnx_models",
        help="Directory to save the exported ONNX model."
    )
    parser.add_argument(
        "--filename", type=str, default="detector_dynamic.onnx",
        help="Name for the output ONNX file."
    )
    parser.add_argument(
        "--input-width", type=int, default=640,
        help="Target input width for calculating *example* tensor height (e.g., 640)."
    )
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    input_w = args.input_width if args.input_width > 0 else None
    export_config = ExportConfig(detection_input_width=input_w)
    export_model_to_onnx(export_config, output_path, args.filename)