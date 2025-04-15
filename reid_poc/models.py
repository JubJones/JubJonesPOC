"""Functions for loading object detection and Re-Identification models."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN

# Ensure necessary torchvision transforms are available
try:
    from torchvision.transforms.v2 import Compose
except ImportError:
    from torchvision.transforms import Compose # Fallback

# BoxMOT imports only needed for ReID model loading
try:
    from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    from boxmot.appearance.backends.base_backend import BaseModelBackend
except ImportError as e:
    logging.critical(f"Failed to import boxmot components needed for ReID. Is boxmot installed? Error: {e}")
    # Allow script to potentially continue if only detection is used, but ReID will fail.
    BaseModelBackend = type(None) # Define a dummy type
    ReidAutoBackend = None


logger = logging.getLogger(__name__)

def get_reid_device_specifier_string(device: torch.device) -> str:
    """Converts a torch.device object to the string format expected by BoxMOT/ReID backends."""
    if device.type == 'cuda':
        # Return '0', '1' etc. for cuda devices, default to '0' if index is None
        return str(device.index if device.index is not None else 0)
    if device.type == 'mps':
        return 'mps' # Mac MPS support
    return 'cpu' # Default to CPU


def load_detector(device: torch.device) -> Tuple[FasterRCNN, Compose]:
    """Loads the Faster R-CNN ResNet50 FPN object detector with default weights."""
    logger.info("Loading SHARED Faster R-CNN detector...")
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        model.to(device)
        model.eval() # Set to evaluation mode
        transforms = weights.transforms()
        logger.info("Faster R-CNN Detector loaded successfully.")
        return model, transforms
    except Exception as e:
        logger.critical(f"FATAL ERROR loading Faster R-CNN: {e}", exc_info=True)
        raise RuntimeError("Failed to load object detector") from e

def load_reid_model(weights_path: Path, device: torch.device) -> Optional[BaseModelBackend]:
    """Loads the OSNet Re-Identification model using BoxMOT's ReidAutoBackend."""
    if ReidAutoBackend is None:
        logger.critical("FATAL ERROR: BoxMOT ReidAutoBackend not available. Cannot load ReID model.")
        raise ImportError("BoxMOT components required for ReID loading are missing.")

    logger.info(f"Loading SHARED OSNet ReID model from: {weights_path}")
    if not weights_path.is_file():
        logger.critical(f"FATAL ERROR: ReID weights file not found: {weights_path}")
        raise FileNotFoundError(f"ReID weights not found: {weights_path}")

    try:
        reid_device_specifier = get_reid_device_specifier_string(device)
        # Assuming FP32 ('half=False') for ReID based on original code
        reid_model_handler = ReidAutoBackend(weights=weights_path, device=reid_device_specifier, half=False)
        model = reid_model_handler.model # Get the underlying model instance

        # Warmup if the model supports it
        if hasattr(model, "warmup"):
            logger.info("Warming up ReID model...")
            model.warmup()
            logger.info("ReID model warmup complete.")

        logger.info("OSNet ReID Model loaded successfully.")
        return model
    except Exception as e:
        logger.critical(f"FATAL ERROR loading ReID model: {e}", exc_info=True)
        raise RuntimeError("Failed to load ReID model") from e