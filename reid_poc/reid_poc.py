import os
import sys  # Added for sys.exit
from pathlib import Path  # Import Path

import cv2
import numpy as np
import torch
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from scipy.spatial.distance import cosine as cosine_distance
from ultralytics import RTDETR

# --- Configuration ---
# Ensure the detector model path exists or Ultralytics can download it
DETECTOR_MODEL_PATH = "rtdetr-l.pt"
# Use a common lightweight OSNet model from BoxMOT's defaults
# BoxMOT will attempt to download this if not found locally in expected cache dirs
REID_MODEL_WEIGHTS_NAME = "osnet_x0_25_msmt17.pt"

# <<< IMPORTANT >>>: Make sure these image files exist in the same directory as the script
# or provide the full paths.
IMAGE_PATHS = ["image1.jpg", "image2.jpg", "image3.jpg"]

PERSON_CLASS_ID = 0  # COCO class ID for 'person'
CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence
REID_SIMILARITY_THRESHOLD = 0.6  # Threshold for considering two features as the same person

# --- Helper Functions ---

def get_device():
    """Selects the best available compute device."""
    if torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders (MPS) device.")
        return torch.device("mps")
    else:
        print("Using CPU device.")
        return torch.device("cpu")

def calculate_cosine_similarity(feat1: np.ndarray | None, feat2: np.ndarray | None) -> float:
    """Calculates cosine similarity between two feature vectors."""
    if feat1 is None or feat2 is None:
        print("Warning: Cannot calculate similarity with None features.")
        return 0.0
    # Ensure features are 1D arrays for cosine distance calculation
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    if feat1.shape != feat2.shape:
        print(f"Warning: Feature shapes differ: {feat1.shape} vs {feat2.shape}")
        return 0.0
    if np.all(feat1 == 0) or np.all(feat2 == 0):
        print("Warning: One or both feature vectors are all zeros.")
        return 0.0 # Avoid NaN from zero vectors

    # Cosine distance = 1 - cosine similarity
    distance = cosine_distance(feat1, feat2)
    similarity = 1.0 - distance
    # Clip similarity to [0, 1] range to handle potential floating point inaccuracies
    return float(np.clip(similarity, 0.0, 1.0))


def detect_and_extract(
    image_path: str, detector_model, reid_backend_instance
) -> np.ndarray | None:
    """
    Detects the most confident person in an image and extracts ReID features.

    Args:
        image_path: Path to the input image.
        detector_model: Loaded RTDETR model.
        reid_backend_instance: The specific backend instance (e.g., PyTorchBackend)
                               obtained from ReidAutoBackend.model.

    Returns:
        NumPy array of the ReID feature vector, or None if no person detected/extracted.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    print(f"\nProcessing image: {image_path}")
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Could not read image {image_path} with OpenCV.")
            return None
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

    # 1. Detect Persons
    try:
        results = detector_model.predict(
            img_bgr,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
    except Exception as e:
        print(f"  Error during object detection: {e}")
        return None

    # Ensure results are valid and detections exist
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("  No persons detected meeting the confidence threshold.")
        return None

    # 2. Get the best detection (highest confidence)
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
    confs = results[0].boxes.conf.cpu().numpy()  # Confidences

    if len(confs) == 0:
         print("  No persons detected (after potential filtering).")
         return None

    best_idx = np.argmax(confs)
    best_box_xyxy = boxes_xyxy[best_idx]
    best_conf = confs[best_idx]
    print(f"  Detected person with confidence {best_conf:.2f} at box {best_box_xyxy.astype(int)}")

    # 3. Extract Re-ID Features using the backend's integrated method
    try:
        # Ensure the box is in a NumPy array format, even if it's just one box
        person_bbox_np = np.array([best_box_xyxy])

        # Call get_features on the specific backend instance
        # It handles cropping and preprocessing based on the bounding box
        features = reid_backend_instance.get_features(person_bbox_np, img_bgr)

        if features is None or features.size == 0:
            print("  Error: Feature extraction returned None or empty array.")
            return None

        # Features should already be normalized and ready for comparison
        print(f"  Extracted features (shape: {features.shape})")
        # Return the first (and only) feature vector extracted
        return features[0] if features.ndim > 1 else features

    except Exception as e:
        print(f"  Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Main Execution ---
if __name__ == "__main__":

    # --- Prerequisite Checks ---
    if not all(os.path.exists(p) for p in IMAGE_PATHS):
        missing = [p for p in IMAGE_PATHS if not os.path.exists(p)]
        print(f"FATAL ERROR: Missing image files: {', '.join(missing)}")
        print(f"Please ensure '{IMAGE_PATHS[0]}', '{IMAGE_PATHS[1]}', and '{IMAGE_PATHS[2]}' exist in the current directory.")
        sys.exit(1) # Use sys.exit for cleaner exit

    # --- Device Selection ---
    DEVICE = get_device()

    # --- Model Loading ---
    print(f"\n--- Loading Models ---")
    print(f"Loading RTDETR detector from: {DETECTOR_MODEL_PATH}")
    try:
        detector = RTDETR(DETECTOR_MODEL_PATH)
        detector.to(DEVICE)
        # Simple warmup
        _ = detector.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        print("RTDETR Detector loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR loading detector: {e}")
        print("Ensure the model file exists or can be downloaded by ultralytics.")
        sys.exit(1)

    print(f"Loading ReID model backend (using weights: {REID_MODEL_WEIGHTS_NAME})")
    try:
        # *** FIX: Convert the weights string to a Path object ***
        reid_weights_path = Path(REID_MODEL_WEIGHTS_NAME)

        # Instantiate ReidAutoBackend
        reid_model_handler = ReidAutoBackend(
            weights=reid_weights_path, # Pass the Path object
            device=DEVICE,
            half=False # Keep False for broader compatibility unless GPU supports FP16 well
        )
        # The actual backend instance (e.g., FastReidBackend) is in .model
        reid_backend = reid_model_handler.model

        # Optional: Warmup the ReID model if the backend supports it
        if hasattr(reid_backend, 'warmup'):
             print("Warming up ReID model...")
             reid_backend.warmup()
        print("ReID Model backend loaded successfully.")

    except ImportError as e:
        print(f"FATAL ERROR loading ReID model: {e}")
        print("An import error occurred. This often means a dependency of boxmot (like 'fastreid') is missing.")
        print("Please check boxmot installation and its dependencies.")
        sys.exit(1)
    except FileNotFoundError as e:
         print(f"FATAL ERROR loading ReID model: {e}")
         print(f"Could not find or download the ReID weights: {REID_MODEL_WEIGHTS_NAME}")
         sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR loading ReID model: {e}")
        print("Ensure 'boxmot' is installed correctly and weights can be downloaded/found.")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- Feature Extraction ---
    print("\n--- Extracting Features ---")
    features_list = []
    for img_path in IMAGE_PATHS:
        # Pass the specific backend instance (reid_backend) to the function
        features = detect_and_extract(img_path, detector, reid_backend)
        features_list.append(features) # Appends the feature array or None

    # Check if feature extraction was successful for all images
    if None in features_list:
        print("\nError: Feature extraction failed for one or more images.")
        for i, f in enumerate(features_list):
            if f is None:
                print(f"  - Failed for image: {IMAGE_PATHS[i]}")
        print("Cannot perform comparisons.")
        sys.exit(1)

    # Unpack features if successful
    features1, features2, features3 = features_list

    # --- Comparisons ---
    print("\n--- Re-ID Comparisons ---")

    # Compare Image 1 and Image 2
    similarity_1_2 = calculate_cosine_similarity(features1, features2)
    is_same_1_2 = similarity_1_2 >= REID_SIMILARITY_THRESHOLD
    print(f"\nImage 1 ({IMAGE_PATHS[0]}) vs Image 2 ({IMAGE_PATHS[1]}):")
    print(f"  Cosine Similarity: {similarity_1_2:.4f}")
    print(f"  Result: {'Same Person' if is_same_1_2 else 'Different Person'} (Threshold: {REID_SIMILARITY_THRESHOLD})")

    # Compare Image 1 and Image 3
    similarity_1_3 = calculate_cosine_similarity(features1, features3)
    is_same_1_3 = similarity_1_3 >= REID_SIMILARITY_THRESHOLD
    print(f"\nImage 1 ({IMAGE_PATHS[0]}) vs Image 3 ({IMAGE_PATHS[2]}):")
    print(f"  Cosine Similarity: {similarity_1_3:.4f}")
    print(f"  Result: {'Same Person' if is_same_1_3 else 'Different Person'} (Threshold: {REID_SIMILARITY_THRESHOLD})")

    # Optional: Compare Image 2 and Image 3
    similarity_2_3 = calculate_cosine_similarity(features2, features3)
    is_same_2_3 = similarity_2_3 >= REID_SIMILARITY_THRESHOLD
    print(f"\nImage 2 ({IMAGE_PATHS[1]}) vs Image 3 ({IMAGE_PATHS[2]}):")
    print(f"  Cosine Similarity: {similarity_2_3:.4f}")
    print(f"  Result: {'Same Person' if is_same_2_3 else 'Different Person'} (Threshold: {REID_SIMILARITY_THRESHOLD})")


    print("\n--- POC Finished ---")