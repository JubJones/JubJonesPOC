import os

import cv2
import numpy as np
import torch
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from scipy.spatial.distance import cosine as cosine_distance
from ultralytics import RTDETR

# --- Configuration ---
DETECTOR_MODEL_PATH = "rtdetr-l.pt"
# Use a common lightweight OSNet model from BoxMOT's defaults
REID_MODEL_WEIGHTS = "osnet_x0_25_msmt17.pt"
IMAGE_PATHS = ["image1.jpg", "image2.jpg", "image3.jpg"]
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
REID_SIMILARITY_THRESHOLD = 0.6


# REID_DISTANCE_THRESHOLD = 0.4 # Alternative

# --- Device Selection ---
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders (MPS) device.")
        return torch.device("mps")
    else:
        print("Using CPU device.")
        return torch.device("cpu")


DEVICE = get_device()

# --- Model Loading ---
print(f"Loading RTDETR detector from: {DETECTOR_MODEL_PATH}")
try:
    detector = RTDETR(DETECTOR_MODEL_PATH)
    detector.to(DEVICE)
    _ = detector.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("RTDETR Detector loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading detector: {e}")
    exit()

print(f"Loading ReID model backend (using weights: {REID_MODEL_WEIGHTS})")
try:
    # Instantiate ReidAutoBackend
    reid_model_handler = ReidAutoBackend(
        weights=REID_MODEL_WEIGHTS,
        device=DEVICE,
        half=False  # Set to True if you want FP16 inference (check compatibility)
    )
    # The actual backend instance (e.g., PyTorchBackend) is in .model
    reid_backend = reid_model_handler.model
    # Optional: Warmup the ReID model (might need adjustment based on backend)
    if hasattr(reid_backend, 'warmup'):
        reid_backend.warmup()
    print("ReID Model backend loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR loading ReID model: {e}")
    print("Make sure 'boxmot' is installed correctly and weights can be downloaded/found.")
    import traceback

    traceback.print_exc()
    exit()


# --- Helper Functions ---
def detect_and_extract(image_path: str, detector_model, reid_backend_instance) -> np.ndarray | None:
    """
    Detects the most confident person in an image and extracts ReID features
    using the backend's get_features method.

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
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # 1. Detect Persons
    results = detector_model.predict(img_bgr, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("  No persons detected.")
        return None

    # 2. Get the best detection (highest confidence)
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
    confs = results[0].boxes.conf.cpu().numpy()  # Confidences

    best_idx = np.argmax(confs)
    best_box_xyxy = boxes_xyxy[best_idx]  # Keep as float for potential backend use, or .astype(int) if needed
    best_conf = confs[best_idx]
    print(f"  Detected person with confidence {best_conf:.2f} at box {best_box_xyxy.astype(int)}")

    # 3. Extract Re-ID Features using the backend's integrated method
    # Pass the full image and the bounding box(es) as a NumPy array
    # The get_features method in BaseModeBackend handles cropping and preprocessing
    try:
        # Ensure the box is in a NumPy array format, even if it's just one box
        person_bbox_np = np.array([best_box_xyxy])

        # Call get_features on the specific backend instance
        features = reid_backend_instance.get_features(person_bbox_np, img_bgr)

        if features is None or features.size == 0:
            print("  Error: Feature extraction returned None or empty array.")
            return None

        # get_features already returns normalized features for the first (and only) box
        print(f"  Extracted features (shape: {features.shape})")
        return features  # features should already be the (1, D) or (D,) feature vector

    except Exception as e:
        print(f"  Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Calculates cosine similarity between two feature vectors."""
    if feat1 is None or feat2 is None:
        return 0.0
    # Ensure features are 1D arrays for cosine distance calculation
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    if feat1.shape != feat2.shape:
        print(f"Warning: Feature shapes differ: {feat1.shape} vs {feat2.shape}")
        return 0.0
    # Cosine distance = 1 - cosine similarity
    distance = cosine_distance(feat1, feat2)
    similarity = 1.0 - distance
    return float(similarity)


# --- Main Execution ---
features_list = []
for img_path in IMAGE_PATHS:
    # Pass the specific backend instance (reid_backend) to the function
    features = detect_and_extract(img_path, detector, reid_backend)
    features_list.append(features)

if len(features_list) != 3:
    print("\nError: Could not process all three images successfully.")
    # Add more detail if needed:
    for i, f in enumerate(features_list):
        if f is None:
            print(f"Failed to get features for image: {IMAGE_PATHS[i]}")
    exit()

# Check if any feature extraction failed
if None in features_list:
    print("\nError: Feature extraction failed for one or more images. Cannot perform comparisons.")
    exit()

features1, features2, features3 = features_list

# --- Comparisons ---
print("\n--- Re-ID Comparisons ---")

if features1 is not None and features2 is not None:
    similarity_1_2 = calculate_cosine_similarity(features1, features2)
    is_same_1_2 = similarity_1_2 >= REID_SIMILARITY_THRESHOLD
    print(f"Image 1 vs Image 2:")
    print(f"  Cosine Similarity: {similarity_1_2:.4f}")
    print(f"  Result: {'Same Person' if is_same_1_2 else 'Different Person'} (Threshold: {REID_SIMILARITY_THRESHOLD})")
else:
    print("Image 1 vs Image 2: Comparison skipped (missing features).")

if features1 is not None and features3 is not None:
    similarity_1_3 = calculate_cosine_similarity(features1, features3)
    is_same_1_3 = similarity_1_3 >= REID_SIMILARITY_THRESHOLD
    print(f"\nImage 1 vs Image 3:")
    print(f"  Cosine Similarity: {similarity_1_3:.4f}")
    print(f"  Result: {'Same Person' if is_same_1_3 else 'Different Person'} (Threshold: {REID_SIMILARITY_THRESHOLD})")
else:
    print("\nImage 1 vs Image 3: Comparison skipped (missing features).")

print("\nPOC Finished.")
