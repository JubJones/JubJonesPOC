import os
import sys
from pathlib import Path
import itertools

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from scipy.spatial.distance import cosine as cosine_distance
from ultralytics import RTDETR

# --- Configuration ---
DETECTOR_MODEL_PATH = "rtdetr-l.pt"
REID_MODEL_WEIGHTS_NAME = "osnet_x0_25_msmt17.pt"

# <<< Use the same image paths as before >>>
IMAGE_PATHS = [
    # "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c1_000217.jpg",
    # "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c1_000352.jpg",
    # "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c2_000000.jpg",
    "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c1_000352.jpg",
    "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c3_000000.jpg",
]

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
REID_SIMILARITY_THRESHOLD = 0.7

# --- Helper Functions ---


def get_device():
    """Selects the best available compute device."""
    # Prioritize MPS if available on Mac, then CUDA, then CPU
    # Note: MPS support might vary depending on torch/boxmot versions
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Add extra check for MPS build availability
        # Some PyTorch versions might report available but not fully functional
        try:
            # Test MPS with a small tensor operation
            _ = torch.tensor([1.0], device="mps") + torch.tensor([1.0], device="mps")
            print("Using Metal Performance Shaders (MPS) device.")
            return torch.device("mps")
        except Exception as e:
            print(f"MPS reported available, but test failed ({e}). Falling back...")

    if torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    else:
        print("Using CPU device.")
        return torch.device("cpu")


def calculate_cosine_similarity(
    feat1: np.ndarray | None, feat2: np.ndarray | None
) -> float:
    """Calculates cosine similarity between two feature vectors."""
    if feat1 is None or feat2 is None:
        # print("Warning: Cannot calculate similarity with None features.") # Less verbose
        return 0.0
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0 or feat2.size == 0:
        # print(f"Warning: Feature shapes invalid or empty: {feat1.shape} vs {feat2.shape}") # Less verbose
        return 0.0
    if np.all(feat1 == 0) or np.all(feat2 == 0):
        # print("Warning: One or both feature vectors are all zeros.") # Less verbose
        return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all():
        # print("Warning: Feature vector contains NaN or Inf values.") # Less verbose
        return 0.0

    try:
        distance = cosine_distance(feat1, feat2)
        distance = max(0.0, float(distance))  # Ensure non-negative float
    except Exception as e:
        # print(f"Error calculating cosine distance: {e}") # Less verbose
        return 0.0

    similarity = 1.0 - distance
    return float(np.clip(similarity, 0.0, 1.0))


def detect_persons_and_extract_features(
    image_path: str, detector_model, reid_backend_instance
) -> list[dict]:
    """
    Detects ALL persons in an image above a threshold and extracts ReID features.

    Args:
        image_path: Path to the input image.
        detector_model: Loaded RTDETR model.
        reid_backend_instance: The specific ReID backend instance.

    Returns:
        A list of dictionaries. Each dictionary represents a detected person
        and contains {'bbox': [x1, y1, x2, y2], 'conf': float, 'feature': np.ndarray}.
        Returns an empty list if no persons are detected or an error occurs.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return []

    print(f"\nProcessing image: {os.path.basename(image_path)}")
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Could not read image {image_path} with OpenCV.")
            return []
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return []

    # 1. Detect ALL Persons
    detected_persons = []
    try:
        results = detector_model.predict(
            img_bgr,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            print("  No persons detected meeting the confidence threshold.")
            return []

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        print(f"  Detected {len(boxes_xyxy)} persons.")

        # 2. Extract Features for ALL Detected Persons
        if len(boxes_xyxy) > 0:
            try:
                # Pass all detected bounding boxes to the ReID backend
                # Note: Ensure boxes are float32 for some backends if needed, though numpy default is float64
                features = reid_backend_instance.get_features(
                    boxes_xyxy.astype(np.float32), img_bgr
                )

                if features is None or features.shape[0] != len(boxes_xyxy):
                    print(
                        f"  Error: Feature extraction failed or returned unexpected shape ({features.shape if features is not None else 'None'}). Expected {len(boxes_xyxy)} features."
                    )
                    return []  # Or handle partial failure if desired

                print(
                    f"  Extracted features for {features.shape[0]} persons (shape: {features.shape})"
                )

                # Check features for NaN/Inf
                if not np.isfinite(features).all():
                    print(
                        "  Error: Extracted features contain NaN or Inf values. Skipping this image."
                    )
                    return []

                # Store data for each person
                for i in range(len(boxes_xyxy)):
                    detected_persons.append(
                        {
                            "bbox": boxes_xyxy[i].astype(
                                int
                            ),  # Store as int for drawing
                            "conf": float(confs[i]),
                            "feature": features[i],
                        }
                    )

            except Exception as e:
                print(f"  Error during batch feature extraction: {e}")
                import traceback

                traceback.print_exc()
                return []  # Critical error during extraction

    except Exception as e:
        print(f"  Error during object detection: {e}")
        return []

    return detected_persons


def plot_comparison(
    img_path1,
    person_data1,
    person_index1,
    img_path2,
    person_data2,
    person_index2,
    similarity,
    threshold,
):
    """
    Plots two images side-by-side, highlighting the compared persons and showing similarity.
    """
    try:
        img1_bgr = cv2.imread(img_path1)
        img2_bgr = cv2.imread(img_path2)
        if img1_bgr is None or img2_bgr is None:
            print("Error: Could not read images for plotting.")
            return

        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes
        box1 = person_data1["bbox"]
        box2 = person_data2["bbox"]
        color = (
            (0, 255, 0) if similarity >= threshold else (255, 0, 0)
        )  # Green if same, Red if different
        thickness = 2

        cv2.rectangle(
            img1_rgb, (box1[0], box1[1]), (box1[2], box1[3]), color, thickness
        )
        cv2.rectangle(
            img2_rgb, (box2[0], box2[1]), (box2[2], box2[3]), color, thickness
        )

        # Add text label near the box
        label1 = f"P{person_index1 + 1}"  # Person index (1-based)
        label2 = f"P{person_index2 + 1}"
        cv2.putText(
            img1_rgb,
            label1,
            (box1[0], box1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            thickness,
        )
        cv2.putText(
            img2_rgb,
            label2,
            (box2[0], box2[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            thickness,
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjusted figsize

        ax1.imshow(img1_rgb)
        ax1.set_title(f"{os.path.basename(img_path1)}\nPerson {person_index1 + 1}")
        ax1.axis("off")

        ax2.imshow(img2_rgb)
        ax2.set_title(f"{os.path.basename(img_path2)}\nPerson {person_index2 + 1}")
        ax2.axis("off")

        result_text = "Same Person" if similarity >= threshold else "Different Person"
        fig.suptitle(
            f"Comparison Result: {result_text}\nCosine Similarity: {similarity:.4f} (Threshold: {threshold})",
            fontsize=14,
            y=0.98,
        )  # Adjusted y position

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to prevent title overlap
        plt.show()

    except Exception as e:
        print(f"Error during plotting comparison: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Prerequisite Checks ---
    image_files_exist = True
    for p in IMAGE_PATHS:
        if not os.path.exists(p):
            print(f"FATAL ERROR: Missing image file: {p}")
            image_files_exist = False
    if not image_files_exist:
        print("Please ensure all image paths listed in IMAGE_PATHS are correct.")
        sys.exit(1)

    # --- Device Selection ---
    DEVICE = get_device()

    # --- Model Loading ---
    print(f"\n--- Loading Models ---")
    # Load Detector
    try:
        print(f"Loading RTDETR detector from: {DETECTOR_MODEL_PATH}")
        detector = RTDETR(DETECTOR_MODEL_PATH)
        detector.to(DEVICE)
        _ = detector.predict(
            np.zeros((640, 640, 3), dtype=np.uint8), verbose=False
        )  # Warmup
        print("RTDETR Detector loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR loading detector: {e}")
        sys.exit(1)

    # Load ReID Model
    try:
        print(f"Loading ReID model backend (using weights: {REID_MODEL_WEIGHTS_NAME})")
        reid_weights_path = Path(REID_MODEL_WEIGHTS_NAME)
        reid_model_handler = ReidAutoBackend(
            weights=reid_weights_path, device=DEVICE, half=False
        )
        reid_backend = reid_model_handler.model
        if hasattr(reid_backend, "warmup"):
            print("Warming up ReID model...")
            reid_backend.warmup()
        print("ReID Model backend loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR loading ReID model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # --- Feature Extraction for All Images ---
    print("\n--- Extracting Features from All Images ---")
    all_image_data = {}
    for img_path in IMAGE_PATHS:
        person_data = detect_persons_and_extract_features(
            img_path, detector, reid_backend
        )
        if person_data:  # Only add if detection/extraction was successful
            all_image_data[img_path] = person_data
        else:
            print(
                f"Warning: No persons extracted for {os.path.basename(img_path)}. It will be skipped in comparisons."
            )

    # --- Comparisons and Plotting ---
    print("\n--- Performing Pairwise Comparisons ---")

    # Use itertools.combinations to get unique pairs of image paths
    image_pairs = list(itertools.combinations(IMAGE_PATHS, 2))

    if not image_pairs:
        print("Need at least two images with successful detections to compare.")
        sys.exit(0)

    for img_path1, img_path2 in image_pairs:
        base_name1 = os.path.basename(img_path1)
        base_name2 = os.path.basename(img_path2)
        print(f"\n--- Comparing [{base_name1}] vs [{base_name2}] ---")

        # Retrieve data, defaulting to empty list if extraction failed for an image
        persons1 = all_image_data.get(img_path1, [])
        persons2 = all_image_data.get(img_path2, [])

        if not persons1:
            print(f"  Skipping: No persons detected/extracted in {base_name1}.")
            continue
        if not persons2:
            print(f"  Skipping: No persons detected/extracted in {base_name2}.")
            continue

        print(
            f"  Comparing {len(persons1)} persons from {base_name1} with {len(persons2)} persons from {base_name2}"
        )

        # Nested loop for pairwise comparison between persons in the two images
        for i, p1_data in enumerate(persons1):
            for j, p2_data in enumerate(persons2):
                similarity = calculate_cosine_similarity(
                    p1_data["feature"], p2_data["feature"]
                )

                print(
                    f"  - {base_name1} Person {i + 1} vs {base_name2} Person {j + 1}: Similarity = {similarity:.4f} {'[SAME]' if similarity >= REID_SIMILARITY_THRESHOLD else '[DIFF]'} "
                )

                # Plot the comparison
                plot_comparison(
                    img_path1,
                    p1_data,
                    i,
                    img_path2,
                    p2_data,
                    j,
                    similarity,
                    REID_SIMILARITY_THRESHOLD,
                )

    print("\n--- POC Finished ---")
