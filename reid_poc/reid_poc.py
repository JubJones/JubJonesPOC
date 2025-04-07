import itertools
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from scipy.spatial.distance import cosine as cosine_distance
from ultralytics import RTDETR

# --- Configuration ---
DETECTOR_MODEL_PATH = "rtdetr-l.pt"

# Define the ReID models to use
REID_CONFIGS = [
    {"name": "OSNet", "weights": "osnet_x0_25_msmt17.pt", "active": True},
    {"name": "CLIP", "weights": "clip_market1501.pt", "active": True},
]

# Filter for active models
ACTIVE_REID_CONFIGS = [config for config in REID_CONFIGS if config.get("active", True)]
if not ACTIVE_REID_CONFIGS:
    print("FATAL ERROR: No active ReID models configured in REID_CONFIGS.")
    sys.exit(1)

REID_MODEL_NAMES = [config["name"] for config in ACTIVE_REID_CONFIGS]

IMAGE_PATHS = [
    "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c1_000217.jpg",
    "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c1_000352.jpg",
    # "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c1_000352.jpg",
    # "/Users/krittinsetdhavanich/Downloads/JubJonesPOC/reid_poc/test_images/campus/s47/c3_000000.jpg",
]

PERSON_CLASS_ID = 0  # COCO class ID for 'person'
CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence
REID_SIMILARITY_THRESHOLD = 0.7  # Threshold applied to similarity scores


# --- Helper Functions ---


def get_device():
    """Selects the best available compute device."""
    # Prioritize MPS if available on Mac, then CUDA, then CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
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
        return 0.0
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    if feat1.shape != feat2.shape or feat1.size == 0 or feat2.size == 0:
        return 0.0
    if np.all(feat1 == 0) or np.all(feat2 == 0):
        return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all():
        return 0.0

    try:
        # Ensure inputs are float64 for scipy's cosine distance
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        distance = max(0.0, float(distance))
    except Exception as e:
        # print(f"Error calculating cosine distance: {e}") # Less verbose
        return 0.0

    similarity = 1.0 - distance
    return float(np.clip(similarity, 0.0, 1.0))


def detect_persons(image_path: str, detector_model) -> list[dict]:
    """
    Detects ALL persons in an image above a threshold.

    Args:
        image_path: Path to the input image.
        detector_model: Loaded RTDETR model.

    Returns:
        A list of dictionaries. Each dictionary represents a detected person
        and contains {'bbox': [x1, y1, x2, y2], 'conf': float}.
        Returns an empty list if no persons are detected or an error occurs.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return []

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Could not read image {image_path} with OpenCV.")
            return []
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return []

    # Detect ALL Persons
    detected_persons_info = []
    try:
        results = detector_model.predict(
            img_bgr,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for i in range(len(boxes_xyxy)):
                detected_persons_info.append(
                    {"bbox": boxes_xyxy[i].astype(int), "conf": float(confs[i])}
                )

    except Exception as e:
        print(f"  Error during object detection in {os.path.basename(image_path)}: {e}")
        return []

    return detected_persons_info


def plot_comparison(
    img_path1,
    person_data1,
    person_index1,
    img_path2,
    person_data2,
    person_index2,
    similarities: dict,  # Changed: Accept dict of similarities
    threshold: float,
    model_names: list,  # Added: List of model names used
):
    """
    Plots two images side-by-side, highlighting the compared persons and showing similarity for multiple models.
    """
    fig = None
    try:
        img1_bgr = cv2.imread(img_path1)
        img2_bgr = cv2.imread(img_path2)
        if img1_bgr is None or img2_bgr is None:
            print(
                f"Error: Could not read images for plotting: {img_path1} or {img_path2}"
            )
            return

        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

        # Determine box color - Use the first active model's result for consistency
        primary_model_name = model_names[0]
        primary_similarity = similarities.get(primary_model_name, 0.0)
        color = (
            (0, 255, 0) if primary_similarity >= threshold else (255, 0, 0)
        )  # Green if primary model says same, Red otherwise
        thickness = 2

        # Draw boxes and labels
        box1 = person_data1["bbox"]
        box2 = person_data2["bbox"]
        cv2.rectangle(
            img1_rgb, (box1[0], box1[1]), (box1[2], box1[3]), color, thickness
        )
        cv2.rectangle(
            img2_rgb, (box2[0], box2[1]), (box2[2], box2[3]), color, thickness
        )
        label1 = f"P{person_index1 + 1}"
        label2 = f"P{person_index2 + 1}"
        # Put text slightly inside the box if near top edge
        text_y1 = box1[1] - 10 if box1[1] > 20 else box1[1] + 20
        text_y2 = box2[1] - 10 if box2[1] > 20 else box2[1] + 20
        cv2.putText(
            img1_rgb,
            label1,
            (box1[0], text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            thickness,
        )
        cv2.putText(
            img2_rgb,
            label2,
            (box2[0], text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            thickness,
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))  # Slightly taller figsize

        ax1.imshow(img1_rgb)
        ax1.set_title(f"{os.path.basename(img_path1)}\nPerson {person_index1 + 1}")
        ax1.axis("off")
        ax2.imshow(img2_rgb)
        ax2.set_title(f"{os.path.basename(img_path2)}\nPerson {person_index2 + 1}")
        ax2.axis("off")

        # Create multi-line title with results for all models
        title_lines = []
        for name in model_names:
            sim = similarities.get(name, None)
            if sim is None:
                result_str = f"{name}: Error/Skipped"
            else:
                result = "Same Person" if sim >= threshold else "Different Person"
                result_str = f"{name}: {sim:.4f} ({result})"
            title_lines.append(result_str)

        fig.suptitle(
            "Comparison Results (Threshold: {})\n".format(threshold)
            + "\n".join(title_lines),
            fontsize=12,
            y=0.99,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout
        plt.show()

    except Exception as e:
        print(f"Error during plotting comparison: {e}")
        if fig is not None:
            plt.close(fig)


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
    if len(IMAGE_PATHS) < 2:
        print("FATAL ERROR: Need at least two images in IMAGE_PATHS to compare.")
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

    # Load ReID Models
    reid_backends = {}
    print(f"\n--- Loading ReID Models ({', '.join(REID_MODEL_NAMES)}) ---")
    for config in ACTIVE_REID_CONFIGS:
        model_name = config["name"]
        weights_identifier = config["weights"]
        print(f"Loading {model_name} (weights: {weights_identifier})...")
        try:
            weights_path_obj = Path(weights_identifier)

            reid_model_handler = ReidAutoBackend(
                weights=weights_path_obj, device=DEVICE, half=False
            )
            backend = reid_model_handler.model
            if hasattr(backend, "warmup"):
                print(f"Warming up {model_name}...")
                # warmup model by running inference once
                backend.warmup()
            reid_backends[model_name] = backend
            print(f"{model_name} ReID Model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Failed loading {model_name} ReID model: {e}")
            print(f"Skipping {model_name} due to loading error.")

    if not reid_backends:
        print("FATAL ERROR: No ReID models were loaded successfully.")
        sys.exit(1)

    # Update list of names based on successful loads
    ACTIVE_REID_MODEL_NAMES = list(reid_backends.keys())
    print(f"Active ReID models for processing: {', '.join(ACTIVE_REID_MODEL_NAMES)}")

    # --- Feature Extraction for All Images ---
    print("\n--- Detecting Persons and Extracting Features from All Images ---")
    all_image_data = {}
    processing_times = {}  # Stores batch extraction times per image/model

    for img_path in IMAGE_PATHS:
        img_basename = os.path.basename(img_path)
        print(f"\nProcessing image: {img_basename}")

        # 1. Detect Persons
        detected_persons_basic_info = detect_persons(img_path, detector)

        if not detected_persons_basic_info:
            print(
                f"  No persons detected in {img_basename}. Skipping feature extraction."
            )
            all_image_data[img_path] = []  # Mark as processed but no persons
            continue
        else:
            print(f"  Detected {len(detected_persons_basic_info)} persons.")

        # Prepare for feature extraction
        all_boxes_xyxy = np.array([p["bbox"] for p in detected_persons_basic_info])
        persons_processed_data = []
        batch_extraction_times = {}
        image_read_successful = (
            False  # Flag to ensure image is read before extraction loop
        )

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError(
                    f"Failed to read image {img_basename} for feature extraction."
                )
            image_read_successful = True

            # 2. Extract features for each ReID model
            features_per_model_per_person = [
                {} for _ in range(len(detected_persons_basic_info))
            ]

            for model_name, reid_backend in reid_backends.items():
                print(f"  Extracting features using {model_name}...")
                t_start = time.time()
                model_features = None  # Initialize features for this model
                try:
                    # Extract features for all boxes in the image at once
                    model_features = reid_backend.get_features(
                        all_boxes_xyxy.astype(np.float32), img_bgr
                    )
                    t_end = time.time()
                    batch_duration = t_end - t_start
                    batch_extraction_times[model_name] = batch_duration
                    print(
                        f"    Extracted {model_features.shape[0] if model_features is not None else 'None'} features in {batch_duration:.4f} seconds."
                    )

                    # --- Validation Checks ---
                    if model_features is None or model_features.shape[0] != len(
                        detected_persons_basic_info
                    ):
                        print(
                            f"    Error: {model_name} feature extraction mismatch or failure. Expected {len(detected_persons_basic_info)}, got {model_features.shape[0] if model_features is not None else 'None'}. Storing None for this model."
                        )
                        batch_extraction_times[model_name] = (
                            None  # Mark time as invalid
                        )
                        # Assign None to all persons for this model
                        for i in range(len(detected_persons_basic_info)):
                            features_per_model_per_person[i][model_name] = None
                        continue  # Go to next model

                    if not np.isfinite(model_features).all():
                        print(
                            f"    Error: {model_name} features contain NaN/Inf. Storing None for this model."
                        )
                        batch_extraction_times[model_name] = (
                            None  # Mark time as invalid
                        )
                        for i in range(len(detected_persons_basic_info)):
                            features_per_model_per_person[i][model_name] = None
                        continue  # Go to next model

                    # --- Assign features ---
                    for i in range(len(detected_persons_basic_info)):
                        features_per_model_per_person[i][model_name] = model_features[i]

                except Exception as e:
                    print(f"    ERROR during {model_name} feature extraction: {e}")
                    batch_extraction_times[model_name] = None  # Mark time as invalid
                    # Assign None to all persons for this model if extraction failed
                    for i in range(len(detected_persons_basic_info)):
                        features_per_model_per_person[i][model_name] = None

            # 3. Assemble final data structure for the image
            any_successful_extraction = False
            for i, basic_info in enumerate(detected_persons_basic_info):
                # Check if at least one model successfully extracted features for *this* person
                person_has_valid_feature = any(
                    feat is not None
                    for feat in features_per_model_per_person[i].values()
                )

                if person_has_valid_feature:
                    persons_processed_data.append(
                        {
                            "bbox": basic_info["bbox"],
                            "conf": basic_info["conf"],
                            "features": features_per_model_per_person[
                                i
                            ],  # Dict of features by model name
                        }
                    )
                    any_successful_extraction = True
                else:
                    print(
                        f"    Skipping Person {i + 1} in {img_basename} - feature extraction failed for all models."
                    )

            if any_successful_extraction:
                all_image_data[img_path] = persons_processed_data
                processing_times[img_path] = batch_extraction_times  # Store batch times
                print(
                    f"  Successfully processed {len(persons_processed_data)} persons with features in {img_basename}."
                )
            else:
                print(
                    f"  Warning: Feature extraction failed for all detected persons/models in {img_basename}. Skipping this image."
                )
                all_image_data[img_path] = []  # Indicate failed extraction overall

        except Exception as e:
            print(
                f"  Error processing image {img_basename} for feature extraction: {e}"
            )
            all_image_data[img_path] = []  # Indicate failure

    # --- Comparisons and Plotting ---
    print("\n--- Performing Pairwise Comparisons ---")
    image_pairs = list(itertools.combinations(IMAGE_PATHS, 2))

    if not image_pairs:
        print("Need at least two images processed successfully to compare.")
        sys.exit(0)

    total_comparisons = 0
    # Initialize comparison timing dict using only successfully loaded models
    model_comparison_times = {name: 0.0 for name in ACTIVE_REID_MODEL_NAMES}

    for img_path1, img_path2 in image_pairs:
        base_name1 = os.path.basename(img_path1)
        base_name2 = os.path.basename(img_path2)
        print(f"\n--- Comparing [{base_name1}] vs [{base_name2}] ---")

        # Report extraction times for these images (if available)
        times1 = processing_times.get(img_path1, {})
        times2 = processing_times.get(img_path2, {})
        if times1 or times2:  # Only print if times were recorded
            print("  Batch Feature Extraction Times (seconds):")
            for name in ACTIVE_REID_MODEL_NAMES:
                t1 = times1.get(name, "N/A")
                t2 = times2.get(name, "N/A")
                # Format time only if it's a valid float
                t1_str = f"{t1:.4f}" if isinstance(t1, float) else str(t1)
                t2_str = f"{t2:.4f}" if isinstance(t2, float) else str(t2)
                print(f"    {name}: {base_name1}={t1_str}, {base_name2}={t2_str}")
        else:
            print(
                "  Batch Feature Extraction Times: Not available (extraction may have failed)."
            )

        persons1 = all_image_data.get(img_path1, [])
        persons2 = all_image_data.get(img_path2, [])

        if not persons1 or not persons2:
            print(
                f"  Skipping comparison: No valid persons with features available in one or both images ('{base_name1}': {len(persons1)} persons, '{base_name2}': {len(persons2)} persons)."
            )
            continue

        print(
            f"  Comparing {len(persons1)} persons from {base_name1} with {len(persons2)} persons from {base_name2}"
        )

        for i, p1_data in enumerate(persons1):
            for j, p2_data in enumerate(persons2):
                total_comparisons += len(ACTIVE_REID_MODEL_NAMES)
                similarities = {}
                results_text = []

                # Compare using each active model
                for model_name in ACTIVE_REID_MODEL_NAMES:
                    t_sim_start = time.time()
                    feature1 = p1_data["features"].get(model_name)
                    feature2 = p2_data["features"].get(model_name)

                    # Calculate similarity only if both features are valid numpy arrays
                    if isinstance(feature1, np.ndarray) and isinstance(
                        feature2, np.ndarray
                    ):
                        sim = calculate_cosine_similarity(feature1, feature2)
                        similarities[model_name] = sim
                        is_same = sim >= REID_SIMILARITY_THRESHOLD
                        results_text.append(
                            f"{model_name}={sim:.3f} {'[SAME]' if is_same else '[DIFF]'}"
                        )
                    else:
                        similarities[model_name] = None
                        results_text.append(f"{model_name}=N/A [SKIP]")

                    t_sim_end = time.time()
                    # Add time only if calculation was performed
                    if isinstance(feature1, np.ndarray) and isinstance(
                        feature2, np.ndarray
                    ):
                        model_comparison_times[model_name] += t_sim_end - t_sim_start

                print(
                    f"  - {base_name1} P{i + 1} vs {base_name2} P{j + 1}: {' | '.join(results_text)}"
                )

                # Plot the comparison, passing the dictionary of similarities
                plot_comparison(
                    img_path1,
                    p1_data,
                    i,
                    img_path2,
                    p2_data,
                    j,
                    similarities,
                    REID_SIMILARITY_THRESHOLD,
                    ACTIVE_REID_MODEL_NAMES,
                )

    print("\n--- POC Finished ---")
