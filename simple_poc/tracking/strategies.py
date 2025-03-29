import abc
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from ultralytics import YOLO, RTDETR


class DetectionTrackingStrategy(abc.ABC):
    """Abstract base class for object detection and tracking strategies."""

    @abc.abstractmethod
    def __init__(self, model_path: str):
        """Load the specific detection/tracking model."""
        pass

    @abc.abstractmethod
    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        """
        Process a single frame to detect and track objects (specifically persons).

        Args:
            frame: The input frame in BGR format (from OpenCV).

        Returns:
            A tuple containing:
            - List of bounding boxes ([center_x, center_y, width, height]).
            - List of track IDs (integer for tracked objects, -1 for detections without ID).
            - List of confidence scores for each detection/track.
        """
        pass


class YoloStrategy(DetectionTrackingStrategy):
    """Detection and tracking using YOLO models (v8, v9, etc.) via Ultralytics."""

    def __init__(self, model_path: str):
        print(f"Initializing YOLO strategy with model: {model_path}")
        try:
            self.model = YOLO(model_path)
            # Perform a quick inference check on a dummy frame
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_frame, verbose=False)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model '{model_path}': {e}")
            raise  # Re-raise the exception to signal failure

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []
        placeholder_id = -1  # ID for detections without a track

        try:
            # Perform tracking, filtering for 'person' class (index 0 in COCO)
            results = self.model.track(frame, persist=False, classes=0, verbose=False)

            if results and results[0].boxes is not None:
                res_boxes = results[0].boxes
                # Extract results, converting to numpy/list as needed
                boxes_xywh = (
                    res_boxes.xywh.cpu().numpy().tolist()
                    if res_boxes.xywh is not None
                    else []
                )
                confidences = (
                    res_boxes.conf.cpu().numpy().tolist()
                    if res_boxes.conf is not None
                    else []
                )

                # Use tracker IDs if available, otherwise assign placeholder
                if res_boxes.id is not None:
                    track_ids = res_boxes.id.int().cpu().tolist()
                else:
                    track_ids = [placeholder_id] * len(boxes_xywh)

                # Ensure all lists have the same length, truncating if necessary
                min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
                boxes_xywh = boxes_xywh[:min_len]
                track_ids = track_ids[:min_len]
                confidences = confidences[:min_len]

        except Exception as e:
            print(f"Error during YOLO processing: {e}")
            # Return empty lists on error to avoid crashing downstream processing
            return [], [], []

        return boxes_xywh, track_ids, confidences


class RTDetrStrategy(DetectionTrackingStrategy):
    """Detection and tracking using RT-DETR models via Ultralytics."""

    def __init__(self, model_path: str):
        print(f"Initializing RT-DETR strategy with model: {model_path}")
        try:
            self.model = RTDETR(model_path)
            # Perform a quick inference check
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_frame, verbose=False)
            print("RT-DETR model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load RT-DETR model '{model_path}': {e}")
            raise

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []
        placeholder_id = -1

        try:
            # Perform tracking, filtering for 'person' class (index 0)
            results = self.model.track(frame, persist=False, classes=0, verbose=False)

            if results and results[0].boxes is not None:
                res_boxes = results[0].boxes
                boxes_xywh = (
                    res_boxes.xywh.cpu().numpy().tolist()
                    if res_boxes.xywh is not None
                    else []
                )
                confidences = (
                    res_boxes.conf.cpu().numpy().tolist()
                    if res_boxes.conf is not None
                    else []
                )

                if res_boxes.id is not None:
                    track_ids = res_boxes.id.int().cpu().tolist()
                else:
                    track_ids = [placeholder_id] * len(boxes_xywh)

                min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
                boxes_xywh = boxes_xywh[:min_len]
                track_ids = track_ids[:min_len]
                confidences = confidences[:min_len]

        except Exception as e:
            print(f"Error during RT-DETR processing: {e}")
            return [], [], []

        return boxes_xywh, track_ids, confidences


class FasterRCNNStrategy(DetectionTrackingStrategy):
    """Detection using Faster R-CNN (ResNet50 FPN) from TorchVision."""

    def __init__(self, model_path: str):
        # model_path is ignored; using standard TorchVision weights
        print("Initializing Faster R-CNN strategy (using default TorchVision weights)")
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights
            )
            self.model.to(self.device)
            self.model.eval()

            # Get the preprocessing transforms recommended for the model weights
            self.transforms = weights.transforms()
            # COCO class index for 'person' is 1 with default TorchVision weights
            self.person_label_index = 1
            self.score_threshold = (
                0.5  # Minimum confidence score to consider a detection
            )
            self.placeholder_id = -1  # FasterRCNN doesn't provide tracking IDs

            print("Faster R-CNN model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load Faster R-CNN model: {e}")
            raise

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []

        try:
            # --- Preprocessing ---
            # Convert frame from BGR (OpenCV) to RGB
            img_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert NumPy array to PIL Image (required by TorchVision transforms)
            img_pil = Image.fromarray(img_rgb_np)
            # Apply TorchVision transforms (includes normalization, tensor conversion)
            input_tensor = self.transforms(img_pil)
            # Add batch dimension and send to the computation device
            input_batch = [input_tensor.to(self.device)]

            # --- Inference ---
            with torch.no_grad():
                predictions = self.model(input_batch)

            # --- Postprocessing ---
            # Extract predictions for the first (and only) image in the batch
            pred_boxes_xyxy = predictions[0]["boxes"].cpu().numpy()
            pred_labels = predictions[0]["labels"].cpu().numpy()
            pred_scores = predictions[0]["scores"].cpu().numpy()

            for box_xyxy, label, score in zip(
                pred_boxes_xyxy, pred_labels, pred_scores
            ):
                # Filter for 'person' class and confidence threshold
                if label == self.person_label_index and score >= self.score_threshold:
                    x1, y1, x2, y2 = box_xyxy
                    width = x2 - x1
                    height = y2 - y1

                    # Ensure valid box dimensions before converting format
                    if width > 0 and height > 0:
                        center_x = x1 + width / 2
                        center_y = y1 + height / 2
                        boxes_xywh.append([center_x, center_y, width, height])
                        # Faster R-CNN is detection-only, assign placeholder ID
                        track_ids.append(self.placeholder_id)
                        confidences.append(float(score))

        except Exception as e:
            print(f"Error during Faster R-CNN processing step: {e}")
            # import traceback # Uncomment for detailed debugging
            # print(traceback.format_exc()) # Uncomment for detailed debugging
            return [], [], []  # Return empty on error

        return boxes_xywh, track_ids, confidences
