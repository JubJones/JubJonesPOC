import abc
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from ultralytics import YOLO, RTDETR


class DetectionTrackingStrategy(abc.ABC):
    """Abstract base class for detection and tracking strategies."""

    @abc.abstractmethod
    def __init__(self, model_path: str):
        """Load the model."""
        pass

    @abc.abstractmethod
    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """
        Process a single frame to detect and track objects.

        Args:
            frame: The input frame in BGR format.

        Returns:
            A tuple containing:
            - List of bounding boxes (xywh format: [center_x, center_y, width, height]).
            - List of track IDs (or placeholders if not tracking).
            - List of confidence scores.
        """
        pass


class YoloStrategy(DetectionTrackingStrategy):
    """Detection and tracking using YOLO models from Ultralytics."""

    def __init__(self, model_path: str):
        print(f"Initializing YOLO strategy with model: {model_path}")
        try:
            self.model = YOLO(model_path)
            # Perform a dummy inference to check model loading
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_frame, verbose=False)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []

        try:
            # classes=0 filters for the 'person' class assuming COCO training
            results = self.model.track(frame, persist=True, classes=0, verbose=False)

            if results and results[0].boxes is not None:
                boxes_xywh = results[0].boxes.xywh.cpu().numpy().tolist() if results[0].boxes.xywh is not None else []
                confidences = results[0].boxes.conf.cpu().numpy().tolist() if results[0].boxes.conf is not None else []

                # Tracking IDs might be None if no tracks are established yet or model doesn't support tracking well
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    # Assign placeholder IDs if tracking fails/not available
                    track_ids = [-1] * len(boxes_xywh)

                # Ensure all lists have the same length
                min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
                boxes_xywh = boxes_xywh[:min_len]
                track_ids = track_ids[:min_len]
                confidences = confidences[:min_len]

        except Exception as e:
            print(f"Error during YOLO processing: {e}")
            # Return empty lists on error

        return boxes_xywh, track_ids, confidences


class RTDetrStrategy(DetectionTrackingStrategy):
    """Detection and tracking using RT-DETR models from Ultralytics."""

    def __init__(self, model_path: str):
        print(f"Initializing RT-DETR strategy with model: {model_path}")
        try:
            self.model = RTDETR(model_path)
             # Perform a dummy inference to check model loading
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_frame, verbose=False)
            print("RT-DETR model loaded successfully.")
        except Exception as e:
            print(f"Error loading RT-DETR model: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []

        try:
            results = self.model.track(frame, persist=True, classes=0, verbose=False)

            if results and results[0].boxes is not None:
                boxes_xywh = results[0].boxes.xywh.cpu().numpy().tolist() if results[0].boxes.xywh is not None else []
                confidences = results[0].boxes.conf.cpu().numpy().tolist() if results[0].boxes.conf is not None else []

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    track_ids = [-1] * len(boxes_xywh) # Placeholder IDs

                min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
                boxes_xywh = boxes_xywh[:min_len]
                track_ids = track_ids[:min_len]
                confidences = confidences[:min_len]

        except Exception as e:
            print(f"Error during RT-DETR processing: {e}")

        return boxes_xywh, track_ids, confidences


class FasterRCNNStrategy(DetectionTrackingStrategy):
    """Detection using Faster R-CNN from TorchVision."""

    def __init__(self, model_path: str):
        # model_path might be ignored if using default weights, but kept for consistency
        print(f"Initializing Faster R-CNN strategy (using default TorchVision weights, path '{model_path}' ignored)")
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            self.transforms = weights.transforms() # Get the transforms associated with the weights
            self.person_label_index = 1 # In COCO dataset used by TorchVision default weights
            self.score_threshold = 0.5
            print("Faster R-CNN model loaded successfully.")
        except Exception as e:
            print(f"Error loading Faster R-CNN model: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []

        try:
            # 1. Preprocess
            # Convert BGR (OpenCV) to RGB NumPy array
            img_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert RGB NumPy array to PIL Image  <--- *** FIXED ***
            img_pil = Image.fromarray(img_rgb_np)

            # Apply torchvision transforms (which expects a PIL Image)
            input_tensor = self.transforms(img_pil) # <--- *** FIXED ***

            # Add batch dimension and send to device
            input_batch = [input_tensor.to(self.device)]

            # 2. Inference
            with torch.no_grad():
                predictions = self.model(input_batch)

            # 3. Postprocess
            pred_boxes = predictions[0]['boxes'].cpu().numpy()
            pred_labels = predictions[0]['labels'].cpu().numpy()
            pred_scores = predictions[0]['scores'].cpu().numpy()

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if label == self.person_label_index and score >= self.score_threshold:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    if w > 0 and h > 0:
                        boxes_xywh.append([cx, cy, w, h])
                        track_ids.append(-1)
                        confidences.append(float(score))

        except Exception as e:
            # Print the specific error during processing step
            print(f"Error during Faster R-CNN processing step: {e}")
            # import traceback # Uncomment for detailed traceback during debugging
            # print(traceback.format_exc()) # Uncomment for detailed traceback

        return boxes_xywh, track_ids, confidences
