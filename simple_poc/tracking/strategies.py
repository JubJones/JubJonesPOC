# ================================================
# FILE: simple_poc/tracking/strategies.py
# ================================================
import abc
from typing import List, Tuple

import cv2
import numpy as np
import torch # Ensure torch is imported
import torchvision
from PIL import Image
# from sympy.printing.tree import print_node # <--- Remove this unused import if present
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from ultralytics import YOLO, RTDETR
from rfdetr import RFDETRLarge # Added import for RF-DETR


# --- (Other classes remain the same: DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy) ---


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


class RfDetrStrategy(DetectionTrackingStrategy):
    """Detection using RF-DETR base model from the rfdetr library."""

    def __init__(self, model_path: str):
        # model_path is currently ignored; RFDETRBase loads its default weights.
        print(
            f"Initializing RF-DETR strategy (model_path '{model_path}' ignored, using RFDETRLarge defaults)"
        )
        try:
            print("Attempting to load RFDETRLarge...")
            # --- Attempt to force CPU ---
            # Explicitly request CPU device if MPS fallback/native fails
            target_device = 'cpu'
            print(f"Attempting to load RFDETRLarge and force execution on device: '{target_device}'")
            self.model = RFDETRLarge() # Initialize first
            print("RFDETRLarge object created. Attempting to move model to CPU...")
            try:
                self.model.to(target_device)
                if hasattr(self.model, 'model') and isinstance(self.model.model, torch.nn.Module):
                    self.model.model.to(target_device)
                print(f"Attempted to move model components to '{target_device}'.")
                print(f"Performing dummy inference check on '{target_device}'...")
                dummy_pil = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
                _ = self.model.predict(dummy_pil)
                print("Dummy inference check on CPU successful.")
            except Exception as move_exc:
                 print(f"Warning: Error attempting to move model to CPU: {move_exc}.")
            # --- End Force CPU ---

            print("RF-DETR model initialization proceeding...")

            self.person_label_index = 1
            print(f"Set Person Class Index to: {self.person_label_index}")

            self.score_threshold = (
                0.5  # Minimum confidence score, same as reference
            )
            self.placeholder_id = -1  # RF-DETR predict is detection-only
            print(f"RF-DETR strategy initialized, intended device: '{target_device}'.")


        except ImportError:
             print("ERROR: Critical - Failed to import RFDETRLarge. Is 'rfdetr' library installed correctly?")
             raise
        except Exception as e:
            # Catching the original MPS error here if forcing CPU failed or wasn't possible
            print(f"ERROR: Critical - Failed during RFDETRLarge initialization or CPU forcing: {e}")
            raise # Re-throw the exception to be caught by PersonTracker


    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []

        if not hasattr(self, 'model') or self.model is None:
             print("ERROR: RF-DETR model not initialized in process_frame.")
             return [], [], []

        try:
            # --- Preprocessing ---
            # print("Preprocessing RFDETR") # Optional: Keep if needed
            img_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb_np)

            # --- Inference ---
            # print("Running RFDETR prediction") # Optional: Keep if needed
            detections = self.model.predict(img_pil, threshold=self.score_threshold)

            # --- Postprocessing ---
            # print("Postprocessing RFDETR") # Optional: Keep if needed
            if detections:
                # print(f"Raw Detections found: Count={len(detections)}") # Optional: More detailed log
                pred_boxes_xyxy = detections.xyxy
                pred_labels = detections.class_id
                pred_scores = detections.confidence

                detection_count_in_frame = 0 # Counter for this frame

                for box_xyxy, label, score in zip(
                    pred_boxes_xyxy, pred_labels, pred_scores
                ):
                    detection_count_in_frame += 1
                    # <<< ADD MORE LOGGING >>>
                    print(f"  [RFDETR Detection {detection_count_in_frame}] Raw Label: {label}, Score: {score:.4f}") # Log raw label and score

                    # Filter for 'person' class (NOW USING INDEX 1) and confidence threshold
                    if label == self.person_label_index:
                        print(f"    -> MATCHED Person Index ({self.person_label_index})! Score: {score:.4f}")
                        if score >= self.score_threshold:
                            print(f"      -> PASSED Threshold ({self.score_threshold})! ADDING BOX.")
                            x1, y1, x2, y2 = box_xyxy
                            width = x2 - x1
                            height = y2 - y1

                            if width > 0 and height > 0:
                                center_x = x1 + width / 2
                                center_y = y1 + height / 2
                                boxes_xywh.append([center_x, center_y, width, height])
                                track_ids.append(self.placeholder_id)
                                confidences.append(float(score))
                            else:
                                print(f"      -> SKIPPED Box (Invalid Dimensions: w={width}, h={height})")
                        else:
                            print(f"      -> FAILED Threshold ({self.score_threshold}).")
                    # <<< END LOGGING >>>

            # else: # Optional log if no detections at all
                # print("No raw detections returned by model.predict().")

        except Exception as e:
            print(f"Error during RF-DETR processing step: {e}")
            import traceback # Add traceback for detailed errors
            print(traceback.format_exc())
            return [], [], []

        # Log final counts for this frame
        print(f"  -> RFDETR Frame Processed. Found {len(boxes_xywh)} persons passing filters.")
        return boxes_xywh, track_ids, confidences