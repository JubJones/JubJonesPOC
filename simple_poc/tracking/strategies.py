import abc
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from rfdetr import RFDETRLarge
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from ultralytics import YOLO, RTDETR


FORCE_DEVICE = "auto"  # Change this to "cuda", "mps", or "auto" as needed
# FORCE_DEVICE = "cpu"  # Change this to "cuda", "mps", or "auto" as needed


def get_selected_device() -> torch.device:
    """
    Gets the torch.device based on FORCE_DEVICE flag and availability.
    Prioritizes CUDA > MPS > CPU for "auto".
    """
    print(f"--- Determining Device (FORCE_DEVICE='{FORCE_DEVICE}') ---")

    if FORCE_DEVICE.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                device = torch.device(FORCE_DEVICE)
                device_name = torch.cuda.get_device_name(device)
                print(f"Selected device: {device} ({device_name})")
                return device
            except Exception as e:
                print(
                    f"WARNING: Requested CUDA device '{FORCE_DEVICE}' not valid or available ({e}). Falling back to CPU."
                )
                return torch.device("cpu")
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    elif FORCE_DEVICE == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Selected device: {device}")
            return device
        else:
            print("WARNING: MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    elif FORCE_DEVICE == "cpu":
        print("Selected device: cpu")
        return torch.device("cpu")

    elif FORCE_DEVICE == "auto":
        print("Attempting auto-detection: CUDA > MPS > CPU")
        # 1. Try CUDA
        if torch.cuda.is_available():
            try:
                # Usually defaults to cuda:0. Let PyTorch handle default.
                device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(device)
                print(f"Auto-selected device: {device} ({device_name})")
                return device
            except Exception as e:
                # This might happen if CUDA is available but the default device has issues
                print(
                    f"WARNING: CUDA available but failed to initialize ({e}). Checking MPS."
                )
        else:
            print("CUDA not available.")

        # 2. Try MPS (if CUDA not available or failed)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Auto-selected device: {device}")
            return device
        else:
            print("MPS not available.")

        # 3. Fallback to CPU
        device = torch.device("cpu")
        print(f"Auto-selected device: {device}")
        return device

    else:
        print(
            f"WARNING: Unknown FORCE_DEVICE value '{FORCE_DEVICE}'. Falling back to CPU."
        )
        return torch.device("cpu")


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
        self.selected_device = get_selected_device()
        print(
            f"Initializing YOLO strategy with model: {model_path} on device: {self.selected_device}"
        )
        try:
            self.model = YOLO(model_path)
            self.model.to(self.selected_device)

            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

            self.model.predict(dummy_frame, device=self.selected_device, verbose=False)
            print(f"YOLO model loaded successfully onto {self.selected_device}.")
        except Exception as e:
            print(
                f"ERROR: Failed to load YOLO model '{model_path}' onto {self.selected_device}: {e}"
            )
            raise

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []
        placeholder_id = -1

        try:
            results = self.model.track(
                frame,
                persist=False,
                classes=0,
                device=self.selected_device,
                verbose=False,
            )

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
            print(f"Error during YOLO processing: {e}")

            return [], [], []

        return boxes_xywh, track_ids, confidences


class RTDetrStrategy(DetectionTrackingStrategy):
    """Detection and tracking using RT-DETR models via Ultralytics."""

    def __init__(self, model_path: str):
        self.selected_device = get_selected_device()
        print(
            f"Initializing RT-DETR strategy with model: {model_path} on device: {self.selected_device}"
        )
        try:
            self.model = RTDETR(model_path)
            self.model.to(self.selected_device)

            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_frame, device=self.selected_device, verbose=False)
            print(f"RT-DETR model loaded successfully onto {self.selected_device}.")
        except Exception as e:
            print(
                f"ERROR: Failed to load RT-DETR model '{model_path}' onto {self.selected_device}: {e}"
            )
            raise

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []
        placeholder_id = -1

        try:
            results = self.model.predict(
                frame,
                # persist=False,
                classes=0,
                device=self.selected_device,
                verbose=False,
                conf=0.5,
            )

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
        # self.device = get_selected_device()
        self.device = torch.device("cpu")
        print(
            f"Initializing Faster R-CNN strategy (using default TorchVision weights) on device: {self.device}"
        )
        try:
            print(f"Using device: {self.device}")

            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights
            )
            self.model.to(self.device)
            self.model.eval()

            self.transforms = weights.transforms()

            self.person_label_index = 1
            self.score_threshold = 0.5
            self.placeholder_id = -1

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
            img_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(img_rgb_np)

            input_tensor = self.transforms(img_pil)

            input_batch = [input_tensor.to(self.device)]

            with torch.no_grad():
                predictions = self.model(input_batch)

            pred_boxes_xyxy = predictions[0]["boxes"].cpu().numpy()
            pred_labels = predictions[0]["labels"].cpu().numpy()
            pred_scores = predictions[0]["scores"].cpu().numpy()

            for box_xyxy, label, score in zip(
                pred_boxes_xyxy, pred_labels, pred_scores
            ):
                if label == self.person_label_index and score >= self.score_threshold:
                    x1, y1, x2, y2 = box_xyxy
                    width = x2 - x1
                    height = y2 - y1

                    if width > 0 and height > 0:
                        center_x = x1 + width / 2
                        center_y = y1 + height / 2
                        boxes_xywh.append([center_x, center_y, width, height])

                        track_ids.append(self.placeholder_id)
                        confidences.append(float(score))

        except Exception as e:
            print(f"Error during Faster R-CNN processing step: {e}")

            return [], [], []

        return boxes_xywh, track_ids, confidences


class RfDetrStrategy(DetectionTrackingStrategy):
    """Detection using RF-DETR model from the rfdetr library."""

    def __init__(self, model_path: str):
        self.device = get_selected_device()

        device_str = str(self.device.type)

        print(
            f"Initializing RF-DETR strategy (using RFDETRLarge defaults), attempting to use device: '{device_str}'"
        )
        try:
            print("Attempting to load RFDETRLarge...")

            self.model = RFDETRLarge(device=device_str)

            print(f"RFDETRLarge object created, configured for device '{device_str}'.")

            print(
                f"Performing dummy inference check (expecting model on '{device_str}')..."
            )
            dummy_pil = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
            with torch.no_grad():
                _ = self.model.predict(dummy_pil)
            print(f"Dummy inference check successful.")

            self.person_label_index = 1

            self.score_threshold = 0.5
            self.placeholder_id = -1
            print(
                f"RF-DETR strategy initialized. Inference will target device: '{device_str}'."
            )

        except ImportError:
            print(
                "ERROR: Critical - RFDETRLarge could not be imported. Is 'rfdetr' library installed correctly?"
            )
            raise
        except TypeError as te:
            print(
                f"ERROR: Failed to initialize RFDETRLarge - Potential issue with 'device' argument: {te}"
            )
            print(
                "       The RFDETRLarge constructor might not accept a 'device' argument."
            )
            print(
                "       Check the 'rfdetr' library documentation for device specification."
            )
            raise te from None
        except Exception as e:
            print(f"ERROR: Critical - Failed during RFDETRLarge initialization: {e}")
            import traceback

            print(traceback.format_exc())
            raise

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        boxes_xywh = []
        track_ids = []
        confidences = []

        if not hasattr(self, "model") or self.model is None:
            print("ERROR: RF-DETR model not initialized in process_frame.")
            return [], [], []

        try:
            img_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb_np)

            detections = self.model.predict(img_pil, threshold=self.score_threshold)

            if detections:
                pred_boxes_xyxy = detections.xyxy
                pred_labels = detections.class_id
                pred_scores = detections.confidence

                for box_xyxy, label, score in zip(
                    pred_boxes_xyxy, pred_labels, pred_scores
                ):
                    if (
                        label == self.person_label_index
                        and score >= self.score_threshold
                    ):
                        x1, y1, x2, y2 = box_xyxy
                        width = x2 - x1
                        height = y2 - y1

                        if width > 0 and height > 0:
                            center_x = x1 + width / 2
                            center_y = y1 + height / 2
                            boxes_xywh.append([center_x, center_y, width, height])
                            track_ids.append(self.placeholder_id)
                            confidences.append(float(score))

        except Exception as e:
            print(f"Error during RF-DETR processing step: {e}")
            import traceback

            print(traceback.format_exc())
            return [], [], []

        return boxes_xywh, track_ids, confidences
