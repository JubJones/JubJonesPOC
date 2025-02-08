import argparse

import cv2
import numpy as np
import torch
# YOLOv7 imports
from yolov7.models.experimental import attempt_load
# StrongSORT import
from strong_sort.strong_sort import StrongSORT
from utils.general import non_max_suppression, scale_coords, check_img_size, set_logging
from utils.plots import plot_one_box


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    Resize image while keeping aspect ratio and add padding.
    """
    shape = im.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute scaling ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute new unpadded dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        # Make sure padding is a multiple of 32
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def run(source, yolo_weights, strong_sort_weights, img_size, conf_thres, iou_thres, device, classes):
    """
    Run detection and tracking on the provided video source.
    """
    set_logging()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    half = device.type != "cpu"  # use FP16 only on CUDA

    # Load YOLOv7 model
    model = attempt_load(yolo_weights, map_location=device)
    imgsz = check_img_size(img_size, s=model.stride.max())
    if half:
        model.half()
    model.eval()

    # Initialize StrongSORT tracker with default parameters (tweak as needed)
    tracker = StrongSORT(
        strong_sort_weights,
        device,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100
    )

    # Open video source (webcam, file, URL, etc.)
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    assert cap.isOpened(), f"Failed to open source: {source}"
    print("Tracking started – press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img0 = frame.copy()

        # Preprocess image: letterbox resize, convert BGR→RGB, transpose to CHW format
        img, ratio, dwdh = letterbox(img0, new_shape=imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if half:
            img_tensor = img_tensor.half()

        # Inference
        with torch.no_grad():
            pred = model(img_tensor)[0]

        # Apply non-max suppression filtering
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

        # Collect detections and rescale boxes to original image size
        detections = []
        class_ids = []
        if pred[0] is not None and len(pred[0]):
            # Rescale coordinates to original image size
            pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred[0]:
                x1, y1, x2, y2 = xyxy
                detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])
                class_ids.append(int(cls.item()))

        # If detections exist, convert bbox format and update tracker
        if len(detections):
            detections = np.array(detections)
            class_ids = np.array(class_ids)
            bbox_xyxy = detections[:, :4]  # [x1, y1, x2, y2]
            confidences = detections[:, 4]  # confidence scores

            # Convert bounding boxes from xyxy to xywh format
            bbox_xywh = np.zeros_like(bbox_xyxy)
            bbox_xywh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2  # center x
            bbox_xywh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2  # center y
            bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]  # width
            bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]  # height

            # Now update the tracker with the required arguments: bbox_xywh, confidences, class_ids, and the original image.
            outputs = tracker.update(bbox_xywh, confidences, class_ids, img0)
            if outputs is not None:
                for output in outputs:
                    bbox = output[:4]  # bounding box (xyxy)
                    track_id = output[4]  # tracker ID
                    plot_one_box(bbox, img0, label=f"ID {int(track_id)}", color=(255, 0, 0), line_thickness=2)

        cv2.imshow("Tracking", img0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="POC for detecting and tracking people using YOLOv7 and StrongSORT"
    )
    parser.add_argument('--source', type=str, default='0', help="Video source: webcam (0), file, URL, etc.")
    parser.add_argument('--yolo-weights', type=str, default='yolov7.pt', help="Path to YOLOv7 weights")
    parser.add_argument('--strong-sort-weights', type=str, default='osnet_x0_25_msmt17.pt',
                        help="Path to StrongSORT (OSNet) weights")
    parser.add_argument('--img-size', type=int, default=640, help="Inference image size")
    parser.add_argument('--conf-thres', type=float, default=0.25, help="Object confidence threshold")
    parser.add_argument('--iou-thres', type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument('--device', default='cuda:0', help="CUDA device (e.g. 0 or 'cpu')")
    parser.add_argument('--classes', nargs='+', type=int, default=[0],
                        help="Filter detections by class index; for people use 0 (COCO indexing)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(
        args.source,
        args.yolo_weights,
        args.strong_sort_weights,
        args.img_size,
        args.conf_thres,
        args.iou_thres,
        args.device,
        args.classes
    )
