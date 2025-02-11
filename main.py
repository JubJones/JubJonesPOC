import argparse
import cv2
import numpy as np
import torch

# YOLOv7 and StrongSORT imports
from modeling.yolov7.models.experimental import attempt_load
from modeling.strong_sort.strong_sort import StrongSORT
from modeling.utils.general import non_max_suppression, scale_coords, check_img_size, set_logging
from modeling.utils.plots import plot_one_box


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
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)

    dw /= 2  # divide padding into two sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def compute_homography(frame_width, frame_height, map_width=600, map_height=800):
    """
    Compute the homography transformation from the camera view to a top-down map.

    src_points defines the area in the camera frame we take (imagine cutting out a piece of your picture).
    dst_points defines where on the blank paper (the map) those points will go.

    For instance, here we select a region from the bottom part of the frame (where people are) and map it to a neat rectangle.
    You may need to adjust these points for each video.
    """
    # For example, select the ground area in the frame by using a bit of the full width and from 60% (top) to 100% (bottom) of the height.
    src_points = np.array([
        [50, frame_height],                         # bottom left of the area on the frame
        [frame_width - 50, frame_height],             # bottom right of the area on the frame
        [frame_width - 50, int(frame_height * 0.1)],    # top right of the area on the frame
        [50, int(frame_height * 0.1)]                 # top left of the area on the frame
    ], dtype=np.float32)

    # Now, we decide where these points should map on our bird's eye view (the map).
    # Think of this as drawing a rectangle on a blank paper where you want that piece to be pasted.
    dst_points = np.array([
        [0, map_height],      # bottom left on the map
        [map_width, map_height],  # bottom right on the map
        [map_width, 0],       # top right on the map
        [0, 0]                # top left on the map
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H, src_points, dst_points


def run(source, yolo_weights, strong_sort_weights, img_size, conf_thres, iou_thres, device, classes):
    """
    Run detection and tracking on the video source, and project the detected people into a top-down map.
    """
    set_logging()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    half = device.type != "cpu"  # use FP16 on CUDA

    # Load YOLOv7 model
    model = attempt_load(yolo_weights, map_location=device)
    imgsz = check_img_size(img_size, s=model.stride.max())
    if half:
        model.half()
    model.eval()

    # Initialize StrongSORT tracker
    tracker = StrongSORT(
        strong_sort_weights,
        device,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100
    )

    # Open the video source
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    assert cap.isOpened(), f"Failed to open source: {source}"
    print("Tracking started – press 'q' to quit.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # default FPS if not obtainable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

    # Define the resolution for the top-down map
    map_width, map_height = 600, 800
    # Compute the homography matrix AND retrieve our src and dst points
    H, src_points, dst_points = compute_homography(frame_width, frame_height, map_width, map_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img0 = frame.copy()

        # Preprocess the image: letterbox, convert BGR→RGB, and rearrange to CHW
        img, ratio, dwdh = letterbox(img0, new_shape=imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if half:
            img_tensor = img_tensor.half()

        # Run inference
        with torch.no_grad():
            pred = model(img_tensor)[0]

        # Apply non-max suppression
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

        # Collect detections and rescale boxes to original image size
        detections = []
        class_ids = []
        if pred[0] is not None and len(pred[0]):
            pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred[0]:
                x1, y1, x2, y2 = xyxy
                detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])
                class_ids.append(int(cls.item()))

        # Prepare a blank map image (white background) and draw the dst_points polygon on it.
        img_map = np.full((map_height, map_width, 3), fill_value=255, dtype=np.uint8)
        dst_pts_int = dst_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_map, [dst_pts_int], True, (255, 0, 0), 2)  # Blue polygon on map

        if len(detections):
            detections = np.array(detections)
            class_ids = np.array(class_ids)
            bbox_xyxy = detections[:, :4]
            confidences = detections[:, 4]

            # Convert bounding boxes from xyxy to xywh for the tracker
            bbox_xywh = np.zeros_like(bbox_xyxy)
            bbox_xywh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
            bbox_xywh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
            bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
            bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]

            outputs = tracker.update(bbox_xywh, confidences, class_ids, img0)
            if outputs is not None:
                for output in outputs:
                    bbox = output[:4]  # [x1, y1, x2, y2]
                    track_id = int(output[4])
                    # Draw bounding box and ID on the original frame
                    plot_one_box(bbox, img0, label=f"ID {track_id}", color=(0, 0, 255), line_thickness=2)

                    # Compute the bottom-center of the bounding box (ground contact)
                    center_x = (bbox[0] + bbox[2]) / 2
                    bottom_y = bbox[3]
                    pt = np.array([[[center_x, bottom_y]]], dtype=np.float32)
                    # Transform the point to the map using our homography matrix
                    pt_transformed = cv2.perspectiveTransform(pt, H)
                    pt_mapped = (int(pt_transformed[0][0][0]), int(pt_transformed[0][0][1]))

                    # Debug prints
                    print(f"Detection bottom center: ({center_x}, {bottom_y})")
                    print(f"Mapped point: {pt_mapped}")

                    # Draw the marker on the map
                    cv2.circle(img_map, pt_mapped, 5, (0, 255, 0), -1)
                    cv2.putText(img_map, str(track_id), (pt_mapped[0] + 10, pt_mapped[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the src_points area (the region used from the original frame) as a polygon.
        src_pts_int = src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img0, [src_pts_int], True, (0, 255, 255), 2)  # Yellow polygon on frame

        # Show both the original frame with detections and the top-down map
        cv2.imshow("Tracking", img0)
        cv2.imshow("Map", img_map)
        out.write(img0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Simple demo: detect & track people and project them into a bird's-eye view using YOLOv7 and StrongSORT."
    )
    parser.add_argument("--source", type=str, default="0", help="Video source: webcam index (e.g., 0) or file path.")
    parser.add_argument("--yolo-weights", type=str, default="yolov7.pt", help="Path to YOLOv7 weights")
    parser.add_argument("--strong-sort-weights", type=str, default="osnet_x0_25_msmt17.pt",
                        help="Path to StrongSORT (OSNet) weights")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="cuda:0", help="Device to run on, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--classes", nargs="+", type=int, default=[0],
                        help="Filter by class: by default [0] (people in COCO indexing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.yolo_weights, args.strong_sort_weights, args.img_size,
        args.conf_thres, args.iou_thres, args.device, args.classes)