import argparse
from modeling_logic.tracker import YOLOv7StrongSortTracker


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Simple demo: detect & track people and project them into a bird's-eye view using YOLOv7 and StrongSORT."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g., 0) or file path.",
    )
    parser.add_argument(
        "--yolo-weights", type=str, default="yolov7.pt", help="Path to YOLOv7 weights"
    )
    parser.add_argument(
        "--strong-sort-weights",
        type=str,
        default="osnet_x0_25_msmt17.pt",
        help="Path to StrongSORT (OSNet) weights",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="Inference image size"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="Object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device to run on, e.g. 'cuda:0' or 'cpu'"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0],
        help="Filter by class: by default [0] (people in COCO indexing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tracker = YOLOv7StrongSortTracker(
        yolo_weights=args.yolo_weights,
        strong_sort_weights=args.strong_sort_weights,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        classes=args.classes,
    )
    tracker.run(args.source)
