import cv2
import torch
import numpy as np

# YOLOv7 and StrongSORT imports
from modeling.yolov7.models.experimental import attempt_load
from modeling.strong_sort.strong_sort import StrongSORT
from modeling.utils.general import non_max_suppression, scale_coords, check_img_size, set_logging
from modeling.utils.plots import plot_one_box

from modeling_logic.data_preprocessing import letterbox, compute_homography


class YOLOv7StrongSortTracker:
    def __init__(self, yolo_weights, strong_sort_weights, img_size=640, conf_thres=0.25,
                 iou_thres=0.45, device="cuda:0", classes=[0]):
        set_logging()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != "cpu"

        # Load YOLOv7 model and check image size compatibility
        self.model = attempt_load(yolo_weights, map_location=self.device)
        self.img_size = check_img_size(img_size, s=self.model.stride.max())
        if self.half:
            self.model.half()
        self.model.eval()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes

        # Initialize StrongSORT tracker
        self.tracker = StrongSORT(
            strong_sort_weights,
            self.device,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100
        )

    def preprocess(self, img0):
        """
        Preprocess the frame: apply letterbox resize, convert colors, and prepare the tensor.
        """
        img, ratio, dwdh = letterbox(img0, new_shape=self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to CHW
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if self.half:
            img_tensor = img_tensor.half()
        return img_tensor, ratio, dwdh

    def postprocess(self, pred, img0, img_tensor):
        """
        Apply non-max suppression and rescale detections to the original image size.
        """
        detections = []
        class_ids = []
        if pred[0] is not None and len(pred[0]):
            pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred[0]:
                x1, y1, x2, y2 = xyxy
                detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])
                class_ids.append(int(cls.item()))
        return detections, class_ids

    def run(self, source):
        """
        Run detection and tracking on the given video source.
        """
        cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception(f"Failed to open source: {source}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # use default if FPS is not available
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

        # Define the resolution for the top-down map and compute the homography
        map_width, map_height = 600, 800
        H, src_points, dst_points = compute_homography(frame_width, frame_height, map_width, map_height)

        print("Tracking started – press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img0 = frame.copy()
            img_tensor, ratio, dwdh = self.preprocess(img0)

            # Run inference
            with torch.no_grad():
                pred = self.model(img_tensor)[0]

            # Apply non-max suppression and extract detections
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)
            detections, class_ids = self.postprocess(pred, img0, img_tensor)

            # Prepare a blank map image (white background) and draw the designated polygon
            img_map = np.full((map_height, map_width, 3), 255, dtype=np.uint8)
            dst_pts_int = dst_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_map, [dst_pts_int], True, (255, 0, 0), 2)

            if detections:
                detections_np = np.array(detections)
                class_ids_np = np.array(class_ids)
                bbox_xyxy = detections_np[:, :4]
                confidences = detections_np[:, 4]

                # Convert bounding boxes from [x1,y1,x2,y2] to [cx,cy,w,h] for StrongSORT
                bbox_xywh = np.zeros_like(bbox_xyxy)
                bbox_xywh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
                bbox_xywh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
                bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
                bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]

                outputs = self.tracker.update(bbox_xywh, confidences, class_ids_np, img0)
                if outputs is not None:
                    for output in outputs:
                        bbox = output[:4]  # [x1, y1, x2, y2]
                        track_id = int(output[4])
                        plot_one_box(bbox, img0, label=f"ID {track_id}", color=(0, 0, 255), line_thickness=2)

                        # Compute the bottom-center of the bounding box
                        center_x = (bbox[0] + bbox[2]) / 2
                        bottom_y = bbox[3]
                        pt = np.array([[[center_x, bottom_y]]], dtype=np.float32)

                        # Map the point to the top-down view using the homography matrix
                        pt_transformed = cv2.perspectiveTransform(pt, H)
                        pt_mapped = (int(pt_transformed[0][0][0]), int(pt_transformed[0][0][1]))

                        print(f"Detection bottom center: ({center_x}, {bottom_y})")
                        print(f"Mapped point: {pt_mapped}")

                        cv2.circle(img_map, pt_mapped, 5, (0, 255, 0), -1)
                        cv2.putText(img_map, str(track_id), (pt_mapped[0] + 10, pt_mapped[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw the source polygon on the original image
            src_pts_int = src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img0, [src_pts_int], True, (0, 255, 255), 2)

            cv2.imshow("Tracking", img0)
            cv2.imshow("Map", img_map)
            out.write(img0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()