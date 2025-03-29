from ui.app import MTMMCTrackerApp


if __name__ == "__main__":
    # Example: Choose the model type and path here
    # app = MTMMCTrackerApp(model_path="yolov11x.pt", model_type='yolo')
    # app = MTMMCTrackerApp(model_path="yolov9e.pt", model_type='yolo')
    app = MTMMCTrackerApp(model_path="rtdetr-x.pt", model_type='rtdetr')
    # app = MTMMCTrackerApp(model_path="fasterrcnn_resnet50_fpn", model_type='fasterrcnn') # Path ignored for default torchvision weights

    demo = app.build_ui()
    # Set share=True if needed for external access
    demo.launch(share=False)