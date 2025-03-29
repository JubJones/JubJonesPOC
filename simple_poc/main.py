from ui.app import MTMMCTrackerApp

if __name__ == "__main__":
    # app = MTMMCTrackerApp(model_path="yolov11l.pt")
    app = MTMMCTrackerApp(model_path="yolov9e.pt")
    # app = MTMMCTrackerApp(model_path="rtdetr-x.pt") # Don't forget to swap in the tracker.py
    demo = app.build_ui()
    demo.launch(share=True)
