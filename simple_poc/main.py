from ui.app import MTMMCTrackerApp

if __name__ == "__main__":
    app = MTMMCTrackerApp(model_path="yolov11l.pt")
    demo = app.build_ui()
    demo.launch(share=True)
