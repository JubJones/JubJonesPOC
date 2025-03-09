from ui.app import PersonTrackerApp


def main():
    app = PersonTrackerApp(model_path="yolo11n.pt")
    demo = app.build_ui()
    demo.launch(share=True)


if __name__ == "__main__":
    main()
