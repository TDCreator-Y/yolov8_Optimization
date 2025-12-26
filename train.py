from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolov8n.pt")
    model.train(
        data=r"yolov8mini_train.yaml",
        epochs=15,
        imgsz=320,
        batch=8,
        cache="ram",
        workers=1,
        project="results_Optimization",
        name="yolov8n_baseline",
        optimizer="SGD",
        lr0=0.01,
    )
