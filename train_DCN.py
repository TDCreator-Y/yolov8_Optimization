from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"ultralytics/cfg/models/v8/yolov8_DCN.yaml")

    model.train(
        data=r"yolov8mini_train.yaml",
        epochs=3,  # 先做结构验证
        imgsz=320,  # 小尺寸先跑通
        batch=4,  # RTX2060 可以
        cache="ram",
        workers=1,
        project="results_Optimization",
        name="yolov8n_DCN",
        optimizer="SGD",
        lr0=0.01,
    )
