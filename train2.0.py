import warnings

warnings.filterwarnings("ignore", message=".*adaptive_avg_pool2d_backward_cuda.*")

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\deeplearning\ultralytics-8.3.225\ultralytics\cfg\models\v8\yolov8_DCN.yaml")
    model.train(
        data=r"yolov8mini_train.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        cache="ram",
        workers=1,
        project="results_Optimization",
        name="yolov8n_DCN_20",
        optimizer="AdamW",
        lr0=0.001,
    )
