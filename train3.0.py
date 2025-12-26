# ===============================
# train3.0.py  (DCN runtime replace version)
# ===============================

import os
import sys

# 确保能 import 到 my_modules
sys.path.append(os.path.dirname(__file__))

import torch

from my_modules.dcn_c2f_runtime import DCN_C2f
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f


def replace_backbone_c2f_with_dcn(yolo_model):
    net = yolo_model.model
    layers = net.model if hasattr(net, "model") else net

    replaced = 0
    for i, m in enumerate(layers):
        if isinstance(m, C2f):
            try:
                out_c = m.cv2.conv.out_channels
            except Exception:
                continue

            if out_c in (256, 512):
                c1 = m.cv1.conv.in_channels
                c2 = out_c
                n = len(m.m)
                shortcut = getattr(m, "shortcut", False)
                e = getattr(m, "e", 0.5)

                new_m = DCN_C2f(c1=c1, c2=c2, n=n, shortcut=shortcut, e=e)

                # ===== 拷贝 YOLOv8 内部属性（非常关键）=====
                for attr in ["i", "f", "type", "np"]:
                    if hasattr(m, attr):
                        setattr(new_m, attr, getattr(m, attr))

                layers[i] = new_m
                print(f"[DCN] Replaced C2f at layer {i} (out_c={out_c}, n={n})")
                replaced += 1

    print(f"[DCN] Total replaced C2f blocks: {replaced}")


if __name__ == "__main__":
    # -------- 1. 构建模型（一定用 yaml，不用 pt） --------
    yaml_path = r"D:\deeplearning\ultralytics-8.3.225\ultralytics\cfg\models\v8\yolov8.yaml"
    model = YOLO(yaml_path)

    # -------- 2. 运行时替换 DCN --------
    replace_backbone_c2f_with_dcn(model)

    # -------- 3. forward 校验（非常重要） --------
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 640, 640)
        _ = model.model(dummy)
    print("[DCN] Forward check passed ✔")

    # -------- 4. 训练 --------
    model.train(
        data=r"yolov8mini_train.yaml",
        epochs=20,
        imgsz=640,
        batch=8,  # 显存不够就改 4
        workers=1,
        cache="ram",  # 内存不够就 False
        project="results_Optimization",
        name="yolov8n_DCN_20",
        optimizer="AdamW",
        lr0=0.0005,
        resume=False,
    )
