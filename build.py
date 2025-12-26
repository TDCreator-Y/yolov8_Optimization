from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8_DCN.yaml")
print("✅ DCN_C2f build SUCCESS")


#=================================================原CA
self.cv2 = nn.ModuleList(
    nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
)