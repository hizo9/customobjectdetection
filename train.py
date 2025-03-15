# imports
from ultralytics import YOLO

# config
model = YOLO("models/yolo11m.pt")

# code
results = model.train(data="customobject.yaml", epochs=100, imgsz=640)