from ultralytics import YOLO

class YoloV8:
    def __init__(self):
        self.model = YOLO('../yolov8s-seg.pt')

    def get_model(self):
        return self.model
