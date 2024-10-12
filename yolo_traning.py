from ultralytics import YOLO

class traning_YOLO():
    def __init__(self):
        self.model = None

    def load_model(self):
        # Load a model
        self.model = YOLO("yolo11s.pt") 
    def human_detect(self, img):
        result = self.model()