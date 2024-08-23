import torchvision.models as models

class MobileNet:
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)

    def get_model(self):
        return self.model
