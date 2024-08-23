import torchvision.models as models
class ResNet50():
   def __init__(self):
       self.model = models.resnet50(pretrained=True)

   def getModel(self):
       return self.model