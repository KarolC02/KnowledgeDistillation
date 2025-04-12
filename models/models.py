from torchvision.models import resnet18, ResNet18_Weights, alexnet, AlexNet_Weights
from models.custom_model_defs import myCNN, fastCNN, OneLayerNN

model_dict = {
    "resnet18_pretrained": lambda: resnet18(weights=ResNet18_Weights.DEFAULT),
    "AlexNet": lambda: alexnet(weights=AlexNet_Weights.DEFAULT),
    "myCNN": myCNN,
    "fastCNN": fastCNN,
    "OneLayerNN": OneLayerNN
}
