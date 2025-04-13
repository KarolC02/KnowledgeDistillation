from torchvision.models import resnet18, ResNet18_Weights, alexnet, AlexNet_Weights, ResNet50_Weights, resnet50, convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from models.custom_model_defs import myCNN, fastCNN, OneLayerNN, myImprovedCNN

model_dict = {
    "resnet18_pretrained": lambda: resnet18(weights=ResNet18_Weights.DEFAULT),
    "resnet50_pretrained": lambda: resnet50(weights=ResNet50_Weights.DEFAULT),
    "AlexNet_pretrained": lambda: alexnet(weights=AlexNet_Weights.DEFAULT),
    "ConvNeXt_pretrained": lambda: convnext_base(weights = ConvNeXt_Base_Weights.DEFAULT),
    "vit_b_16": lambda: vit_b_16(weights = ViT_B_16_Weights.DEFAULT),
    "myCNN": myCNN,
    "fastCNN": fastCNN,
    "OneLayerNN": OneLayerNN,
    "myImprovedCNN": myImprovedCNN
}
