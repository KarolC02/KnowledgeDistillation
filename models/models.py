from torchvision.models import resnet18, ResNet18_Weights, alexnet, AlexNet_Weights, ResNet50_Weights, resnet50, convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, swin_b, Swin_B_Weights, swin_v2_b, Swin_V2_B_Weights
from torchvision.models import densenet121, DenseNet121_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights, mnasnet1_0, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights
from torchvision.models.mnasnet import MNASNet1_0_Weights

model_dict = {
    "resnet18_pretrained": lambda: resnet18(weights=ResNet18_Weights.DEFAULT),
    "resnet18": lambda: resnet18(weights=None),
    "resnet50_pretrained": lambda: resnet50(weights=ResNet50_Weights.DEFAULT),
    "resnet50": lambda: resnet50(weights=None),
    "AlexNet_pretrained": lambda: alexnet(weights=AlexNet_Weights.DEFAULT),
    "AlexNet": lambda: alexnet(weights=None),
    "ConvNeXt_pretrained": lambda: convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT),
    "ConvNeXt": lambda: convnext_base(weights=None),
    "vit_b_16_pretrained": lambda: vit_b_16(weights=ViT_B_16_Weights.DEFAULT),
    "vit_b_16": lambda: vit_b_16(weights=None),
    "vit_l_16_pretrained": lambda: vit_l_16(weights=ViT_L_16_Weights.DEFAULT),
    "vit_l_16": lambda: vit_l_16(weights=None),
    "swin_b_pretrained": lambda: swin_b(weights=Swin_B_Weights.DEFAULT),
    "swin_b": lambda: swin_b(weights=None),
    "swin_v2_b_pretrained": lambda: swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT),
    "swin_v2_b": lambda: swin_v2_b(weights=None),
    
    "densenet121_pretrained": lambda: densenet121(weights=DenseNet121_Weights.DEFAULT),
    "densenet121": lambda: densenet121(weights=None),
    "resnext50_32x4d_pretrained": lambda: resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT),
    "resnext50_32x4d": lambda: resnext50_32x4d(weights=None),
    "mnasnet1_0_pretrained": lambda: mnasnet1_0(weights=MNASNet1_0_Weights.DEFAULT),
    "mnasnet1_0": lambda: mnasnet1_0(weights=None),
    "shufflenet_v2_x1_0_pretrained": lambda: shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT),
    "shufflenet_v2_x1_0": lambda: shufflenet_v2_x1_0(weights=None),
    
    "vgg16_pretrained": lambda: vgg16(weights=VGG16_Weights.DEFAULT),
    "vgg16": lambda: vgg16(weights=None),
    "vgg19_pretrained": lambda: vgg19(weights=VGG19_Weights.DEFAULT),
    "vgg19": lambda: vgg19(weights=None),
}
