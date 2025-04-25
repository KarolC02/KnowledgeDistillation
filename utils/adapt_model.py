from torch import nn

def adapt_model_to_classes(model, num_classes):
    if hasattr(model, 'fc'):  # Typical for ResNet, DenseNet, ShuffleNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):  # VGG, ConvNeXt, MnasNet
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:  # AlexNet
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'heads'):  # ViT-B/16, ViT-L/16
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):  # Swin-B, Swin-V2-B
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported yet for class adaptation.")
    return model
