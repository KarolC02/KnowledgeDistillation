import torch.nn as nn
import warnings

def adapt_model_to_classes(model: nn.Module, num_classes: int):
    """Adapt a classification model to have a different number of output classes.
    
    Supports common CNN architectures (AlexNet, VGG, ResNet, DenseNet, SqueezeNet, etc.)
    and vision transformers (ViT, Swin Transformer). Replaces the final classification layer 
    with a new layer of size `num_classes`, while keeping all other layers intact.
    
    Returns the modified model, or None if the model type is not supported for adaptation.
    """
    # 1. ResNet, Inception, etc. – final FC layer named 'fc'
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        return model

    # 2. Models with 'classifier' attribute
    if hasattr(model, 'classifier'):
        # If it's a single Linear (e.g., DenseNet)
        if isinstance(model.classifier, nn.Linear):
            in_feat = model.classifier.in_features
            model.classifier = nn.Linear(in_feat, num_classes)
            return model
        # If it's a Sequential (AlexNet, VGG, SqueezeNet, MobileNet, etc.)
        if isinstance(model.classifier, nn.Sequential):
            # Replace the last layer of the Sequential
            last_idx = len(model.classifier) - 1
            last_layer = model.classifier[last_idx]
            # Case: last layer is Linear (AlexNet, VGG, MobileNet v2/v3)
            if isinstance(last_layer, nn.Linear):
                in_feat = last_layer.in_features
                model.classifier[last_idx] = nn.Linear(in_feat, num_classes)
                return model
            # Case: last layer is Conv2d (SqueezeNet final conv layer)
            if isinstance(last_layer, nn.Conv2d):
                in_ch = last_layer.in_channels
                # Preserve kernel size/stride/padding of the conv
                model.classifier[last_idx] = nn.Conv2d(in_ch, num_classes, 
                                                      kernel_size=last_layer.kernel_size, 
                                                      stride=last_layer.stride, 
                                                      padding=last_layer.padding,
                                                      bias=(last_layer.bias is not None))
                return model
            # (If last_layer is something else like dropout, check the layer before it)
            if last_idx > 0 and isinstance(model.classifier[last_idx-1], nn.Linear):
                in_feat = model.classifier[last_idx-1].in_features
                model.classifier[last_idx-1] = nn.Linear(in_feat, num_classes)
                # If last layer was dropout or activation, we can keep it as-is
                return model

    # 3. Vision Transformers or other models with 'head' attribute
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, num_classes)
        return model

    # 4. (Optional) Inception v3 auxiliary classifier
    if hasattr(model, 'AuxLogits'):  # Inception v3 has AuxLogits with its own fc
        try:
            in_feat = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(in_feat, num_classes)
        except AttributeError:
            pass  # continue even if AuxLogits isn't exactly as expected
    if hasattr(model, 'aux_logits') and model.aux_logits:  # handle Inception aux flag if needed
        # Inception's main classifier:
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            in_feat = model.fc.in_features
            model.fc = nn.Linear(in_feat, num_classes)
            return model

    # 5. HuggingFace style models (e.g., ViTForImageClassification) 
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        # Some HuggingFace vision models use 'classifier' for the output layer
        in_feat = model.classifier.in_features
        model.classifier = nn.Linear(in_feat, num_classes)
        return model

    # If none of the above patterns matched, we cannot adapt this model
    model_name = model.__class__.__name__
    warnings.warn(f"adapt_model_to_classes: Unsupported model type '{model_name}'. "
                  f"Skipping adaptation.", UserWarning)
    return None
