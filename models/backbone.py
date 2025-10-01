import torch
import torchvision.models as models
from config import Config
import torch.nn as nn

def get_backbone_model(backbone_name=Config.BACKBONE, pretrained=True):
    if backbone_name == "resnet50":
        # Use new weights API to avoid deprecation warnings
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
        model.fc = nn.Identity()
        feature_dim = 2048
    elif backbone_name == "densenet121":
        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=weights)
        else:
            model = models.densenet121(weights=None)
        model.classifier = nn.Identity()
        feature_dim = 1024
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return model, feature_dim