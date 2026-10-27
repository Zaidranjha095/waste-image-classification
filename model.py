"""MobileNetV2 waste classifier.

The checkpoint in weights/model.pth stores this module's state_dict.
build_model(pretrained=False) recreates the exact architecture for loading.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
ARCH = "torchvision.models.mobilenet_v2"


def build_model(num_classes: int = len(CLASSES), pretrained: bool = False) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
