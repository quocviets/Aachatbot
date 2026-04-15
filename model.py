import torch.nn as nn
from torchvision import models


def build_mobilenetv3_small(num_classes):
    
    model = models.mobilenet_v3_small(
        weights="IMAGENET1K_V1"
    )

    # thay classifier cuối
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes
    )

    return model