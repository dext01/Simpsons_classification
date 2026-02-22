import torch.nn as nn
import torchvision.models as models


def get_resnet18_finetune(num_classes=42):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model