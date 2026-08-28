"""
Model architecture module for PyTorch image classification.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SimpleCNN(nn.Module):
    """
    Lightweight 3-layer Convolutional Neural Network optimized for 32x32 images.
    Useful for fast testing and CPU execution.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """
    Factory function to instantiate models.
    
    Args:
        architecture: Name of model architecture ('resnet18' or 'simple_cnn')
        num_classes: Number of output classification targets
        pretrained: Whether to load pre-trained ImageNet weights
    """
    arch = architecture.lower().strip()
    
    if arch == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        
        # Modify the first conv layer for 32x32 CIFAR-10 images (3x3 kernel instead of 7x7 stride 2)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool to preserve spatial resolution
        
        # Adjust final classification layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    elif arch in ("simple_cnn", "cnn"):
        return SimpleCNN(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unsupported architecture: '{architecture}'. Choose 'resnet18' or 'simple_cnn'.")


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
