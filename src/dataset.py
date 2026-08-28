"""
Dataset and DataLoader utility module for CIFAR-10 image classification.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 class labels in canonical order
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Normalization constants calculated across CIFAR-10 dataset
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    
    Args:
        train: If True, applies data augmentation (RandomCrop, RandomHorizontalFlip).
               If False, applies standard deterministic normalization.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns training and validation PyTorch DataLoaders.
    
    Args:
        data_dir: Path to directory for storing/loading CIFAR-10 dataset.
        batch_size: Mini-batch size.
        num_workers: Number of subprocesses for data loading.
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
