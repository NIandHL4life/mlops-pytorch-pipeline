"""
Unit tests for PyTorch models, data transformations, and inference API.
"""

import pytest
import torch
from src.model import get_model, count_parameters, SimpleCNN
from src.dataset import get_transforms, CIFAR10_CLASSES


def test_simple_cnn_forward_shape():
    """Test that SimpleCNN outputs expected (batch, 10) tensor."""
    model = SimpleCNN(num_classes=10)
    model.eval()
    dummy_input = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 10)


def test_resnet18_cifar_forward_shape():
    """Test that adapted ResNet-18 handles 32x32 images without crashing."""
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 10)


def test_invalid_architecture():
    """Test that requesting an unknown architecture raises ValueError."""
    with pytest.raises(ValueError):
        get_model(architecture="non_existent_arch")


def test_dataset_transforms():
    """Test training and validation transform pipelines."""
    train_tf = get_transforms(train=True)
    val_tf = get_transforms(train=False)

    assert train_tf is not None
    assert val_tf is not None
    assert len(CIFAR10_CLASSES) == 10


def test_parameter_counting():
    """Test trainable parameter counter helper."""
    model = SimpleCNN(num_classes=10)
    params = count_parameters(model)
    assert isinstance(params, int)
    assert params > 0
