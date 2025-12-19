import torch.nn as nn
from .basic_networks import FlexibleResNet, get_flexible_resnet_layer_mapping
import math

def coil_cnn_small(num_classes=100, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), activation(), nn.MaxPool2d(2),   # 128->64
        nn.Conv2d(64, 128, 3, padding=1), activation(), nn.MaxPool2d(2), # 64->32
        nn.Conv2d(128, 256, 3, padding=1), activation(), nn.MaxPool2d(2),# 32->16
        nn.Conv2d(256, 256, 3, padding=1), activation(), nn.MaxPool2d(2),# 16->8
        nn.Flatten(),
        nn.Linear(256 * 8 * 8, 512), activation(),
        nn.Linear(512, num_classes)
    )

def coil_resnet_small(num_classes=100, activation=nn.GELU):
    # Use same specs as MNIST ResNetSmall2, but with stem_stride=2
    return FlexibleResNet([64, 128, 256, 512], 1, num_classes=num_classes, in_channels=3, activation=activation, stem_stride=2)

def _scale_channels_list(channels, divisor):
    return [max(8, math.ceil(c / divisor)) for c in channels]

def coil_resnet_small_scaled(num_classes=100, activation=nn.GELU, divisor=1):
    base = [64, 128, 256, 512]
    if divisor != 1:
        base = _scale_channels_list(base, divisor)
    return FlexibleResNet(base, 1, num_classes=num_classes, in_channels=3, activation=activation, stem_stride=2)

def get_coil100_architectures():
    return [
        "coil_cnn_small",
        "coil_resnet_small",
        # scaled variants
        "coil_resnet_small_half",
        "coil_resnet_small_quarter",
    ]

def get_coil100_network(architecture, num_classes=100):
    a = architecture.lower()
    if a == "coil_cnn_small":
        return coil_cnn_small(num_classes=num_classes)
    if a == "coil_resnet_small":
        return coil_resnet_small(num_classes=num_classes)
    if a == "coil_resnet_small_half":
        return coil_resnet_small_scaled(num_classes=num_classes, divisor=2)
    if a == "coil_resnet_small_quarter":
        return coil_resnet_small_scaled(num_classes=num_classes, divisor=4)
    raise ValueError(f"Unknown COIL100 architecture: {architecture}")

def get_coil100_network_layer(architecture, index, num_classes=100):
    """
    Returns (layer_name, capture_mode) for FlexibleResNet architectures.
    """
    a = architecture.lower()
    if a == "coil_resnet_small":
        blocks_per_stage = 1  # 4 stages
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=4)
        return mapping[index]
    raise ValueError(f"Layer mapping not implemented for COIL100 architecture: {architecture}")