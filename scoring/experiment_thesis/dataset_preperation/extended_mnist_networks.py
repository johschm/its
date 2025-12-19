import math
import torch.nn as nn
from escnn import nn as escnn_nn
from .basic_networks import FlexibleResNet, ESCNNFlexibleResNet, get_flexible_resnet_layer_mapping

def extended_simple_cnn(num_classes=47, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        activation(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        activation(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 7 * 7, 256),
        activation(),
        nn.Linear(256, num_classes)
    )

def extended_deep_cnn(num_classes=47, activation=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), activation(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), activation(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), activation(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 256, 3, padding=0), nn.BatchNorm2d(256), activation(),
        nn.Flatten(),
        nn.Linear(256, num_classes)
    )

def extended_resnet44(num_classes=47, activation=nn.GELU):
    return FlexibleResNet([32, 64, 128], 7, num_classes=num_classes, in_channels=1, activation=activation)


def extended_equivariant_resnet44(num_classes=47, act_cls=escnn_nn.ReLU, num_rotations=8):
    base = [32, 64, 128]
    sizes = [math.ceil(1.2 * s / (num_rotations ** 0.5)) for s in base]
    return ESCNNFlexibleResNet(sizes, 7, num_classes=num_classes, in_channels=1,
                               act_cls=act_cls, rotations=num_rotations)


def extended_resnet_small(num_classes=47, activation=nn.GELU):
    return FlexibleResNet([64, 128, 256,512], 1, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=1)

def _scale_channels_list(channels, divisor):
    return [max(8, math.ceil(c / divisor)) for c in channels]

def extended_resnet_small_scaled(num_classes=47, activation=nn.GELU, divisor=1):
    base = [64, 128, 256, 512]
    if divisor != 1:
        base = _scale_channels_list(base, divisor)
    return FlexibleResNet(base, 1, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=1)



def get_extended_mnist_architectures():
    return [
        "extended_simple_cnn",
        "extended_deep_cnn",
        "extended_resnet44",
        "extended_resnet_small",
        "extended_equivariant_resnet44",
        # scaled variants
        "extended_resnet_small_half",
        "extended_resnet_small_quarter",
    ]

def get_extended_mnist_network(architecture, num_classes=47, num_rotations=8):
    a = architecture.lower()
    if a == "extended_simple_cnn":
        return extended_simple_cnn(num_classes=num_classes)
    if a == "extended_deep_cnn":
        return extended_deep_cnn(num_classes=num_classes)
    if a == "extended_resnet44":
        return extended_resnet44(num_classes=num_classes)
    if a == "extended_resnet_small":
        return extended_resnet_small(num_classes=num_classes, activation=nn.GELU)
    if a == "extended_resnet_small_half":
        return extended_resnet_small_scaled(num_classes=num_classes, divisor=2)
    if a == "extended_resnet_small_quarter":
        return extended_resnet_small_scaled(num_classes=num_classes, divisor=4)

    if a == "extended_equivariant_resnet44":
        return extended_equivariant_resnet44(num_classes=num_classes, num_rotations=num_rotations)
    raise ValueError(f"Unknown extended MNIST architecture: {architecture}")

def get_extended_mnist_network_layer(architecture, index, num_classes=47, num_rotations=8):
    """
    Returns (layer_name, capture_mode) for FlexibleResNet architectures.
    """
    a = architecture.lower()
    
    if a == "extended_resnet44":
        blocks_per_stage = 7
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=3)
        return mapping[index]
    elif a == "extended_resnet_small":
        blocks_per_stage = 1
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=4)
        return mapping[index]

    
    raise ValueError(f"Layer mapping not implemented for extended MNIST architecture: {architecture}")