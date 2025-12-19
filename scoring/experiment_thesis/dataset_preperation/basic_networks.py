import math

import torch
import torch.nn as nn
from escnn import nn as escnn_nn
from escnn import gspaces

from rot_resnet import ESCNNFlexibleResNet

#TODO risurfconv does not train. Likely not worth fixing it as pointnet plus pca does even if it only maps to 4 canoncial poses instead of complete invariance.
# PointNet++ based components for ModelNet
try:
    from model.pointnet_plus import PointNetPlus, SAModule, PointNetPlusHalfSized, PointNetPlusQuarterSized
    from dataset.geometric_wrapper import BatchNormalizeScale,BatchNormalizeScaleEuclidean, TensorGeometricModelWrapper, NormalizeRotationVectorized
    # Add module wrapper to allow usage in nn.Sequential
    from dataset.geometric_wrapper import NormalizeRotationVectorizedModule
    _POINTNET_AVAILABLE = True
except Exception:
    _POINTNET_AVAILABLE = False

# RISurfConv (rotation-invariant surface convolution) components (optional)
try:
    from model.risurfconv.risurconv_cls import get_model as get_risurfconv_model
    _RISURFCONV_AVAILABLE = True
except Exception:
    _RISURFCONV_AVAILABLE = False

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.act = activation()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        shortcut = x if self.proj is None else self.proj(x)
        return self.act(y + shortcut)

class FlexibleResNet(nn.Module):
    def __init__(self, channels, blocks_per_stage, num_classes=10, in_channels=1, activation=nn.ReLU, stem_stride=1):
        super().__init__()
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * len(channels)
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have same length")
        self.stem_conv = nn.Conv2d(in_channels, channels[0], 3, stride=stem_stride, padding=1, bias=False)  # changed
        self.stem_bn   = nn.BatchNorm2d(channels[0])
        self.stem_act  = activation()

        stages = []
        in_ch = channels[0]
        for idx, (out_ch, n) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if idx == 0 else 2
            blocks = [BasicBlock(in_ch, out_ch, stride=stride, activation=activation)]
            in_ch = out_ch
            for _ in range(n-1):
                blocks.append(BasicBlock(in_ch, in_ch, stride=1, activation=activation))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            x = stage(x)
        x = self.head_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def convert_flexible_resnet_to_dropout_sequential(model: FlexibleResNet, dropout_p: float = 0.5) -> nn.Sequential:
    """
    Converts a FlexibleResNet to a nn.Sequential model with a dropout layer
    after the flatten operation.
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    layers = [
        model.stem_conv,
        model.stem_bn,
        model.stem_act,
        *model.stages,
        model.head_pool,
        nn.Flatten(1),
        nn.Dropout(p=dropout_p),
        model.fc
    ]
    return nn.Sequential(*layers)


def convert_flexible_resnet_to_sequential(model: FlexibleResNet) -> nn.Sequential:
    """
    Converts a FlexibleResNet to a nn.Sequential model.
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    layers = [
        model.stem_conv,
        model.stem_bn,
        model.stem_act,
        *model.stages,
        model.head_pool,
        nn.Flatten(1),
        model.fc
    ]
    return nn.Sequential(*layers)





def _build_pointnetplus(num_classes=10, normalize=True, deterministic_fps=False,smaller=None):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    if smaller is None:
        core = PointNetPlus(num_classes=num_classes)
    elif smaller=='half':
        core = PointNetPlusHalfSized(num_classes=num_classes)
    elif smaller=='quarter':
        core = PointNetPlusQuarterSized(num_classes=num_classes)
    if normalize:
        core = nn.Sequential(BatchNormalizeScale(), core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model

def _build_pointnetplus_euclidean(num_classes=10, normalize=True, deterministic_fps=False,smaller=None):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    if smaller is None:
        core = PointNetPlus(num_classes=num_classes)
    elif smaller=='half':
        core = PointNetPlusHalfSized(num_classes=num_classes)
    elif smaller=='quarter':
        core = PointNetPlusQuarterSized(num_classes=num_classes)
    if normalize:
        core = nn.Sequential(BatchNormalizeScaleEuclidean(), core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model

# Change: Norm -> PCA (randomize=False)
def _build_pointnetplus_pca(
    num_classes=10,
    normalize=True,
    deterministic_fps=False,
    ensure_proper_rotation=True,
    sort=False,
    max_points=-1,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = []
    if normalize:
        pre.append(BatchNormalizeScale())
    pre.append(
        NormalizeRotationVectorizedModule(
            max_points=max_points,
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=False,
        )
    )
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model

# Change: Norm -> PCA (randomize=True)
def _build_pointnetplus_pca_randomize(
    num_classes=10,
    normalize=True,
    deterministic_fps=False,
    ensure_proper_rotation=True,
    sort=False,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = []
    if normalize:
        pre.append(BatchNormalizeScale())
    pre.append(
        NormalizeRotationVectorizedModule(
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=True,
        )
    )
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model

# New: PCA -> Norm (randomize=True)
def _build_pointnetplus_pca_then_norm_randomize(
    num_classes=10,
    normalize=True,
    deterministic_fps=False,
    ensure_proper_rotation=True,
    sort=False,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = [
        NormalizeRotationVectorizedModule(
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=True,
        )
    ]
    if normalize:
        pre.append(BatchNormalizeScale())
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
                print("Set deterministic FPS in SAModule before training")
    return model

# New: PCA -> Norm (randomize=True)
def _build_pointnetplus_pca_then_norm_randomize_euclidean(
    num_classes=10,
    normalize=True,
    deterministic_fps=False,
    ensure_proper_rotation=True,
    sort=False,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = [
        NormalizeRotationVectorizedModule(
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=True,
        )
    ]
    if normalize:
        pre.append(BatchNormalizeScaleEuclidean())
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
                print("Set deterministic FPS in SAModule before training")
    return model


class _RISurfConvLogitsOnly(nn.Module):
    """Wrap RISurfConv model (which returns (logits, features)) to expose only logits."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        logits, _ = self.base(x)
        return logits

def _build_risurfconv(num_classes=10, n=1, normal_channel=False):
    if not _RISURFCONV_AVAILABLE:
        raise ImportError("RISurfConv modules not available")
    base = get_risurfconv_model(num_class=num_classes, n=n, normal_channel=normal_channel)
    wrapped = _RISurfConvLogitsOnly(base)
    return wrapped

def get_modelnet_architectures():
    return [
        "pointnetplus",
        "pointnetplus_pca",
        "pointnetplus_pca_randomize",
        "pointnetplus_pca_then_norm_randomize",
        "pca_randomize",
        "risurfconv",
        # scaled variants (note: shrinking PointNet++ internals requires knowledge of its API;
        # these currently route to the same builder. Extend if PointNetPlus supports width args.)
        "pointnetplus_half",
        "pointnetplus_quarter",
    ]

def get_possible_architectures(dataset_info):
    name = dataset_info.name.lower()
    if name in ["mnist", "rotatedmnist", "rotated_mnist"]:
        from .mnist_networks import get_mnist_architectures
        return get_mnist_architectures()
    if name in ["biggermnist", "bigger_mnist"]:
        from .bigger_mnist_networks import get_bigger_mnist_architectures
        return get_bigger_mnist_architectures()
    if name == "emnist":
        from .extended_mnist_networks import get_extended_mnist_architectures
        return get_extended_mnist_architectures()
    if name in ["biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        from .bigger_extended_mnist_networks import get_bigger_extended_mnist_architectures
        return get_bigger_extended_mnist_architectures()
    if name == "coil100":
        from .coil100_networks import get_coil100_architectures
        return get_coil100_architectures()
    if name in ["modelnet", "modelnet10"]:
        return get_modelnet_architectures()
    if name == "si_score":
        from .si_network import get_si_network_architectures
        return get_si_network_architectures()
    if name == "tu_berlin":
        return ["bi_lstm", "bi_lstm_one_hot"]
    raise ValueError(f"Unknown dataset: {dataset_info.name}")



def get_network(dataset_info, architecture, num_classes=None, num_rotations=8):
    name = dataset_info.name.lower()
    num_classes = dataset_info.num_classes if num_classes is None else num_classes
    if name in ["mnist", "rotatedmnist", "rotated_mnist"]:
        from .mnist_networks import get_mnist_network
        return get_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    if name in ["biggermnist", "bigger_mnist"]:
        from .bigger_mnist_networks import get_bigger_mnist_network
        return get_bigger_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    if name == "emnist":
        from .extended_mnist_networks import get_extended_mnist_network
        return get_extended_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    if name in ["biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        from .bigger_extended_mnist_networks import get_bigger_extended_mnist_network
        return get_bigger_extended_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    if name == "coil100":
        from .coil100_networks import get_coil100_network
        return get_coil100_network(architecture, num_classes=num_classes)
    if name in ["modelnet", "modelnet10"]:
        a = architecture.lower()
        if a == "pointnetplus":
            return _build_pointnetplus(num_classes=num_classes)
        if a == "pointnetplus_euclidean":
            return _build_pointnetplus_euclidean(num_classes=num_classes)
        if a == "pointnetplus_pca_then_norm_randomize_euclidean":
            return _build_pointnetplus_pca_then_norm_randomize_euclidean(num_classes=num_classes,sort=True)
        if a == "pointnetplus_half":
            # TODO: if PointNetPlus supports a width/scale arg, call it here with reduced channels.
            return _build_pointnetplus(num_classes=num_classes,smaller='half')
        if a == "pointnetplus_quarter":
            # TODO: reduce internal widths when PointNetPlus supports it
            return _build_pointnetplus(num_classes=num_classes,smaller='quarter')
        if a == "pointnetplus_pca":
            return _build_pointnetplus_pca(num_classes=num_classes)
        if a in ["pointnetplus_pca_randomize", "pca_randomize", "pca randomize"]:
            return _build_pointnetplus_pca_randomize(num_classes=num_classes)
        if a in ["pointnetplus_pca_then_norm_randomize", "pca_then_norm_randomize"]:
            return _build_pointnetplus_pca_then_norm_randomize(num_classes=num_classes)
        if a in ["pointnetplus_pca_then_norm_randomize_sort", "pca_then_norm_randomize"]:
            return _build_pointnetplus_pca_then_norm_randomize(num_classes=num_classes,sort=True)
        if a == "risurfconv":
            return _build_risurfconv(num_classes=num_classes)
        raise ValueError(f"Unknown ModelNet architecture: {architecture}")
    if name == "si_score" or name == "si_score_resnet" or name == "si_score_resnet_no_crop" or name == "si_score_vit_no_crop":
        from .si_network import get_si_network
        return get_si_network(architecture, num_classes=num_classes)
    if name == "tu_berlin":
        a = architecture.lower()
        if a in ("bi_lstm", "bi_lstm_half", "bi_lstm_quarter"):
            # default hidden size 256, halves -> 128, quarters -> 64
            hidden = 256
            if a == "bi_lstm_half":
                hidden = 128
            elif a == "bi_lstm_quarter":
                hidden = 64
            preprocess_module = NormalizeToRangeBatched()
            main = BILSTMSKETCHClassifier(
                input_size=3,
                hidden_size=hidden,
                num_layers=2,
                num_classes=num_classes if num_classes is not None else 250,
                rnn_type='lstm',
                preprocess_module=preprocess_module,
                num_mlp_layers=1,
                dropout=0.5,
                augmentation=StrokeAugment()
            )
            return main
        elif a == "bi_lstm_pca":
            preprocess_module = nn.Sequential(
                NormalizeRotationStrokeBatched(),
                NormalizeToRangeBatched(),
            )
            main = BILSTMSKETCHClassifier(
                input_size=3,
                hidden_size=256,
                num_layers=2,
                num_classes=num_classes if num_classes is not None else 250,
                rnn_type='lstm',
                preprocess_module=preprocess_module,
                num_mlp_layers=1,
                dropout=0.5,
                augmentation=StrokeAugment()
            )
            return main
        elif a in ("bi_lstm_one_hot", "bi_lstm_one_hot_half", "bi_lstm_one_hot_quarter"):
            hidden = 256
            if a == "bi_lstm_one_hot_half":
                hidden = 128
            elif a == "bi_lstm_one_hot_quarter":
                hidden = 64
            preprocess_module = nn.Sequential(
                ConvertToOneHotPenState(),
                NormalizeToRangeBatched()
            )
            main = BILSTMSKETCHClassifier(
                input_size=5,  # (dx, dy, p1, p2, p3)
                hidden_size=hidden,
                num_layers=2,
                num_classes=num_classes if num_classes is not None else 250,
                rnn_type='lstm',
                preprocess_module=preprocess_module,
                num_mlp_layers=1,
                dropout=0.5,
                augmentation=StrokeAugment()
            )
            return main
        raise ValueError(f"Unknown TU-Berlin architecture: {architecture}")
    raise ValueError(f"Unknown dataset: {dataset_info.name}")

def get_network_layer(dataset_info, architecture, index, num_classes=None, num_rotations=8):
    name = dataset_info.name.lower()
    num_classes = dataset_info.num_classes if num_classes is None else num_classes
    if name == "si_score" or name == "si_score_resnet":
        from .si_network import get_si_network_layer
        return get_si_network_layer(architecture, index, num_classes=num_classes)
    
    # Handle ModelNet architectures
    if name in ["modelnet", "modelnet10"]:
        a = architecture.lower()
        if a not in MODELNET_LAYER_MAPPINGS:
            raise ValueError(f"No layer mapping defined for ModelNet architecture: {architecture}")
        mapping = MODELNET_LAYER_MAPPINGS[a]
        if index not in mapping:
            raise ValueError(f"No layer mapping for index {index} in ModelNet architecture {architecture}")
        return mapping[index]

    # Handle TU Berlin architectures
    if name == "tu_berlin":
        a = architecture.lower()
        if a not in TU_BERLIN_LAYER_MAPPINGS:
            raise ValueError(f"No layer mapping defined for TU Berlin architecture: {architecture}")
        mapping = TU_BERLIN_LAYER_MAPPINGS[a]
        if index not in mapping:
            raise ValueError(f"No layer mapping for index {index} in TU Berlin architecture {architecture}")
        return mapping[index]

    # Handle FlexibleResNet architectures for various datasets
    if name in ["mnist", "rotatedmnist", "rotated_mnist"]:
        from .mnist_networks import get_mnist_network_layer
        return get_mnist_network_layer(architecture, index, num_classes=num_classes, num_rotations=num_rotations)
    if name in ["biggermnist", "bigger_mnist"]:
        from .bigger_mnist_networks import get_bigger_mnist_network_layer
        return get_bigger_mnist_network_layer(architecture, index, num_classes=num_classes, num_rotations=num_rotations)
    if name == "emnist":
        from .extended_mnist_networks import get_extended_mnist_network_layer
        return get_extended_mnist_network_layer(architecture, index, num_classes=num_classes, num_rotations=num_rotations)
    if name in ["biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        from .bigger_extended_mnist_networks import get_bigger_extended_mnist_network_layer
        return get_bigger_extended_mnist_network_layer(architecture, index, num_classes=num_classes, num_rotations=num_rotations)
    if name == "coil100":
        from .coil100_networks import get_coil100_network_layer
        return get_coil100_network_layer(architecture, index, num_classes=num_classes)
    
    raise ValueError(f"get_network_layer not implemented for dataset: {dataset_info.name}")


def get_max_layer_index(dataset_info, architecture, num_classes=None, num_rotations=8) -> int:
    """
    Tries to find the maximum valid layer index by calling get_network_layer until it fails.
    Returns the last successful index.
    """
    index = 0
    while True:
        try:
            # We only need to check if it runs without error
            get_network_layer(dataset_info, architecture, index, num_classes=num_classes, num_rotations=num_rotations)
            index += 1
        except (ValueError, KeyError, IndexError):
            # These exceptions are typically raised for an invalid index.
            # The last valid index was `index - 1`.
            return index - 1


class ToGeometric(nn.Module):
    def __init__(self, field_type: escnn_nn.FieldType):
        super().__init__()
        self.field_type = field_type
    def forward(self, x: torch.Tensor):
        if isinstance(x, escnn_nn.GeometricTensor):
            return x  # already geometric
        return escnn_nn.GeometricTensor(x, self.field_type)

class FromGeometric(nn.Module):
    def forward(self, x):
        return x.tensor if isinstance(x, escnn_nn.GeometricTensor) else x


class P4CNN(nn.Module):
    def __init__(self, num_classes=10,activation=escnn_nn.ReLU):
        super().__init__()
        r2_act = gspaces.rot2dOnR2(N=8)

        in_type = escnn_nn.FieldType(r2_act, [r2_act.trivial_repr])
        feat_type = escnn_nn.FieldType(r2_act, 20 * [r2_act.regular_repr])

        self.conv1 = escnn_nn.R2Conv(in_type, feat_type, kernel_size=3, padding=1, bias=False)
        self.bn1 = escnn_nn.InnerBatchNorm(feat_type)
        self.conv2 = escnn_nn.R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False)
        self.bn2 = escnn_nn.InnerBatchNorm(feat_type)
        self.conv3 = escnn_nn.R2Conv(feat_type, feat_type, kernel_size=7, bias=False)
        self.bn3 = escnn_nn.InnerBatchNorm(feat_type)

        # switched Max -> Avg for equivariant pooling
        self.pool1 = escnn_nn.PointwiseAvgPool(feat_type, 2)
        self.pool2 = escnn_nn.PointwiseAvgPool(feat_type, 2)

        # Proper equivariant activation (needs field type)
        self.act = activation(feat_type, inplace=True)

        self.gpool = escnn_nn.GroupPooling(feat_type)

        self.fc1 = nn.Linear(20, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = escnn_nn.GeometricTensor(x, self.conv1.in_type)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.gpool(x)              # GeometricTensor with trivial reps
        x = x.tensor.view(x.tensor.size(0), -1)  # (B,20)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def get_flexible_resnet_layer_mapping(blocks_per_stage,stages):
    """
    Generate layer mapping for FlexibleResNet based on its structure.
    Index 0 is always the final classifier 'fc'.
    Higher indices map to blocks in reverse order (later blocks have lower indices).
    
    Args:
        blocks_per_stage: List of number of blocks per stage or int for uniform blocks
    
    Returns:
        Dict mapping indices to (layer_name, capture_mode) tuples
    """
    if isinstance(blocks_per_stage, int):
        # If uniform blocks, we need to know how many stages - assume 3 stages for common cases
        blocks_per_stage = [blocks_per_stage] * stages
    
    mapping = {0: ("fc", "input")}
    
    index = 1
    # Go through stages in reverse order (last stage first)
    for stage_idx in reversed(range(len(blocks_per_stage))):
        num_blocks = blocks_per_stage[stage_idx]
        # Go through blocks in reverse order within each stage (last block first)
        for block_idx in reversed(range(num_blocks)):
            layer_name = f"stages.{stage_idx}.{block_idx}.act"
            mapping[index] = (layer_name, "input")
            index += 1
    
    return mapping

# Layer mappings for ModelNet architectures
MODELNET_LAYER_MAPPINGS = {
    "pointnetplus": {
        0: ("model.1.mlp.6", "input"),
        1: ("model.1.mlp.4", "input"),
        2: ("model.1.mlp.3", "input"),
        3: ("model.1.mlp.1", "input"),
        4: ("model.1.mlp.0", "input"),
    },
    # PCA then Norm then Core -> core at index 2
    "pointnetplus_pca": {
        0: ("model.2.mlp.6", "input"),
        1: ("model.2.mlp.4", "input"),
        2: ("model.2.mlp.3", "input"),
        3: ("model.2.mlp.1", "input"),
        4: ("model.2.mlp.0", "input"),
    },
    # Identical mapping; sequence is [PCA, Norm, Core] so core is at index 2
    "pointnetplus_pca_randomize": {
        0: ("model.2.mlp.6", "input"),
        1: ("model.2.mlp.4", "input"),
        2: ("model.2.mlp.3", "input"),
        3: ("model.2.mlp.1", "input"),
        4: ("model.2.mlp.0", "input"),
    },
    # New: PCA -> Norm (randomize) has same core index
    "pointnetplus_pca_then_norm_randomize": {
        0: ("model.2.mlp.6", "input"),
        1: ("model.2.mlp.4", "input"),
        2: ("model.2.mlp.3", "input"),
        3: ("model.2.mlp.1", "input"),
        4: ("model.2.mlp.0", "input"),
    },
    "risurfconv": {
        0: ("base.classifier", "input"),
        1: ("base.conv4", "output"),
        2: ("base.conv3", "output"),
        3: ("base.conv2", "output"),
        4: ("base.conv1", "output"),
    },
}

# Layer mappings for TU Berlin architectures
TU_BERLIN_LAYER_MAPPINGS = {
    "bi_lstm": {
        0: ("classifier.4", "input"),  # last linear layer input
        1: ("classifier.1", "input"),  # GELU input
        2: ("classifier.0", "input"),  # first linear layer input
    },
    "bi_lstm_one_hot": {
        0: ("classifier.4", "input"),  # last linear layer input
        1: ("classifier.1", "input"),  # GELU input
        2: ("classifier.0", "input"),  # first linear layer input
    }
}


class NormalizeRotationStrokeBatched(nn.Module):
    """
    PCA-based orientation normalization for 2D stroke sequences.

    Aligns each stroke sequence to its principal axes by rotating the
    absolute coordinates, then converts back to relative deltas.
    Includes options for deterministic eigenvector signing and ensuring
    proper rotation (determinant = +1).
    """

    def __init__(self,
                 max_points: int = -1,
                 sort: bool = False,
                 ensure_proper_rotation: bool = True,
                 fix_sign: bool = False):
        """
        Args:
            max_points (int): The maximum number of points to use for PCA
                from each stroke. If -1, uses all points. Defaults to -1.
            sort (bool): If True, sorts eigenvectors by their corresponding
                eigenvalues in descending order. Defaults to False.
            ensure_proper_rotation (bool): If True, ensures the final
                transformation matrix is a proper rotation (determinant = +1),
                preventing reflections. Defaults to True.
            fix_sign (bool): If True, flips eigenvectors so they align
                consistently with the centroid of each stroke sequence.
                Defaults to False.

        """
        super().__init__()
        self.max_points = max_points
        self.sort = sort
        self.ensure_proper_rotation = ensure_proper_rotation
        self.fix_sign = fix_sign
        self.randomize = True  # for clarity

    @torch.no_grad()
    def forward(self, stroke_seq: torch.Tensor, randomize: bool = None,
                use_svd_for_rotation: bool = False) -> torch.Tensor:
        """
        Args:
            stroke_seq: [B, N, 3] last dim (dx, dy, pen_state)
            randomize: override self.randomize (if None, uses self.randomize)
            use_svd_for_rotation: if True, build orthogonal transform via SVD (U @ Vh)
        Returns:
            [B, N, 3]
        """
        if randomize is None:
            randomize = self.randomize

        B, N, C = stroke_seq.shape
        device = stroke_seq.device
        orig_dtype = stroke_seq.dtype
        compute_dtype = torch.float64  # enforce float64 for all internal math

        # Convert inputs to float64 for computation (but keep pen_state separately)
        coords = stroke_seq[..., :2].to(device=device, dtype=compute_dtype)  # [B,N,2]
        pen_state = stroke_seq[..., 2:].to(device=device, dtype=orig_dtype)  # preserved in original dtype

        # Absolute coordinates
        abs_xy = torch.cumsum(coords, dim=1)  # [B,N,2] in float64

        # Subsample if requested
        if self.max_points <= 0 or N <= self.max_points:
            pos_sel = abs_xy  # [B,N,2]
        else:
            # indices: [B, max_points]
            idx = torch.stack([
                torch.randperm(N, device=device)[:self.max_points]
                for _ in range(B)
            ], dim=0)  # long
            pos_sel = torch.gather(abs_xy, 1, idx.unsqueeze(-1).expand(-1, -1, 2))  # [B, max_points, 2]

        # Mean & centered
        mu = pos_sel.mean(dim=1, keepdim=True)  # [B,1,2]
        centered = pos_sel - mu  # [B, N_sel, 2]

        # Covariance (2x2): (1/n) X^T X
        n_sel = centered.size(1)
        # avoid division by zero, but n_sel should be >=1
        Cmat = torch.bmm(centered.transpose(1, 2), centered) / max(n_sel, 1)  # [B,2,2], float64

        # Regularize covariance to avoid degeneracy (scale eps by trace)
        trace = (torch.diagonal(Cmat, dim1=1, dim2=2).sum(dim=1) / 2.0).clamp(min=1e-12)  # [B]
        eps = (1e-12 + 1e-9 * trace).view(B, 1, 1).to(device=device, dtype=compute_dtype)
        Cmat = Cmat + eps * torch.eye(2, device=device, dtype=compute_dtype).unsqueeze(0)

        # Use symmetric eigen decomposition (more accurate for covariance)
        # eigh returns eigenvalues ascending
        e_vals, e_vecs = torch.linalg.eigh(Cmat)  # e_vecs: [B,2,2], float64

        # Optionally re-order eigenvectors to descending eigenvalue magnitude
        if self.sort:
            idx = e_vals.argsort(dim=-1, descending=True)  # [B,2]
            idx_expand = idx.unsqueeze(1).expand(-1, 2, -1)  # [B,2,2]
            e_vecs = e_vecs.gather(dim=2, index=idx_expand)

        # Optionally use SVD to produce a robust orthogonal matrix
        if use_svd_for_rotation:
            U, S, Vh = torch.linalg.svd(Cmat)  # U: [B,2,2], Vh: [B,2,2]
            R = U @ Vh  # orthogonal, [B,2,2]
            e_vecs = R

        # Randomize (augment) or deterministic sign fixing
        if randomize and self.training:
            # sign flips: generate in float64 then apply
            signs = (torch.randint(0, 2, (B, 2), device=device) * 2 - 1).to(dtype=compute_dtype)  # [-1,1]
            e_vecs = e_vecs * signs.unsqueeze(1)  # broadcast over rows

            if not self.sort:
                swap = torch.rand(B, device=device) > 0.5  # boolean mask
                if swap.any():
                    # swap columns 0 and 1 for those batches
                    swapped = e_vecs.clone()
                    swapped[swap] = e_vecs[swap][:, :, [1, 0]]
                    e_vecs = swapped
        elif self.fix_sign:
            mu_centered = mu.squeeze(1).to(dtype=compute_dtype)  # [B,2]
            dots = torch.einsum("bi,bij->bj", mu_centered, e_vecs)  # [B,2]
            signs = torch.where(dots < 0, -1.0, 1.0).to(dtype=compute_dtype)  # [B,2]
            e_vecs = e_vecs * signs.unsqueeze(1)

        # Ensure proper rotation (determinant +1)
        if self.ensure_proper_rotation:
            det = torch.linalg.det(e_vecs)  # [B]
            flip_mask = det < 0
            if flip_mask.any():
                # flip second column for 2D to correct determinant
                e_vecs[flip_mask, :, 1] *= -1.0

        # Rotate absolute coordinates: abs_xy [B,N,2] * e_vecs [B,2,2] -> [B,N,2]
        rotated_abs = torch.bmm(abs_xy.to(dtype=compute_dtype), e_vecs)  # [B,N,2], float64

        # Convert back to relative deltas
        rel_xy = torch.empty_like(rotated_abs)
        rel_xy[:, 0] = rotated_abs[:, 0]
        if N > 1:
            rel_xy[:, 1:] = rotated_abs[:, 1:] - rotated_abs[:, :-1]

        # Concatenate pen_state back (convert coords back to original dtype)
        rel_xy_out = rel_xy.to(dtype=orig_dtype)
        out = torch.cat([rel_xy_out, pen_state], dim=-1)  # [B,N, C] where C >=3

        return out


class NormalizeToRangeBatched(nn.Module):
    """
    Normalize absolute coordinates to [-128,128] range for stroke sequences.
    """
    def __init__(self, eps: float = 1e-8,scale_abs=False):
        super().__init__()
        self.eps = eps
        self.scale_abs = scale_abs

    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        # stroke_seq: [B, N, 3] where last dim is (dx, dy, pen_state)
        abs_xy = torch.cumsum(stroke_seq[..., :2], dim=1)           # [B, N, 2]
        center = abs_xy.mean(dim=1, keepdim=True)                   # [B, 1, 2]
        centered = abs_xy - center                                  # [B, N, 2]

        max_abs = centered.abs().amax(dim=(1, 2), keepdim=True)     # [B, 1, 1]
        scale = 128.0 / (max_abs + self.eps)                        # [B, 1, 1]
        scaled =centered * scale

        # Convert scaled absolute coordinates back to deltas
        rel_xy = torch.zeros_like(scaled)
        rel_xy[:, 0] = scaled[:, 0]  # First point becomes the delta from origin
        rel_xy[:, 1:] = scaled[:, 1:] - scaled[:, :-1]              # [B, N, 2]

        max_abs_rel = rel_xy.abs().amax(dim=(1, 2), keepdim=True)     # [B, 1, 1]


        pen_state = stroke_seq[..., 2:]                             # [B, N, 1] or [B, N, 3]
        return torch.cat([rel_xy, pen_state], dim=-1)             # [B, N, 3] or [B, N, 5]


class ConvertToOneHotPenState(nn.Module):
    """
    Convert single pen state to one-hot encoding (p1, p2, p3).
    
    Input: (dx, dy, pen_state) where pen_state: 0=pen down, 1=pen up (end of stroke)
    Output: (dx, dy, p1, p2, p3) where:
        - p1=1: pen touching paper (drawing continues) - pen_state=0
        - p2=1: pen lifted after this point (stroke end) - pen_state=1 (last drawn point)
        - p3=1: drawing has ended (not rendered) - point AFTER last pen_state=1 and all subsequent
    
    The mask is inferred from the pen states: points after the last pen_state=1 get p3=1.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stroke_seq: [B, N, 3] with (dx, dy, pen_state)
        
        Returns:
            [B, N, 5] with (dx, dy, p1, p2, p3)
        """
        dx_dy = stroke_seq[..., :2]  # [B, N, 2]
        pen_state = stroke_seq[..., 2]  # [B, N]
        
        batch_size, seq_len = stroke_seq.shape[0], stroke_seq.shape[1]
        
        # Find the last occurrence of pen_state=1 in each sequence
        # This marks the last drawn point
        pen_up_mask = (pen_state == 1).float()  # [B, N]
        
        # Create position indices
        position_indices = torch.arange(seq_len, device=stroke_seq.device).unsqueeze(0).expand(batch_size, -1)  # [B, N]
        
        # For each sequence, find the index of the last pen_state=1
        # Set to -1 if no pen_state=1 exists (all drawing)
        pen_up_positions = position_indices * pen_up_mask  # [B, N]
        last_pen_up_idx = pen_up_positions.max(dim=1, keepdim=True).values  # [B, 1]
        
        # Check if there's at least one pen_state=1 in each sequence
        has_pen_up = pen_up_mask.sum(dim=1, keepdim=True) > 0  # [B, 1]
        
        # Points AFTER the last pen_state=1 have p3=1
        is_after_last_pen_up = position_indices > last_pen_up_idx  # [B, N]
        
        # Only set p3=1 if there was actually a pen_state=1 in the sequence
        is_after_last_pen_up = is_after_last_pen_up & has_pen_up  # [B, N]
        
        # Create one-hot encoding
        # p1: pen down (pen_state=0)
        p1 = (pen_state == 0).float()
        
        # p2: pen up (pen_state=1) - these are the last points of strokes
        p2 = (pen_state == 1).float()
        
        # p3: points AFTER the last pen_state=1 (not rendered)
        p3 = is_after_last_pen_up.float()
        
        return torch.cat([dx_dy, p1.unsqueeze(-1), p2.unsqueeze(-1), p3.unsqueeze(-1)], dim=-1)


class BILSTMSKETCHClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 rnn_type: str = 'lstm',
                 pool_type: str = 'attn',
                 dropout: float = 0.3,
                 num_mlp_layers: int = 1,
                 preprocess_module: nn.Module = None,
                 augmentation: nn.Module = None,
                 attn_dropout: float = 0.2):
        super().__init__()

        assert rnn_type in ['lstm', 'gru']
        assert pool_type in ['attn', 'max', 'last']

        self.hidden_size = hidden_size
        self.pool_type = pool_type
        self.input_size = input_size

        # Preprocessing (optional)
        self.preprocess = preprocess_module or nn.Identity()

        # RNN (bidirectional)
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=input_size,  # 3 for (dx,dy,pen_state) or 5 for (dx,dy,p1,p2,p3)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        rnn_output_size = hidden_size * 2

        # Attention pooling
        if self.pool_type == 'attn':
            self.attention = nn.Linear(rnn_output_size, 1, bias=False)
            self.attn_dropout = nn.Dropout(attn_dropout)

        self.augmentation = augmentation
        self.aug=True

        # Classifier (MLP)
        mlp_layers = []
        if num_mlp_layers > 0:
            for i in range(num_mlp_layers):
                in_dim = rnn_output_size if i == 0 else hidden_size * 2
                out_dim = hidden_size * 2
                mlp_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.LayerNorm(out_dim),
                    nn.Dropout(dropout),
                ])
            mlp_layers.append(nn.Linear(hidden_size * 2, num_classes))
            self.classifier = nn.Sequential(*mlp_layers)
        else:
            self.classifier = nn.Linear(rnn_output_size, num_classes)

    def _attention_pooling(self, rnn_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # rnn_out: (batch, seq_len, hidden*2)
        attn_scores = self.attention(rnn_out).squeeze(-1)  # (batch, seq_len)

        # Mask padded positions: set them to -inf before softmax
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        attn_probs = self.attn_dropout(attn_probs)  # apply dropout on attention distribution

        context = (rnn_out * attn_probs).sum(dim=1)  # weighted sum
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size+1) with format (...features, mask)
        For input_size=3: (dx, dy, pen_state, mask)
        For input_size=5: (dx, dy, p1, p2, p3, mask)
        """
        if self.training and self.augmentation is not None and self.aug:
            x = self.augmentation(x)
        inputs, mask = x[..., :-1], x[..., -1].long()  # (B, T, input_size), (B, T)
        lengths = mask.sum(dim=1).cpu()

        # Preprocess coordinates + pen state(s)
        # ConvertToOneHotPenState no longer needs mask as a separate argument
        inputs = self.preprocess(inputs)
        
        # inputs now has shape [B, T, input_size] (5 for one-hot, 3 for regular)
        
        # Pack sequence for RNN
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        rnn_out, hidden = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=inputs.size(1))

        # Pooling
        if self.pool_type == 'attn':
            pooled = self._attention_pooling(rnn_out, mask)
        elif self.pool_type == 'max':
            rnn_out = rnn_out.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            pooled, _ = torch.max(rnn_out, dim=1)
        else:  # last
            h_n = hidden[0] if isinstance(self.rnn, nn.LSTM) else hidden
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return self.classifier(pooled)


class StrokeAugment(nn.Module):
    """
    A PyTorch module for applying non-affine augmentations to batched stroke data.

    This class applies a sequence of augmentations suitable for sequence-based
    drawing data (like handwriting or sketches). It first converts the relative
    offsets to absolute coordinates, applies the augmentations, and then converts
    the coordinates back to relative offsets.

    Augmentations:
    1.  **Jitter**: Adds Gaussian noise to each point to simulate hand tremor.
    2.  **Elastic Deformation**: Applies a smooth, random warping field to the
        entire sequence
    3.  **Stroke Dropout**: Randomly removes entire strokes (sequences of points
        drawn with the pen down) to train the model on incomplete data.
    """
    def __init__(self,
                 jitter_sigma: float = 1.0,
                 stroke_dropout_prob: float = 0.05,
                 elastic_deformation_alpha: float = 2.0,
                 elastic_deformation_sigma: float = 0.08):
        """
        Initializes the augmentation module.

        Args:
            jitter_sigma (float): Standard deviation of Gaussian noise for jitter.
            stroke_dropout_prob (float): Probability of dropping an entire stroke.
            elastic_deformation_alpha (float): Scaling factor for the magnitude of
                                               elastic deformation displacement.
            elastic_deformation_sigma (float): Standard deviation of the Gaussian
                                               kernel for smoothing the deformation field,
                                               as a fraction of the sequence length.
        """
        super().__init__()
        self.jitter_sigma = jitter_sigma
        self.stroke_dropout_prob = stroke_dropout_prob
        self.elastic_deformation_alpha = elastic_deformation_alpha
        self.elastic_deformation_sigma = elastic_deformation_sigma

    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        """
        Applies the augmentations to a batch of stroke sequences.

        Args:
            stroke_seq (torch.Tensor): A tensor of shape (B, T, 4) containing
                                       (dx, dy, pen_state, mask).

        Returns:
            torch.Tensor: The augmented stroke sequence tensor.
        """
        if not self.training:
            return stroke_seq
        x, y, pen, mask = stroke_seq.unbind(dim=-1)
        B, T = x.shape
        valid_mask = (mask > 0).float()

        # Convert relative deltas to absolute coordinates
        abs_x = torch.cumsum(x * valid_mask, dim=1)
        abs_y = torch.cumsum(y * valid_mask, dim=1)

        # 1. Apply Jitter
        if self.jitter_sigma > 0:
            noise = torch.randn_like(abs_x) * self.jitter_sigma
            abs_x += noise * valid_mask
            abs_y += noise * valid_mask

        # 2. Apply Elastic Deformation
        if self.elastic_deformation_alpha > 0 and self.elastic_deformation_sigma > 0:
            k_sigma_pixels = T * self.elastic_deformation_sigma
            kernel_size = int(2 * round(k_sigma_pixels * 3)) + 1

            t_range = torch.arange(kernel_size, device=x.device, dtype=torch.float32) - (kernel_size - 1) // 2
            kernel = torch.exp(-t_range**2 / (2 * k_sigma_pixels**2))
            kernel = (kernel / kernel.sum()).view(1, 1, -1)

            displacement_x = torch.randn(B, 1, T, device=x.device)
            displacement_y = torch.randn(B, 1, T, device=x.device)

            padding = (kernel_size - 1) // 2
            smoothed_dx = F.conv1d(displacement_x, kernel, padding=padding).squeeze(1)
            smoothed_dy = F.conv1d(displacement_y, kernel, padding=padding).squeeze(1)

            smoothed_dx = (smoothed_dx / (smoothed_dx.std() + 1e-9)) * self.elastic_deformation_alpha
            smoothed_dy = (smoothed_dy / (smoothed_dy.std() + 1e-9)) * self.elastic_deformation_alpha

            abs_x += smoothed_dx * valid_mask
            abs_y += smoothed_dy * valid_mask

        # Keep track of the final augmented absolute positions
        final_abs_x, final_abs_y = abs_x, abs_y
        final_pen = pen.clone()

        # 3. Apply Stroke Dropout
        if self.stroke_dropout_prob > 0:
            pen_lifted = (pen > 0).float()
            pen_lifted[:, -1] = 1.0

            stroke_ids = torch.cumsum(pen_lifted, dim=1).long() - 1
            stroke_ids.clamp_(min=0)

            num_strokes = stroke_ids[:, -1] + 1
            max_strokes = int(num_strokes.max())

            drop_decisions = torch.rand(B, max_strokes, device=x.device) < self.stroke_dropout_prob

            point_drop_mask = torch.gather(drop_decisions, 1, stroke_ids)
            point_drop_mask[:, 0] = False

            augmented_mask_bool = valid_mask.bool() & ~point_drop_mask

            final_pen[point_drop_mask] = 1.0

            # **FIX**: Propagate the last valid coordinate through dropped points
            # to ensure the next stroke starts from the correct position.
            # This is a vectorized forward-fill operation.
            indices = torch.arange(T, device=x.device).repeat(B, 1)
            indices[~augmented_mask_bool] = -1 # Invalidate dropped points' indices

            last_valid_indices = torch.cummax(indices, dim=1).values
            last_valid_indices.clamp_(min=0) # Ensure indices are valid

            final_abs_x = torch.gather(abs_x, 1, last_valid_indices)
            final_abs_y = torch.gather(abs_y, 1, last_valid_indices)

        # Convert corrected absolute coordinates back to relative deltas
        rel_x = torch.zeros_like(final_abs_x)
        rel_y = torch.zeros_like(final_abs_y)
        rel_x[:, 0] = final_abs_x[:, 0]
        rel_y[:, 0] = final_abs_y[:, 0]
        rel_x[:, 1:] = final_abs_x[:, 1:] - final_abs_x[:, :-1]
        rel_y[:, 1:] = final_abs_y[:, 1:] - final_abs_y[:, :-1]

        # The original mask is returned, as the effect of dropout is now
        # correctly encoded in the coordinate deltas and pen states.
        return torch.stack([rel_x, rel_y, final_pen, mask], dim=-1)


def get_encoder_for_resnet(model, dim=512, vae=False):
    """
    Creates an encoder and decoder for a FlexibleResNet model.
    The encoder compresses the input to a latent space of dimension 'dim'.
    The decoder reconstructs the input from the latent space.
    For VAE, the encoder returns (mu, log_var) for reparameterization.
    Models are scaled down to approximate half the FLOPs of the original model.
    """
    # More robust check that works even with reloaded modules
    if not (isinstance(model, FlexibleResNet) or
            type(model).__name__ == 'FlexibleResNet' or
            hasattr(model, 'stem_conv') and hasattr(model, 'stages')):
        raise ValueError(f"Model must be an instance of FlexibleResNet, got {type(model)}")

    # Extract model structure
    in_channels = model.stem_conv.in_channels
    stem_stride = model.stem_conv.stride[0]
    activation = type(model.stem_act)

    # Get channels for stem and each stage
    stem_channels = model.stem_conv.out_channels
    stage_channels = []
    for stage in model.stages:
        # The out_channels for a stage is the out_channels of its first block's final conv
        stage_channels.append(
            stage[0].conv3.out_channels if hasattr(stage[0], 'conv3') else stage[0].conv2.out_channels)

    blocks_per_stage = [len(stage) for stage in model.stages]

    # Scale down for half FLOPs: divide channels by sqrt(2)
    # FLOPs ~ channels^2, so halving FLOPs means dividing channels by sqrt(2)
    scale_factor = math.sqrt(2)

    # Scale down channels
    encoder_stem_channels = max(8, int(stem_channels / scale_factor))
    encoder_stage_channels = [max(8, int(ch / scale_factor)) for ch in stage_channels]

    # FlexibleResNet expects channels to be ONLY the stage channels, not including stem
    # The stem channels are set separately via the first element or handled internally
    # Let's check if channels should include stem or not by looking at original model
    encoder_channels = encoder_stage_channels  # Try without stem first

    # Also reduce depth slightly
    # blocks_per_stage should have same length as channels list
    encoder_blocks = [max(1, b // 2) for b in blocks_per_stage]

    # Encoder: Smaller ResNet-like network
    encoder = FlexibleResNet(
        channels=encoder_channels,
        blocks_per_stage=encoder_blocks,
        num_classes=dim,
        in_channels=in_channels,
        activation=activation,
        stem_stride=stem_stride
    )

    # The feature extractor part of the encoder (everything except final FC)
    feature_extractor = nn.Sequential(
        encoder.stem_conv,
        encoder.stem_bn,
        encoder.stem_act,
        *encoder.stages,
        encoder.head_pool,
        nn.Flatten(1)
    )

    # Calculate the actual output dimension after pooling
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 32, 32)  # Assume 32x32 input
        final_feature_dim = feature_extractor(dummy_input).shape[1]

    if vae:
        # For VAE, add two heads: mu and log_var
        mu_head = nn.Linear(final_feature_dim, dim)
        log_var_head = nn.Linear(final_feature_dim, dim)

        class VAEEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = feature_extractor
                self.mu_head = mu_head
                self.log_var_head = log_var_head

            def forward(self, x):
                features = self.feature_extractor(x)
                return self.mu_head(features), self.log_var_head(features)

        encoder_model = VAEEncoder()
    else:
        # Standard encoder: single head
        fc = nn.Linear(final_feature_dim, dim)
        encoder_model = nn.Sequential(feature_extractor, fc)

    # Decoder: Reverse architecture with transposed convolutions
    decoder_layers = []

    # Calculate spatial size after encoding (depends on stride and pooling)
    total_stride = stem_stride * (2 ** len(model.stages))  # Assuming stride 2 per stage
    initial_spatial_size = 32 // total_stride  # Assuming 32x32 input
    initial_spatial_size = max(1, initial_spatial_size)

    # Start with linear layer to expand latent to spatial feature map
    initial_features = encoder_stage_channels[-1]
    decoder_layers.append(nn.Linear(dim, initial_features * initial_spatial_size * initial_spatial_size))
    decoder_layers.append(nn.Unflatten(1, (initial_features, initial_spatial_size, initial_spatial_size)))

    # Reverse the encoder stages
    decoder_stage_channels = encoder_stage_channels[::-1]

    # Build upsampling stages (reverse of encoder stages)
    for i in range(len(decoder_stage_channels) - 1):
        in_ch = decoder_stage_channels[i]
        out_ch = decoder_stage_channels[i + 1]

        # Upsample by 2x (reverse of stride 2 downsampling in encoder)
        decoder_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.BatchNorm2d(out_ch))
        decoder_layers.append(activation())

    # Final upsampling to match stem stride
    final_ch = encoder_stem_channels
    if len(decoder_stage_channels) > 0:
        decoder_layers.append(
            nn.ConvTranspose2d(decoder_stage_channels[-1], final_ch, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.BatchNorm2d(final_ch))
        decoder_layers.append(activation())

    # Final conv to get back to input channels (no stride change)
    if stem_stride > 1:
        decoder_layers.append(nn.ConvTranspose2d(final_ch, in_channels, kernel_size=stem_stride * 2, stride=stem_stride,
                                                 padding=stem_stride // 2))
    else:
        decoder_layers.append(nn.Conv2d(final_ch, in_channels, kernel_size=3, padding=1))

    # Optional: Add sigmoid/tanh activation for image reconstruction
    # decoder_layers.append(nn.Sigmoid())  # Uncomment if images are normalized to [0, 1]

    decoder = nn.Sequential(*decoder_layers)

    return encoder_model, decoder

def make_deterministic(model,random=False,verbose=True):
    model.aug = False
    for module in model.modules():
        if isinstance(module, SAModule):
            module.random_start = random
            if verbose:
                print(f"Set random_start={random} for {module.__class__.__name__}")


def find_last_linear_layer(model: torch.nn.Module) -> torch.nn.Linear:
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Linear):
            return m
    raise ValueError("No nn.Linear layer found in the model")


class PreActBlock(nn.Module):
    """
    Wraps a BasicBlock to return the pre-activation sum (y + shortcut)
    i.e. omits the final self.act(...) so a downstream head can apply it.
    """
    def __init__(self, block: BasicBlock):
        super().__init__()
        # reuse convolutional / BN / proj modules from the original block
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.proj = block.proj
        # keep a local pre-activation (same type as original block.act)
        self._pre_act = type(block.act)()

    def forward(self, x):
        y = self._pre_act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        shortcut = x if self.proj is None else self.proj(x)
        return y + shortcut

def split_flexible_resnet_for_ash(model, split_pos=0):
    """
    Split FlexibleResNet so ASH runs before the final block activation.
    Assumes every block exposes `act` and moves the last block's activation
    into the head by replacing the block with `PreActBlock`.
    """
    if not isinstance(model, FlexibleResNet):
        raise ValueError("Model must be an instance of FlexibleResNet")

    num_stages = len(model.stages)
    if split_pos < 0 or split_pos > num_stages + 1:
        raise ValueError(f"split_pos must be between 0 and {num_stages + 1}")

    # split_pos == 0: full backbone (after pool)
    if split_pos == 0:
        backbone = nn.Sequential(
            model.stem_conv,
            model.stem_bn,
            model.stem_act,
            *model.stages,
            model.head_pool
        )
        head = nn.Sequential(nn.Flatten(1), model.fc)
        return backbone, head

    # Determine how many stages to keep in backbone
    stages_to_backbone = num_stages - split_pos + 1
    backbone_stages = [s for s in model.stages[:stages_to_backbone]]
    head_stages = [s for s in model.stages[stages_to_backbone:]]

    # If no stages in backbone, move stem_act into head
    if len(backbone_stages) == 0:
        backbone = nn.Sequential(model.stem_conv, model.stem_bn)
        head = nn.Sequential(
            model.stem_act,
            *head_stages,
            model.head_pool,
            nn.Flatten(1),
            model.fc
        )
        return backbone, head

    # Replace the final block in the last backbone stage with PreActBlock
    last_stage = backbone_stages[-1]
    last_block = last_stage[-1]
    last_stage_blocks = list(last_stage)
    last_stage_blocks[-1] = PreActBlock(last_block)
    backbone_stages[-1] = nn.Sequential(*last_stage_blocks)

    # Post-activation (same class as the original block.act) goes to the head
    post_act = type(last_block.act)()

    head = nn.Sequential(post_act, *head_stages, model.head_pool, nn.Flatten(1), model.fc)
    backbone = nn.Sequential(model.stem_conv, model.stem_bn, model.stem_act, *backbone_stages)
    return backbone, head


def get_max_split_pos_for_flexible_resnet(model):
    """
    Returns the maximum valid split_pos for a FlexibleResNet (len(stages) + 1).
    """
    if not isinstance(model, FlexibleResNet):
        raise ValueError("Model must be an instance of FlexibleResNet")
    return len(model.stages) + 1

def compare_model_and_split(model: FlexibleResNet, split_pos: int, atol=1e-6, rtol=1e-5):
    model.eval()
    backbone, head = split_flexible_resnet_for_ash(model, split_pos)

    # Put split parts in eval mode
    backbone.eval()
    head.eval()
    print(backbone)

    # deterministic input
    torch.manual_seed(0)
    x = torch.randn(2, model.stem_conv.in_channels, 32, 32)

    with torch.no_grad():
        orig_out = model(x)
        mid = backbone(x)
        # If backbone returns a GeometricTensor or non-tensor, this test expects plain tensors (FlexibleResNet does)
        split_out = head(mid)

    if not torch.allclose(orig_out, split_out, atol=atol, rtol=rtol):
        diff = (orig_out - split_out).abs()
        max_diff = diff.max().item()
        return False, max_diff
    return True, 0.0

if __name__ == "__main__":
    # Build a small test network
    model = FlexibleResNet(channels=[16, 32, 64], blocks_per_stage=[2, 2, 2], num_classes=10, in_channels=1, activation=nn.GELU, stem_stride=1)

    num_stages = len(model.stages)
    max_split = get_max_split_pos_for_flexible_resnet(model)

    any_fail = False
    for split_pos in range(0, max_split + 1):
        ok, info = compare_model_and_split(model, split_pos)
        if ok:
            print(f"split_pos={split_pos}: OK")
        else:
            print(f"split_pos={split_pos}: MISMATCH (max abs diff = {info:.6e})")
            any_fail = True

    if any_fail:
        print("One or more splits produced a different result than the original model.")
    else:
        print("All splits matched the original model output within tolerance.")


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Iterable, Union

class DownsampleICNN(nn.Module):
    """
    ICNN with optional spatial downsampling after specified layers.
    Convex layer form: z_{k+1} = act(W_z^k z_k + W_x^k x_k)
    - W_z^k constrained to be non-negative via zero_clip()
    - Final layer is a linear layer applied after global average pooling
    
    Args:
        n_in_channels: input channels.
        n_filters: int (constant width) or list of channel sizes (len = n_layers+1).
        n_layers: used only if n_filters is int.
        downsample_layers: iterable of layer indices (0-based) after whose update pooling is applied.
        num_classes: if provided, adds a final linear classifier after pooling.
        initial_stride: stride for wx0 (stem) convolution, typically 2 to match ResNet.
    """
    def __init__(self,
                 n_in_channels: int = 1,
                 n_filters: Union[int, Sequence[int]] = 32,
                 kernel_size: int = 3,
                 negative_slope: float = 0.2,
                 init_min: float = 0.0,
                 init_max: float = 1e-3,
                 n_layers: int = 5,
                 downsample_layers: Iterable[int] = (),
                 num_classes: int = None,
                 initial_stride: int = 1):
        super().__init__()
        if isinstance(n_filters, int):
            self.n_layers = n_layers
            n_filters = [n_filters] * (self.n_layers + 1)
        else:
            n_filters = list(n_filters)
            assert len(n_filters) >= 2, "n_filters list must have length >= 2"
            self.n_layers = len(n_filters) - 1

        self.channels = n_filters
        self.neg_slope = negative_slope
        self.downsample_layers = set(downsample_layers)
        self.num_classes = num_classes
        self.initial_stride = initial_stride
        pad = kernel_size // 2

        # First layer (unconstrained) with optional stride for initial downsampling
        self.wx0 = nn.Conv2d(n_in_channels, n_filters[0], kernel_size, 
                            stride=initial_stride, padding=pad, bias=True)

        # Hidden transitions - use stride=2 for layers that need downsampling
        self.wz = nn.ModuleList()
        self.wx_skip = nn.ModuleList()
        
        for i in range(self.n_layers):
            # Check if this layer should downsample
            stride = 2 if i in downsample_layers else 1
            
            self.wz.append(
                nn.Conv2d(n_filters[i], n_filters[i + 1], kernel_size, 
                         stride=stride, padding=pad, bias=False)
            )
            self.wx_skip.append(
                nn.Conv2d(n_in_channels, n_filters[i + 1], kernel_size, 
                         stride=stride, padding=pad, bias=True)
            )

        # Global average pooling before final layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final linear layer (must have non-negative weights for convexity)
        final_out_features = num_classes if num_classes is not None else 1
        self.final = nn.Linear(n_filters[-1], final_out_features, bias=True)

        self._init_pos(init_min, init_max)

    def _init_pos(self, a: float, b: float):
        with torch.no_grad():
            for w in self.wz:
                w.weight.copy_(a + (b - a) * torch.rand_like(w.weight))
            # Initialize final layer positively for convexity
            self.final.weight.data.clamp_(0)
            self.final.weight.copy_(a + (b - a) * torch.rand_like(self.final.weight))

    def zero_clip(self):
        """Clip weights to maintain non-negativity for convexity"""
        for w in self.wz:
            w.weight.data.clamp_(0)
        self.final.weight.data.clamp_(0)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_curr = x
        
        # wx0 applies initial downsampling via stride if initial_stride > 1
        z = F.leaky_relu(self.wx0(x_curr), negative_slope=self.neg_slope)
        
        # If wx0 downsampled, we need to downsample x_curr to match z's spatial size
        if self.initial_stride > 1:
            # Use adaptive pooling to match z's spatial dimensions exactly
            target_size = z.shape[2:]  # (H, W) of z
            x_curr = F.adaptive_avg_pool2d(x_curr, output_size=target_size)
        
        for i in range(self.n_layers):
            # wz[i] may downsample z via stride
            z_part = self.wz[i](z)
            
            # wx_skip[i] has the same stride as wz[i], so it will downsample x_curr automatically
            # We do NOT need to manually downsample x_curr before wx_skip
            x_part = self.wx_skip[i](x_curr)
            
            # After computing x_part, if this layer downsamples, update x_curr for next iteration
            # Use adaptive pooling to match the exact spatial size of z_part
            if i in self.downsample_layers:
                target_size = z_part.shape[2:]  # (H, W) of z_part after downsampling
                x_curr = F.adaptive_avg_pool2d(x_curr, output_size=target_size)
            
            # Diagnostic: if shapes don't match, print to help debugging, then re-raise
            try:
                z = F.leaky_relu(z_part + x_part, negative_slope=self.neg_slope)
            except RuntimeError as e:
                # Print shapes and layer info to help track down mismatch
                print(f"DownsampleICNN shape mismatch at layer {i}:")
                print(f"  z_part.shape = {tuple(z_part.shape)}")
                print(f"  x_part.shape = {tuple(x_part.shape)}")
                print(f"  x_curr.shape = {tuple(x_curr.shape)}")
                print(f"  z.shape (before wz) = {tuple(z.shape)}")
                print(f"  downsample_layers = {sorted(list(self.downsample_layers))}")
                raise

        # Global average pooling
        z = self.global_pool(z)
        z = z.view(z.size(0), -1)
        
        # Final linear layer
        z = self.final(z)
        return z


def convert_flexible_resnet_to_downsample_icnn(
    model: FlexibleResNet,
    negative_slope: float = 0.2,
    init_min: float = 0.0,
    init_max: float = 1e-3
) -> DownsampleICNN:
    """
    Converts a FlexibleResNet model to a DownsampleICNN with matching architecture.
    
    The conversion maps:
    - ResNet stem (conv with stride) -> ICNN wx0 with matching stride
    - Each ResNet BasicBlock (2 convs) -> 2 ICNN layers (wz + wx_skip each)
    - ResNet stage downsampling (stride=2 in first conv) -> ICNN wz layer with stride=2
    - ResNet final classifier -> ICNN linear layer after pooling
    
    The ICNN maintains convexity through:
    - Non-negative weights in wz layers and final linear layer
    - Skip connections from input through wx_skip layers
    
    Args:
        model: FlexibleResNet instance to convert
        negative_slope: Negative slope for LeakyReLU in ICNN
        init_min: Minimum value for positive weight initialization
        init_max: Maximum value for positive weight initialization
    
    Returns:
        DownsampleICNN instance with matching architecture
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")
    
    # Extract structure from ResNet
    in_channels = model.stem_conv.in_channels
    num_classes = model.fc.out_features
    stem_stride = model.stem_conv.stride[0]
    
    # Build filter sizes and track where downsampling occurs
    stage_info = []
    for stage_idx, stage in enumerate(model.stages):
        num_blocks = len(stage)
        first_block = stage[0]
        
        # Get output channels
        if hasattr(first_block, 'conv3'):
            out_channels = first_block.conv3.out_channels
        else:
            out_channels = first_block.conv2.out_channels
        
        # Check if this stage has downsampling (stride=2 in first block's first conv)
        has_stride = (first_block.conv1.stride[0] == 2)
        
        stage_info.append((num_blocks, out_channels, has_stride))
    
    # Create filter list for ICNN
    # Each BasicBlock has 2 convs: conv1 (may have stride=2), conv2 (stride=1)
    # We map: conv1 -> wz layer (may have stride=2), conv2 -> wz layer (stride=1)
    stem_channels = stage_info[0][1]  # First stage output channels
    n_filters = [stem_channels]  # wx0 outputs stem_channels
    
    downsample_layers = []
    layer_idx = 0  # Current ICNN layer index (0-based, after wx0)
    
    for stage_idx, (num_blocks, out_channels, has_stride) in enumerate(stage_info):
        for block_idx in range(num_blocks):
            # First conv of block (may have stride=2 for first block of stages 1+)
            if block_idx == 0 and has_stride:
                # This wz layer should have stride=2
                downsample_layers.append(layer_idx)
                n_filters.append(out_channels)
                layer_idx += 1
            else:
                # Regular layer without stride
                n_filters.append(out_channels)
                layer_idx += 1
            
            # Second conv of block (always stride=1)
            n_filters.append(out_channels)
            layer_idx += 1
    
    # Create ICNN
    icnn = DownsampleICNN(
        n_in_channels=in_channels,
        n_filters=n_filters,
        kernel_size=3,
        negative_slope=negative_slope,
        init_min=init_min,
        init_max=init_max,
        downsample_layers=downsample_layers,
        num_classes=num_classes,
        initial_stride=stem_stride
    )
    
    return icnn

def convert_flexible_resnet_to_gcnn(
    model: FlexibleResNet,
    num_rotations: int = 8,
    reflection: bool = False,
    activation=torch.nn.GELU,
    enable_auto_padding: bool = False
) -> 'GroupResNet':
    """
    Converts a FlexibleResNet model to a GroupResNet with matching architecture.

    Args:
        model: FlexibleResNet instance to convert
        num_rotations: Number of rotations in the group
        reflection: If True, use dihedral group (rotations + reflections)
        activation: Activation function class

    Returns:
        GroupResNet instance with scaled channels to match parameter count
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    # Extract structure from ResNet
    in_channels = model.stem_conv.in_channels
    num_classes = model.fc.out_features
    stem_stride = model.stem_conv.stride[0]

    # Get channel sizes from stages
    stage_channels = []
    for stage in model.stages:
        first_block = stage[0]
        if hasattr(first_block, 'conv3'):
            out_ch = first_block.conv3.out_channels
        else:
            out_ch = first_block.conv2.out_channels
        stage_channels.append(out_ch)

    # Get blocks per stage
    blocks_per_stage = [len(stage) for stage in model.stages]

    # Scale channels to maintain similar parameter count
    # Group convolutions multiply parameters by group size
    # For dihedral groups, the group size is 2 * num_rotations
    group_size = 2 * num_rotations if reflection else num_rotations
    scale_factor = math.sqrt(group_size)
    scaled_channels = [max(8, int(ch / scale_factor)) for ch in stage_channels]

    # Import here to avoid circular dependency
    from rot_resnet import GroupResNet

    return GroupResNet(
        channels=scaled_channels,
        blocks_per_stage=blocks_per_stage,
        num_classes=num_classes,
        in_channels=in_channels,
        activation=activation,
        num_rotations=num_rotations,
        use_reflection=reflection,
        stem_stride=stem_stride,
        pad_blocks = enable_auto_padding
    )


def convert_flexible_resnet_to_escnn(
        model: FlexibleResNet,
        rotations: int = 8,
        reflection: bool = False,
        continuous: bool = False,
        max_frequency: int = None,
        act_cls=escnn_nn.ReLU,
        enable_auto_padding: bool = False,
        pad_input: bool = False
) -> ESCNNFlexibleResNet:
    """
    Converts a FlexibleResNet model to an ESCNNFlexibleResNet with matching architecture.

    CRITICAL INSIGHT: ESCNN uses steerable convolutions where the number of parameters
    is determined by the kernel basis size, NOT by naive channel multiplication.

    For regular representations (discrete groups):
    - Kernel basis size is approximately constant per field pair
    - Total params ≈ fields_in * fields_out * basis_size
    - Basis_size is roughly independent of group size!
    - So we need to INCREASE fields to compensate for reduced total channels

    For continuous groups:
    - Constraints are even stronger, fewer basis elements
    - Need even MORE fields to match parameter count

    Args:
        model: FlexibleResNet instance to convert
        rotations: Number of rotations (for discrete groups)
        reflection: If True, use dihedral/O(2) group instead of cyclic/SO(2)
        continuous: If True, use continuous groups (SO(2) or O(2))
        max_frequency: Maximum frequency for continuous groups (defaults to rotations//2)
        act_cls: ESCNN activation function class

    Returns:
        ESCNNFlexibleResNet instance with scaled channels to match parameter count
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    # Extract structure from ResNet
    in_channels = model.stem_conv.in_channels
    num_classes = model.fc.out_features
    stem_stride = model.stem_conv.stride[0]

    # Get channel sizes from stages
    stage_channels = []
    for stage in model.stages:
        first_block = stage[0]
        if hasattr(first_block, 'conv3'):
            out_ch = first_block.conv3.out_channels
        else:
            out_ch = first_block.conv2.out_channels
        stage_channels.append(out_ch)

    # Get blocks per stage
    blocks_per_stage = [len(stage) for stage in model.stages]

    # Calculate scaling based on kernel basis size
    # For a 3x3 kernel, standard conv has 9 * C_in * C_out parameters
    # For ESCNN with regular repr, basis size is roughly constant (around 9-20 depending on group)
    # So: standard params ≈ 9 * C_in * C_out
    #     ESCNN params ≈ basis_size * fields_in * fields_out
    # To match: fields ≈ C * sqrt(9 / basis_size)

    if continuous:
        if max_frequency is None:
            max_frequency = rotations // 2


        scale_factor = 11/9
        if reflection:
            scale_factor *= math.sqrt(2)
    else:
        # Discrete groups: regular representation
        group_size = 2 * rotations if reflection else rotations

        scale_factor = 1.222222222/math.sqrt(group_size)

    scaled_channels = [max(1, int(ch * scale_factor)) for ch in stage_channels]

    return ESCNNFlexibleResNet(
        fields_per_stage=scaled_channels,
        blocks_per_stage=blocks_per_stage,
        num_classes=num_classes,
        in_channels=in_channels,
        act_cls=act_cls,
        rotations=rotations,
        reflection=reflection,
        continuous=continuous,
        max_frequency=max_frequency,
        stem_stride=stem_stride,
        pad_blocks=enable_auto_padding,
        pad_input=pad_input
    )

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tet_conversion_parameter_counts():
    """Test that converted models have approximately the same parameter count."""
    print("\n" + "="*80)
    print("Testing parameter counts for model conversions")
    print("="*80)

    # Create base FlexibleResNet
    base_channels = [64, 128]
    blocks_per_stage = [2, 2]
    base_model = FlexibleResNet(
        channels=base_channels,
        blocks_per_stage=blocks_per_stage,
        num_classes=10,
        in_channels=1,
        activation=nn.GELU,
        stem_stride=1
    )

    base_params = count_parameters(base_model)
    print(f"\nBase FlexibleResNet parameters: {base_params:,}")

    # Test GroupResNet conversions
    print("\n" + "-"*80)
    print("GroupResNet conversions:")
    print("-"*80)

    for num_rotations in [4, 8]:
        for reflection in [False, True]:
            gcnn = convert_flexible_resnet_to_gcnn(
                base_model,
                num_rotations=num_rotations,
                reflection=reflection
            )
            gcnn_params = count_parameters(gcnn)
            ratio = gcnn_params / base_params
            group_type = "Dihedral" if reflection else "Cyclic"
            print(f"{group_type} N={num_rotations}: {gcnn_params:,} params (ratio: {ratio:.3f})")

    # Test ESCNN conversions
    print("\n" + "-"*80)
    print("ESCNN conversions (discrete):")
    print("-"*80)

    for rotations in [4, 8]:
        for reflection in [False, True]:
            escnn_model = convert_flexible_resnet_to_escnn(
                base_model,
                rotations=rotations,
                reflection=reflection,
                continuous=False
            )
            escnn_params = count_parameters(escnn_model)
            ratio = escnn_params / base_params
            group_type = "D" if reflection else "C"
            print(f"{group_type}{rotations}: {escnn_params:,} params (ratio: {ratio:.3f})")

    # Test ESCNN continuous conversions
    print("\n" + "-"*80)
    print("ESCNN conversions (continuous):")
    print("-"*80)

    for max_freq in [3, 5, 8]:
        for reflection in [False, True]:
            escnn_model = convert_flexible_resnet_to_escnn(
                base_model,
                rotations=8,  # Not used for continuous, but required
                reflection=reflection,
                continuous=True,
                max_frequency=max_freq
            )
            escnn_params = count_parameters(escnn_model)
            ratio = escnn_params / base_params
            group_type = "O(2)" if reflection else "SO(2)"
            print(f"{group_type} max_freq={max_freq}: {escnn_params:,} params (ratio: {ratio:.3f})")


if __name__ == "__main__":
    # Existing split tests
    model2 = FlexibleResNet(
        channels=[32, 64, 128,256],
        blocks_per_stage=[2, 2, 2,2],
        num_classes=10,
        in_channels=1,
        activation=nn.GELU,
        stem_stride=1
    )



    model_escnn = convert_flexible_resnet_to_escnn(model2).to("cuda")
    model_gcnn = convert_flexible_resnet_to_gcnn(model2).to("cuda")

    # speed test main model
    from experiment_thesis.dataset_preperation.basic_networks import convert_flexible_resnet_to_escnn, \
        convert_flexible_resnet_to_gcnn


    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x =torch.randn(64,1,32,32).to("cuda")
    y = torch.randint(0,10,(64,)).to("cuda")
    model2.eval().to(device)
    start_time = time.time()
    for _ in range(1000):
        outputs = model2(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 1000:.6f} seconds")

    x = x.to(device)
    y = y.to(device)
    model2.eval().to(device)
    start_time = time.time()
    for _ in range(1000):
        outputs = model2(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 1000:.6f} seconds")
    # %%
    import time

    x = x.to(device)
    y = y.to(device)
    model.eval().to(device)
    start_time = time.time()
    for _ in range(100):
        outputs = model_escnn(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 100:.6f} seconds")

    x = x.to(device)
    y = y.to(device)
    model.eval().to(device)
    start_time = time.time()
    for _ in range(100):
        outputs = model_escnn(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 100:.6f} seconds")
    # %%
    import time

    x = x.to(device)
    y = y.to(device)
    model.eval().to(device)
    start_time = time.time()
    for _ in range(100):
        outputs = model_gcnn(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 100:.6f} seconds")

    import time

    x = x.to(device)
    y = y.to(device)
    model.eval().to(device)
    start_time = time.time()
    for _ in range(100):
        outputs = model_gcnn(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 100:.6f} seconds")






