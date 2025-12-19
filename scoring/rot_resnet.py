import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from equiadapt import RotoReflectionEquivariantConvLift, RotationEquivariantConvLift, RotoReflectionEquivariantConv, \
    RotationEquivariantConv


# Assumes the following classes are already defined somewhere imported in runtime:
# RotationEquivariantConvLift, RotoReflectionEquivariantConvLift,
# RotationEquivariantConv, RotoReflectionEquivariantConv
# (Not repeated here per instruction.)

# corrected _bn helper (same as yours but guidance: use num_channels=C when creating)
def _bn(num_channels: int, affine: bool = True):
    return nn.BatchNorm2d(num_channels, affine=affine)


class LiftedStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_rotations: int,
                 use_reflection: bool, activation=nn.ReLU, kernel_size=3, padding=1, stride=1):
        super().__init__()
        Lift = RotoReflectionEquivariantConvLift if use_reflection else RotationEquivariantConvLift
        self.lift = Lift(in_ch, out_ch, kernel_size=kernel_size,
                         num_rotations=num_rotations, stride=stride, padding=padding, bias=False)
        self.act = activation()
        self.num_rotations = num_rotations
        self.use_reflection = use_reflection
        self.G = 2 * num_rotations if use_reflection else num_rotations
        # BN over C (not C*G)
        self.bn = _bn(out_ch)

    def forward(self, x):
        x = self.lift(x)  # (B, C, G, H, W)
        B, C, G, H, W = x.shape
        # share BN across group positions by moving G into batch
        x = x.permute(0, 2, 1, 3, 4).reshape(B * G, C, H, W)  # (B*G, C, H, W)
        x = self.act(self.bn(x))
        x = x.reshape(B, G, C, H, W).permute(0, 2, 1, 3, 4)   # back to (B, C, G, H, W)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupBasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_rotations: int,
                 use_reflection: bool, stride: int = 1, activation=nn.ReLU,
                 kernel_size=3, pad_blocks: bool = False):
        super().__init__()
        GroupConv = RotoReflectionEquivariantConv if use_reflection else RotationEquivariantConv
        padding = kernel_size // 2
        self.pad_blocks = pad_blocks
        self.use_reflection = use_reflection
        self.G = 2 * num_rotations if use_reflection else num_rotations
        self.act = activation()

        # equivariant group convs
        self.conv1 = GroupConv(in_ch, out_ch, kernel_size=kernel_size,
                               num_rotations=num_rotations, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = GroupConv(out_ch, out_ch, kernel_size=kernel_size,
                               num_rotations=num_rotations, stride=1,
                               padding=padding, bias=False)

        # BN over C (shared across G)
        self.bn1 = _bn(out_ch)
        self.bn2 = _bn(out_ch)

        # projection: MUST use GroupConv so group axis preserved
        if stride != 1 or in_ch != out_ch:
            self.proj_conv = GroupConv(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)
            self.proj_bn = _bn(out_ch)
            self.proj = True
        else:
            self.proj = False
        self.stem_stride = stride

        self.out_ch = out_ch

    @staticmethod
    def pad_if_even(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (1 - h % 2) % 2
        pad_w = (1 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def forward(self, x):
        if self.pad_blocks:
            x = self.pad_if_even(x)
        B, Cin, G, H, W = x.shape

        # Conv1
        y = self.conv1(x)   # (B, C1, G, H1, W1)
        B1, C1, G1, H1, W1 = y.shape
        y = y.permute(0, 2, 1, 3, 4).reshape(B1 * G1, C1, H1, W1)   # (B*G, C, H, W)
        y = self.act(self.bn1(y))
        y = y.reshape(B1, G1, C1, H1, W1).permute(0, 2, 1, 3, 4)   # (B, C1, G, H1, W1)

        # Conv2
        y = self.conv2(y)   # (B, C2, G, H2, W2)
        B2, C2, G2, H2, W2 = y.shape
        y = y.permute(0, 2, 1, 3, 4).reshape(B2 * G2, C2, H2, W2)
        y = self.bn2(y)
        y = y.reshape(B2, G2, C2, H2, W2).permute(0, 2, 1, 3, 4)

        # Shortcut / projection
        if self.proj:
            sc = self.proj_conv(x)   # (B, C_out, G, H', W')
            Bp, Cp, Gp, Hp, Wp = sc.shape
            sc = sc.permute(0, 2, 1, 3, 4).reshape(Bp * Gp, Cp, Hp, Wp)
            sc = self.proj_bn(sc)
            sc = sc.reshape(Bp, Gp, Cp, Hp, Wp).permute(0, 2, 1, 3, 4)
        else:
            sc = x

        out = self.act(y + sc)
        return out



class GroupResNet(nn.Module):
    def __init__(self, channels, blocks_per_stage, num_classes=10, in_channels=1,
                 num_rotations: int = 4, use_reflection: bool = False,
                 activation=nn.ReLU, stem_channels=None, group_pool: str = 'avg',
                 pad_input: bool = True, pad_blocks: bool = False,stem_stride: int = 1):
        super().__init__()
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * len(channels)
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have same length")
        self.num_rotations = num_rotations
        self.use_reflection = use_reflection
        self.G = 2 * num_rotations if use_reflection else num_rotations
        self.pad_input = pad_input
        self.pad_blocks = pad_blocks

        stem_out = stem_channels or channels[0]
        self.stem = LiftedStem(in_channels, stem_out, num_rotations,
                               use_reflection, activation=activation, stride=stem_stride)
        stages = []
        in_ch = stem_out
        for idx, (out_ch, n_blocks) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if idx == 0 else 2
            blocks = [GroupBasicBlock(in_ch, out_ch, num_rotations, use_reflection,
                                      stride=stride, activation=activation, pad_blocks=pad_blocks)]
            in_ch = out_ch
            for _ in range(n_blocks - 1):
                blocks.append(GroupBasicBlock(in_ch, out_ch, num_rotations, use_reflection,
                                              stride=1, activation=activation, pad_blocks=pad_blocks))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.group_pool = group_pool
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        final_channels = channels[-1] if len(channels) > 0 else stem_out
        if group_pool == 'none':
            self.fc = nn.Linear(final_channels * self.G, num_classes)
        else:
            self.fc = nn.Linear(final_channels, num_classes)

    @staticmethod
    def pad_to_odd_after_downsampling(x, num_downsamples):
        H, W = x.shape[-2:]
        factor = 2 ** num_downsamples
        H_pad = factor * ((H + factor - 1) // factor)
        if H_pad % 2 == 0:
            H_pad += 1
        pad_top = (H_pad - H) // 2
        pad_bottom = H_pad - H - pad_top
        W_pad = factor * ((W + factor - 1) // factor)
        if W_pad % 2 == 0:
            W_pad += 1
        pad_left = (W_pad - W) // 2
        pad_right = W_pad - W - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x):
        if self.pad_input:
            num_downsamples = len(self.stages) - 1
            if self.stem_stride > 1:
                num_downsamples += 1
            x = self.pad_to_odd_after_downsampling(x, num_downsamples)

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        B, C, G, H, W = x.shape
        if self.group_pool == 'avg':
            x = x.mean(dim=2)
        elif self.group_pool == 'max':
            x, _ = x.max(dim=2)
        elif self.group_pool == 'none':
            x = x.reshape(B, C * G, H, W)
        else:
            raise ValueError("group_pool must be one of ['avg','max','none']")
        x = self.head_pool(x).flatten(1)
        return self.fc(x)


def rotate_tensor(x, k):
    """Rotate spatial dims HxW by k * 90 degrees counter-clockwise."""
    k = k % 4  # Normalize to 0-3
    if k == 0:
        return x
    return torch.rot90(x, k, dims=(-2, -1))


def reflect_tensor(x):
    """Reflect across vertical axis (flip width)."""
    return torch.flip(x, dims=[-1])


def apply_group_action(x, g, num_rotations, use_reflection):
    """
    Apply group element g to feature tensor x: (B, C, G, H, W).

    For rotation group:
    - Cyclically shift the group dimension by g positions
    - Rotate spatial dimensions by g * (360/num_rotations) degrees

    For roto-reflection group:
    - g in [0, num_rotations-1]: pure rotations
    - g in [num_rotations, 2*num_rotations-1]: rotations + reflection
    """
    B, C, G, H, W = x.shape
    R = num_rotations

    if use_reflection:
        if g < R:
            # Pure rotation
            r = g
            is_reflected = False
        else:
            # Rotation + reflection
            r = g - R
            is_reflected = True
    else:
        r = g
        is_reflected = False

    # Step 1: Cyclically permute group dimension
    # When we rotate by r, channel at position i moves to position (i + r) % G
    x_transformed = torch.roll(x, shifts=r, dims=2)

    # Step 2: Rotate spatial dimensions
    # For 90-degree rotations: rot90(x, r) rotates by r * 90 degrees
    # But we need to handle arbitrary num_rotations
    if num_rotations == 4:
        # Direct 90-degree rotations
        x_transformed = rotate_tensor(x_transformed, r)
    else:
        # For other rotation counts, we'd need interpolation
        # For now, only support num_rotations that divide 360 evenly with 90° increments
        angle_deg = r * (360 / R)
        if angle_deg % 90 == 0:
            k = int(angle_deg / 90)
            x_transformed = rotate_tensor(x_transformed, k)
        else:
            raise NotImplementedError(f"num_rotations={R} requires interpolation for equivariance testing")

    # Step 3: Apply reflection if needed
    if is_reflected:
        x_transformed = reflect_tensor(x_transformed)
        # When reflecting, we also need to reverse the group dimension ordering
        # because reflection inverts the orientation
        x_transformed = torch.flip(x_transformed, dims=[2])

    return x_transformed

import numpy as np


@torch.no_grad()
def tst_equivariance(block, num_rotations, use_reflection,
                      B=2, C=3, H=31, W=31, tol=1e-4, verbose=True):
    """
    Test equivariance of a GroupBasicBlock.
    """

    G = 2 * num_rotations if use_reflection else num_rotations

    # random input
    x = torch.randn(B, C, G, H, W).cuda()

    # send block to eval to avoid BN randomness
    block.eval()

    # baseline output: f(x)
    fx = block(x)

    errors = []

    for g in range(G):
        # Apply group action to input
        Tx = apply_group_action(x, g, num_rotations, use_reflection)

        # f(T_g x)
        f_Tx = block(Tx)

        # apply action to output
        T_fx = apply_group_action(fx, g, num_rotations, use_reflection)

        # compute error
        err = (f_Tx - T_fx).abs().max().item()
        errors.append(err)

        if verbose:
            print(f"g={g:2d} | equivariance error = {err:.6f}")

    print("\nSummary:")
    print(f"max error = {max(errors):.6f}")
    print(f"mean error = {np.mean(errors):.6f}")
    print(f"All errors < tol?  {all(e < tol for e in errors)}")

    return errors

import torch
import numpy as np

def apply_group_action_image(x_img, g, num_rotations, use_reflection):
    """Apply group action to spatial image tensor x_img of shape (B, C, H, W)."""
    # same logic as apply_group_action but only for the spatial dims (no group axis)
    R = num_rotations
    if use_reflection:
        if g < R:
            r = g
            is_reflected = False
        else:
            r = g - R
            is_reflected = True
    else:
        r = g
        is_reflected = False

    # spatial rotation
    if num_rotations == 4:
        x = rotate_tensor(x_img, r)
    else:
        angle_deg = r * (360 / R)
        if angle_deg % 90 == 0:
            k = int(angle_deg / 90)
            x = rotate_tensor(x_img, k)
        else:
            raise NotImplementedError("num_rotations requires interpolation for image rotations")

    # reflection
    if is_reflected:
        x = reflect_tensor(x)
    return x

@torch.no_grad()
def tst_equivariance_resnet(net: GroupResNet, num_rotations, use_reflection,
                            B=2, in_ch=1, H=31, W=31, tol=1e-4, verbose=True):
    """
    Test equivariance of whole GroupResNet by comparing feature-map equivariance
    (before group-pooling / head pooling). The network should be created with
    group_pool='none' so the group axis is preserved until we extract features.
    """
    device = next(net.parameters()).device

    G = 2 * num_rotations if use_reflection else num_rotations

    # random image input (B, C_in, H, W)
    x_img = torch.randn(B, in_ch, H, W, device=device)

    net.eval()

    # Helper: run the network up to just before the head pooling / fc to get (B,C,G,H,W)
    def forward_to_group_features(x_image):
        # if net.pad_input is True, apply same padding used in forward
        if net.pad_input:
            num_downsamples = len(net.stages) - 1
            x_image = net.pad_to_odd_after_downsampling(x_image, num_downsamples)
        x = net.stem(x_image)   # (B, C, G, H, W)
        for stage in net.stages:
            x = stage(x)
        return x  # (B, C, G, H, W)

    # baseline features
    f_x = forward_to_group_features(x_img)  # (B, C, G, H, W)

    errors = []

    for g in range(G):
        # 1) apply group action to raw image then forward
        Tx_img = apply_group_action_image(x_img, g, num_rotations, use_reflection)
        f_Tx = forward_to_group_features(Tx_img)

        # 2) apply group action to features f_x
        T_fx = apply_group_action(f_x, g, num_rotations, use_reflection)

        # compute max abs error
        err = (f_Tx - T_fx).abs().max().item()
        errors.append(err)
        if verbose:
            print(f"g={g:2d} | equivariance error = {err:.6f}")

    print("\nSummary:")
    print(f"max error = {max(errors):.6f}")
    print(f"mean error = {np.mean(errors):.6f}")
    print(f"All errors < tol?  {all(e < tol for e in errors)}")

    return errors

import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import nn as escnn_nn
from escnn import gspaces

class GBasicBlock(nn.Module):
    def __init__(self, in_type: escnn_nn.FieldType, out_fields: int, stride: int = 1,
                 act_cls=escnn_nn.ReLU, pad_blocks: bool = False):
        super().__init__()

        gspace = in_type.gspace

        # Try to use the regular representation (works only for finite groups).
        # If that raises ValueError (e.g. SO(2) / rot2dOnR2), fall back to a repeated irrep.
        try:
            reg = gspace.regular_repr
            self.out_type = escnn_nn.FieldType(gspace, out_fields * [reg])
            is_continuous = False
        except ValueError:
            # Continuous group: build out_type by repeating a non-trivial irrep (freq=1).
            # Try the common signatures for irrep(...) (some gspaces accept (freq) or (freq, parity)).
            try:
                base_irrep = gspace.irrep(1)
            except TypeError:
                base_irrep = gspace.irrep(1, 1)
            self.out_type = escnn_nn.FieldType(gspace, out_fields * [base_irrep])
            is_continuous = True

        self.pad_blocks = pad_blocks

        # Convs
        self.conv1 = escnn_nn.R2Conv(in_type, self.out_type, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = escnn_nn.R2Conv(self.out_type, self.out_type, kernel_size=3, padding=1, bias=False)

        # BatchNorm + Activation: choose modules that are valid for the representations
        if is_continuous:
            # Continuous irreps (SO(2)/O(2)): use NormBatchNorm and NormNonLinearity
            self.bn1 = escnn_nn.NormBatchNorm(self.out_type)
            self.bn2 = escnn_nn.NormBatchNorm(self.out_type)
            self.act = escnn_nn.NormNonLinearity(self.out_type)
        else:
            # Finite groups / regular repr: pointwise nonlinearities are allowed
            self.bn1 = escnn_nn.InnerBatchNorm(self.out_type)
            self.bn2 = escnn_nn.InnerBatchNorm(self.out_type)
            self.act = act_cls(self.out_type, inplace=True)

        # Projection for skip connection if needed
        self.proj = None
        if stride != 1 or in_type != self.out_type:
            self.proj = escnn_nn.R2Conv(in_type, self.out_type, kernel_size=1, stride=stride, bias=False)


    @staticmethod
    def pad_if_even(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (1 - h % 2) % 2
        pad_w = (1 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def forward(self, x: escnn_nn.GeometricTensor) -> escnn_nn.GeometricTensor:
        if self.pad_blocks:
            x = escnn_nn.GeometricTensor(self.pad_if_even(x.tensor), x.type)
        y = self.act(self.bn1(self.conv1(x)))
        if self.pad_blocks:
            y = escnn_nn.GeometricTensor(self.pad_if_even(y.tensor), y.type)
        y = self.bn2(self.conv2(y))
        skip = x if self.proj is None else self.proj(x)
        return self.act(y + skip)


class ESCNNFlexibleResNet(nn.Module):
    def __init__(self, fields_per_stage, blocks_per_stage, num_classes=10, in_channels=1,
                 act_cls=escnn_nn.ReLU, rotations=8, reflection=False,
                 continuous=False, max_frequency=None, stem_stride=1, pad_blocks=False,
                 pad_input=False):
        super().__init__()

        # gspace
        if continuous:
            if max_frequency is None:
                max_frequency = rotations // 2
            gspace_max_freq = max_frequency * 3
            if reflection:
                self.gspace = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=gspace_max_freq)
            else:
                self.gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=gspace_max_freq)
        else:
            if reflection:
                self.gspace = gspaces.flipRot2dOnR2(N=rotations)
            else:
                self.gspace = gspaces.rot2dOnR2(N=rotations)

        self.continuous = continuous
        self.reflection = reflection
        self.max_frequency = max_frequency
        self.rotations = rotations
        self.pad_blocks = pad_blocks
        self.pad_input = pad_input

        # field types
        if continuous:
            if reflection:
                if max_frequency is None:
                    max_frequency = rotations // 2
                irreps = [self.gspace.irrep(0, 0), self.gspace.irrep(1, 0)]
                for k in range(1, max_frequency + 1):
                    irreps.append(self.gspace.irrep(1, k))
            else:
                if max_frequency is None:
                    max_frequency = rotations // 2
                irreps = [self.gspace.irrep(k) for k in range(max_frequency + 1)]
            fields_needed = fields_per_stage[0]
            field_irreps = [irreps[i % len(irreps)] for i in range(fields_needed)]
            first_type = escnn_nn.FieldType(self.gspace, field_irreps)
        else:
            first_type = escnn_nn.FieldType(self.gspace, fields_per_stage[0] * [self.gspace.regular_repr])
        in_type = escnn_nn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        # stem conv
        self.stem_conv = escnn_nn.R2Conv(in_type, first_type, kernel_size=3,
                                         stride=stem_stride, padding=1, bias=False)
        self.stem_stride = stem_stride

        # detect trivial irreps in first_type
        has_trivial = any(r.is_trivial() for r in first_type.representations)

        if continuous:
            if not has_trivial:
                self.stem_bn = escnn_nn.NormBatchNorm(first_type)
            else:
                self.stem_bn = nn.Identity()
            self.stem_act = escnn_nn.NormNonLinearity(first_type)
        else:
            self.stem_bn = escnn_nn.InnerBatchNorm(first_type)
            self.stem_act = act_cls(first_type, inplace=True)

        # remaining architecture
        stage_modules = []
        current_type = first_type
        for stage_idx, (out_f, n_blocks) in enumerate(zip(fields_per_stage, blocks_per_stage)):
            blocks = []
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                block = GBasicBlock(current_type, out_f, stride=stride, act_cls=act_cls, pad_blocks=pad_blocks)
                current_type = block.out_type
                blocks.append(block)
            stage_modules.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stage_modules)

        try:
            self.group_pool = escnn_nn.GroupPooling(current_type)
        except AssertionError:
            self.group_pool = escnn_nn.NormPool(current_type)

        self.fc = nn.Linear(fields_per_stage[-1], num_classes)

    @staticmethod
    def pad_to_odd_after_downsampling(x, num_downsamples):
        H, W = x.shape[-2:]
        factor = 2 ** num_downsamples
        H_pad = factor * ((H + factor - 1) // factor)
        if H_pad % 2 == 0:
            H_pad += 1
        pad_top = (H_pad - H) // 2
        pad_bottom = H_pad - H - pad_top
        W_pad = factor * ((W + factor - 1) // factor)
        if W_pad % 2 == 0:
            W_pad += 1
        pad_left = (W_pad - W) // 2
        pad_right = W_pad - W - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    def forward(self, x):
        if self.pad_input:
            num_downsamples = len(self.stages) - 1
            if self.stem_stride > 1:
                num_downsamples += 1
            x = self.pad_to_odd_after_downsampling(x, num_downsamples)
        x = escnn_nn.GeometricTensor(x, self.stem_conv.in_type)
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            x = stage(x)
        spatial_mean = x.tensor.mean(dim=(2, 3), keepdim=True)
        x = escnn_nn.GeometricTensor(spatial_mean, x.type)
        x = self.group_pool(x)
        x = x.tensor.view(x.tensor.size(0), -1)
        return self.fc(x)


import torch


import torch

@torch.no_grad()
def tst_ESCNN_invariance(model, N=4, device="cuda"):
    """
    Test invariance of the full ESCNN network (logits should stay the same
    under rotations by 0, 90, ..., (N-1)*360/N degrees).
    """
    model.eval()
    x = torch.randn(1, 1, 63, 63, device=device)

    # baseline output
    y0 = model(x)

    print("\n===== ESCNN FULL NETWORK INVARIANCE TEST =====")
    for k in range(N):
        xr = torch.rot90(x, k, dims=(-2, -1))
        yr = model(xr)

        # compare logits
        err = (yr - y0).abs().max().item()
        print(f"Rotation {k*90:3d}° | max difference = {err:.6f}")

    print("================================================\n")

# Usage example
# model = YourESCNNNetwork().to("cuda")
# test_ESCNN_invariance(model, N=4)



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time

    # Example only – use your own ESCNN model
    model = ESCNNFlexibleResNet(
        fields_per_stage=[8, 16],
        blocks_per_stage=[2, 2],
        num_classes=10,
        in_channels=1,
        rotations=4,
        reflection=False,
        continuous=False,
        stem_stride=1
    ).cuda().eval()

    tst_ESCNN_invariance(model, N=4)

    from equiadapt import RotoReflectionEquivariantConv,RotationEquivariantConv

    block = GroupBasicBlock(
        in_ch=3, out_ch=6,
        num_rotations=4,
        use_reflection=False,
        stride=2
    ).cuda()

    errs = tst_equivariance(
        block,
        num_rotations=4,
        use_reflection=False
    )

    # Standard Conv2d baseline
    class StandardConv(nn.Module):
        def __init__(self, in_ch, out_ch, k, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding)

        def forward(self, x):
            return self.conv(x)

    device = "cuda"

    # instantiate GroupResNet - keep group_pool='none' to preserve G axis
    net = GroupResNet(
        channels=[16],
        blocks_per_stage=[2],
        num_classes=10,
        in_channels=1,
        num_rotations=4,
        use_reflection=False,
        group_pool='none',   # preserve group axis for equivariance test
        pad_input=True,
        pad_blocks=False,
        stem_stride=1,
        stem_channels=16,
    ).to(device).eval()

    errs = tst_equivariance_resnet(
        net,
        num_rotations=4,
        use_reflection=False,
        B=2, in_ch=1, H=31, W=31, tol=1e-4, verbose=True
    )


    device = "cuda"
    B, Cin, Cout, H, W = 8, 64, 128, 64, 64
    num_rotations = 1

    x_equi = torch.randn(B, int(Cin/math.sqrt(1)), num_rotations, H, W, device=device)
    x = torch.randn(B, Cin*num_rotations, H, W, device=device)

    # Instantiate layers
    equiv = RotationEquivariantConv(int(Cin/math.sqrt(1)), int(Cout/math.sqrt(1)), 3, num_rotations=num_rotations, padding=1, device=device).eval().cuda()
    cnn = torch.nn.Conv2d(Cin  * num_rotations, Cout  * num_rotations, 3, padding=1).eval().cuda()

    # Warmup
    for _ in range(10):
        equiv(x_equi)
        cnn(x)



    torch.cuda.synchronize()
    t2 = time.time()
    for _ in range(100):
        cnn(x)
        torch.cuda.synchronize()
    t3 = time.time()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        equiv(x_equi)
        torch.cuda.synchronize()
    t1 = time.time()

    print(f"RotoReflectionConv: {(t1 - t0) / 50 * 1000:.2f} ms/iter")
    print(f"Standard CNN Conv  : {(t3 - t2) / 50 * 1000:.2f} ms/iter")
