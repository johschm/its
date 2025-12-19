import torch

def grid_resample_reflection(x,T):
    """
    Resample the input x using the transformation matrix T.
    :param x: Input image of shape 3xHxW or 1xHxW. or 4 x H x W
    :param T: Transformation matrix of shape b,3x3 or 3x3 or b,2x3 or 2x3
    :return: Resampled tensor.
    """
    # Reshape x using torch grid resample. First last row is dropped per convention of torch.
    #check if transformation matrix is batched
    if len(T.shape) == 2:
        #if not batched, add batch dimension
        T = T.unsqueeze(0)
    affine_T = T[:, :2, :]
    unbatch = False
    if x.dim() == 3:
        #if x is 3d, we need to add the batch dimension
        x = x.unsqueeze(0)
        unbatch = True
    grid = torch.nn.functional.affine_grid(affine_T, x.size(), align_corners=False)
    x_transformed = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return x_transformed[0] if unbatch else x_transformed

def grid_resample_border(x,T):
    """
    Resample the input x using the transformation matrix T.
    :param x: Input image of shape 3xHxW or 1xHxW. or 4 x H x W
    :param T: Transformation matrix of shape b,3x3 or 3x3 or b,2x3 or 2x3
    :return: Resampled tensor.
    """
    # Reshape x using torch grid resample. First last row is dropped per convention of torch.
    #check if transformation matrix is batched
    if len(T.shape) == 2:
        #if not batched, add batch dimension
        T = T.unsqueeze(0)
    affine_T = T[:, :2, :]
    unbatch = False
    if x.dim() == 3:
        #if x is 3d, we need to add the batch dimension
        x = x.unsqueeze(0)
        unbatch = True
    grid = torch.nn.functional.affine_grid(affine_T, x.size(), align_corners=False)
    x_transformed = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return x_transformed[0] if unbatch else x_transformed

def grid_resample(x,T):
    """
    Resample the input x using the transformation matrix T.
    :param x: Input image of shape 3xHxW or 1xHxW. or 4 x H x W
    :param T: Transformation matrix of shape b,3x3 or 3x3 or b,2x3 or 2x3
    :return: Resampled tensor.
    """
    # Reshape x using torch grid resample. First last row is dropped per convention of torch.
    #check if transformation matrix is batched
    if len(T.shape) == 2:
        #if not batched, add batch dimension
        T = T.unsqueeze(0)
    affine_T = T[:, :2, :]
    unbatch = False
    if x.dim() == 3:
        #if x is 3d, we need to add the batch dimension
        x = x.unsqueeze(0)
        unbatch = True
    grid = torch.nn.functional.affine_grid(affine_T, x.size(), align_corners=False)
    x_transformed = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return x_transformed[0] if unbatch else x_transformed


import torch
import torch.nn.functional as F
import kornia


def grid_resample_blur(x, T, baseline_sigma=0.644, eps=1e-6):
    """
    Resample the input x using the transformation matrix T and apply adaptive blur
    to ensure consistent blur levels across different affine transforms_old.
    The baseline_sigma sets the target blur applied to all inputs.
    Identity transforms_old receive extra blur to match baseline.
    More heuristc than based on theory as one cant simulate bilinear interpolation effect using Gaussian blur.

    Args:
        x: Input image of shape (C,H,W) or (B,C,H,W)
        T: Transformation matrix of shape (3,3), (2,3), (B,3,3) or (B,2,3)
        baseline_sigma: Target blur level (Gaussian sigma)
        eps: small value to avoid zero sigma (default=1e-6)

    Returns:
        Resampled (and adaptively blurred) tensor, same batch structure as input.
    """
    # --- Handle transformation matrix batching ---
    if T.dim() == 2:
        T = T.unsqueeze(0)
    affine_T = T[:, :2, :]

    # --- Handle input batching ---
    unbatch = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        unbatch = True

    B, C, H, W = x.shape

    # --- Apply affine transform ---
    grid = F.affine_grid(affine_T, x.size(), align_corners=False)
    warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # --- Compute subpixel offset relative to nearest integer pixel centers ---
    x_coords = (grid[..., 0] + 1) * W / 2 - 0.5
    y_coords = (grid[..., 1] + 1) * H / 2 - 0.5
    frac_x = (x_coords - x_coords.round()).abs()
    frac_y = (y_coords - y_coords.round()).abs()

    sigma_effective = torch.sqrt(frac_x * (1 - frac_x) + frac_y * (1 - frac_y))
    sigma_effective = sigma_effective.mean(dim=[1,2])  # per sample

    # --- Compute additional sigma needed to reach baseline ---
    add_sigma = (baseline_sigma - sigma_effective).clamp(min=eps)

    # --- Apply Gaussian blur per sample ---
    sigmas = torch.stack([add_sigma, add_sigma], dim=1)  # (B,2)
    blurred = kornia.filters.gaussian_blur2d(warped, kernel_size=(3,3), sigma=sigmas)

    return blurred[0] if unbatch else blurred



def transform_3d_point_cloud(x, T):
    # x shape: (B, N, 3) where B is batch, N is num points
    # T shape: (B, 4, 4)
    # Add homogeneous coordinate
    ones = torch.ones_like(x[..., :1])  # (..., 1)
    x_homog = torch.cat([x, ones], dim=-1)  # (..., 4)

    # Prepare transformation matrices for broadcasting
    T_transposed = T.transpose(-1, -2)  # (B, 4, 4)
    T_transposed = T_transposed.unsqueeze(-3)  # (B, 1, 4, 4)

    # Perform batched matrix multiplication
    x_new = (x_homog.unsqueeze(-2) @ T_transposed).squeeze(-2)  # (B, N, 4)

    # Convert back to 3D coordinates
    epsilon = 1e-6
    x_transformed = x_new[..., :3] / (x_new[..., 3:4] + epsilon)

    return x_transformed

def transform_strokes_affine(x, T):
    """
    Apply an affine transform to stroke sequences given as relative deltas.
    Steps:
      1. Convert (dx, dy) to absolute positions via cumulative sum.
      2. Add a zero start point and apply the affine matrix to all points.
      3. Compute new deltas by differencing adjacent transformed points.
      4. Reattach the original pen_state channel.
    :param x: Tensor of shape (B, T, 3) with [dx, dy, pen_state].
    :param T: Affine matrix of shape (2, 3) or (B, 2, 3).
    :return: Tensor of shape (B, T, 3) with transformed deltas and same pen_state.
    """
    if T.dim() == 2:
        T = T.unsqueeze(0)                  # (1, 2, 3)
    B, T_len, _ = x.shape

    # split relative deltas and pen flag
    deltas = x[..., :2]                     # (B, T, 2)
    pen     = x[..., 2:]                    # (B, T, 1)

    # 1) absolute positions
    abs_pos = deltas.cumsum(dim=1)          # (B, T, 2)

    # prepend starting origin
    zeros    = torch.zeros(B, 1, 2, device=x.device, dtype=x.dtype)
    abs_aug  = torch.cat([zeros, abs_pos], dim=1)   # (B, T+1, 2)

    # 2) apply affine on homogeneous coords
    ones     = torch.ones(B, T_len+1, 1, device=x.device, dtype=x.dtype)
    pos_h    = torch.cat([abs_aug, ones], dim=2)    # (B, T+1, 3)
    T_t      = T.transpose(-1, -2)                   # (B, 3, 3)
    abs_trans= pos_h.matmul(T_t)                    # (B, T+1, 3)
    abs_trans = abs_trans[..., :2]                # (B, T+1, 2)
    # 3) compute new relative deltas
    new_deltas = abs_trans[:, 1:] - abs_trans[:, :-1]  # (B, T, 2)

    # 4) recombine
    return torch.cat([new_deltas, pen], dim=2)        # (B, T, 3)






#functions to transform parameters
def transformation_matrix_from_tf_params(param,transformations):
    """
    Create a transformation matrix for the given parameters.
    :param transformations: List of transformations objects from affine_transforms.py
    :return: Transformation matrix.
    """
    sizes = None
    if isinstance(param, torch.Tensor):
        sizes = [transformation["param_size"] for transformation in transformations]



    transforms = [transformation["matrix"] for transformation in transformations]
    return transformation_matrix_from_param(param, sizes, transforms)


def transformation_matrix_from_param(param,param_sizes,transformations):
    """
    Create a transformation matrix for the given parameters.
    :param transformations: List of transformations functions to apply.
    :return: Transformation matrix.
    """
    # now apply the transformations in order to calculate the final transformation matrix
    # create the transformation matrix
    if isinstance(param, torch.Tensor):
        param = torch.split(param, param_sizes, dim=-1)
    T = None
    for i, transformation in enumerate(transformations):
        # get the transformation function
        T = transformation(T, param[i])
    return T

def transformation_matrix_from_single_param(param, transformation):
    """
    Create a transformation matrix for the given parameters.
    :param transformation: Single transformation function
    :return: Transformation matrix.
    """
    # now apply the transformations in order to calculate the final transformation matrix
    # create the transformation matrix
    return transformation(None, param)


