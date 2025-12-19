import numpy as np
import roma
import torch
import math

from utils.helper import identity
from utils.transforms_old.base import Transform
from utils.transforms_old.bounded_transform import BoundedTransform
from utils.transforms_old.periodic_transform import PeriodicTransform
from utils.transforms_old.norm_constrained_transform import NormConstrainedTransform
from utils.transforms_old.apply import grid_resample, transform_3d_point_cloud

#TODO reprojection function do not generally find the closest solution in matrix space
#This is a problem as converting to quaternion and back may result in a solution that is outside the domain to do there being multiple the same represnation in axis angle represenation(likely to due gimbal lock)
#TODO look at libaries like roma for better matrix and conversion functions




#first 2d cases


class Rotation2D(PeriodicTransform):
    """2D rotation transformation."""
    def __init__(self):
        self.dims = 2

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 2D rotation matrix for the given angle parameter.
        
        Args:
            param: Tensor of shape (..., 1) containing rotation angle in radians
            
        Returns:
            Homogeneous rotation matrix of shape (..., 3, 3)
        """
        batch_size = param.shape[:-1]
        # Create a rotation matrix
        rotation_matrix = identity(batch_size, dim=3, dtype=param.dtype, device=param.device)
        # Fill the rotation matrix with the angles
        angle = param.squeeze(-1)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation_matrix[..., 0, 0] = cos_angle
        rotation_matrix[..., 0, 1] = -sin_angle
        rotation_matrix[..., 1, 0] = sin_angle
        rotation_matrix[..., 1, 1] = cos_angle
        return rotation_matrix

    def param_size(self) -> int:
        return 1



    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi


class RotationComplex(Transform):
    """2D rotation using complex numbers."""

    def __init__(self):
        self.dims = 2

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 2D rotation matrix using complex number.

        Args:
            param: Tensor of shape (..., 2) with complex components [a, b]

        Returns:
            Homogeneous rotation matrix of shape (..., 3, 3)
        """
        # Normalize to unit complex number
        param = param / torch.norm(param, dim=-1, keepdim=True)

        a = param[..., 0]  # Real part
        b = param[..., 1]  # Imaginary part

        batch_size = param.shape[:-1]

        # Build 2x2 rotation matrix from complex components
        row0 = torch.stack([a, -b], dim=-1)
        row1 = torch.stack([b, a], dim=-1)
        R = torch.stack([row0, row1], dim=-2)

        # Create homogeneous matrix (3x3)
        H = identity(batch_size, 3, dtype=param.dtype, device=param.device)
        H[..., :2, :2] = R
        return H

    def param_size(self) -> int:
        return 2

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        """Project complex number parameters by normalizing to unit circle."""
        # For complex numbers, we normalize to the unit circle
        return param / torch.norm(param, dim=-1, keepdim=True)

    def calc_bounds(self, domain=None, dtype=torch.float32, device="cpu"):
        # Complex components lie in [-1,1] after normalization
        if domain is None or isinstance(domain, (int, float)):
            return torch.full((self.param_size(),), -1.0, dtype=dtype, device=device), torch.full((self.param_size(),),
                                                                                                  1.0, dtype=dtype,
                                                                                                  device=device)
        low = torch.full((self.param_size(),), -1.0,
                         dtype=dtype, device=device)
        high = torch.full((self.param_size(),), 1.0,
                          dtype=dtype, device=device)
        return low, high

    def orbit(self, n_samples: int, domain=2 * math.pi,dim=0, extend: int = 0, shift: int = 0) -> torch.Tensor:
        """Generate an orbit of complex rotation parameters on the unit circle."""
        # Reuse calc_bounds to properly parse domain

        low_p, high_p = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_p[dim].item(), high_p[dim].item()

        # Generate angles along the orbit
        if True:
            # For full circle, generate one extra point and discard the last one
            total_samples = n_samples + 2 * extend
            angles = torch.linspace(low_p, low_p + 2 * math.pi, total_samples + 1)[:-1]
            if shift != 0:
                angles = angles + shift * 2 * math.pi / n_samples

        # Convert angles to complex numbers on the unit circle
        real_part = torch.cos(angles)
        imag_part = torch.sin(angles)

        # Stack real and imaginary parts to get complex representation [a, b]
        complex_params = torch.stack([real_part, imag_part], dim=-1)

        return complex_params

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        # Normalize parameters to unit circle
        eps = 1e-8  # For numerical stability
        param1_norm = param1 / (torch.norm(param1, dim=-1, keepdim=True) + eps)
        param2_norm = param2 / (torch.norm(param2, dim=-1, keepdim=True) + eps)

        # Compute cosine similarity (dot product)
        cos_theta = torch.sum(param1_norm * param2_norm, dim=-1)

        # Clamp and calculate angular distance
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        return torch.arccos(cos_theta)

    def sample_param(self,batch_size,domain,device="cpu",dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter for the transformation.
        :return: The sampled parameter.
        """
        low, up = self.calc_bounds(domain, dtype=dtype, device=device)
        # Sample a point on the unit circle
        angle = torch.rand(batch_size, device=device, dtype=dtype) * (up - low) + low
        cos_val = torch.cos(angle)
        sin_val = torch.sin(angle)
        return torch.stack([cos_val, sin_val], dim=-1)


#3d cases

class Rotation3DEuler(PeriodicTransform):
    """3D rotation using Euler angles."""
    def __init__(self,extrinsic=True):
        super().__init__()
        self.dims = 3
        self.extrinsic = extrinsic  # Use extrinsic ZYX convention (yaw-pitch-roll)

    @staticmethod
    def static_matrix(param: torch.Tensor,extrinsic=True) -> torch.Tensor:
        """
        Create a 3D rotation matrix using extrinsic zyx Euler angles (yaw-pitch-roll).

        Args:
            param: Tensor of shape (..., 3) with rotation angles [yaw, pitch, roll]
                  - yaw (z-axis rotation): horizontal rotation
                  - pitch (y-axis rotation): vertical tilt
                  - roll (x-axis rotation): twist around forward axis

        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        batch_size = param.shape[:-1]

        # Use ZYX convention (yaw-pitch-roll) which is more intuitive
        yaw = param[..., 0]  # rotation around z-axis
        pitch = param[..., 1]  # rotation around y-axis
        roll = param[..., 2]  # rotation around x-axis

        # Create rotation matrices for each axis
        # X-axis rotation (roll)
        rotation_x = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_x[..., 0, 0] = 1
        rotation_x[..., 1, 1] = torch.cos(roll)
        rotation_x[..., 1, 2] = -torch.sin(roll)
        rotation_x[..., 2, 1] = torch.sin(roll)
        rotation_x[..., 2, 2] = torch.cos(roll)

        # Y-axis rotation (pitch)
        rotation_y = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_y[..., 0, 0] = torch.cos(pitch)
        rotation_y[..., 0, 2] = torch.sin(pitch)
        rotation_y[..., 1, 1] = 1
        rotation_y[..., 2, 0] = -torch.sin(pitch)
        rotation_y[..., 2, 2] = torch.cos(pitch)

        # Z-axis rotation (yaw)
        rotation_z = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_z[..., 0, 0] = torch.cos(yaw)
        rotation_z[..., 0, 1] = -torch.sin(yaw)
        rotation_z[..., 1, 0] = torch.sin(yaw)
        rotation_z[..., 1, 1] = torch.cos(yaw)
        rotation_z[..., 2, 2] = 1

        # Apply rotations in ZYX order: Z * Y * X
        # This means: first roll (X), then pitch (Y), then yaw (Z)
        if extrinsic:
            rotation_matrix = torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))
        else:
            rotation_matrix = torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z))

        # Create homogeneous transformation matrix
        id = identity(batch_size, dim=4, dtype=param.dtype, device=param.device)
        id[..., :3, :3] = rotation_matrix
        return id

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 3D rotation matrix using Euler angles.
        
        Args:
            param: Tensor of shape (..., 3) with rotation angles around x, y, z axes
            
        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        return self.static_matrix(param)


    def param_size(self) -> int:
        return 3

    def orbit(self, n_samples: int, domain=2*math.pi, extend: int = 0, shift: int = 0) -> None:
        """No orbit for multi-parameter transform."""
        return None

    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi


class Rotation3DEulerUniform(PeriodicTransform):
    """3D rotation using Euler angles."""

    def __init__(self, extrinsic=True):
        super().__init__()
        self.dims = 3
        self.extrinsic = extrinsic  # Use extrinsic ZYX convention (yaw-pitch-roll)

    @staticmethod
    def static_matrix(param: torch.Tensor, extrinsic=True) -> torch.Tensor:
        """
        Create a 3D rotation matrix using extrinsic zyx Euler angles (yaw-pitch-roll).

        Args:
            param: Tensor of shape (..., 3) with rotation angles [yaw, pitch, roll]
                  - yaw (z-axis rotation): horizontal rotation
                  - pitch (y-axis rotation): vertical tilt
                  - roll (x-axis rotation): twist around forward axis

        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        batch_size = param.shape[:-1]

        # Use ZYX convention (yaw-pitch-roll) which is more intuitive
        yaw = param[..., 0]  # rotation around z-axis
        pitch = param[..., 1]  # rotation around y-axis
        roll = param[..., 2]  # rotation around x-axis

        # Create rotation matrices for each axis
        # X-axis rotation (roll)
        rotation_x = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_x[..., 0, 0] = 1
        rotation_x[..., 1, 1] = torch.cos(roll)
        rotation_x[..., 1, 2] = -torch.sin(roll)
        rotation_x[..., 2, 1] = torch.sin(roll)
        rotation_x[..., 2, 2] = torch.cos(roll)

        # Y-axis rotation (pitch)
        rotation_y = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_y[..., 0, 0] = torch.cos(pitch)
        rotation_y[..., 0, 2] = torch.sin(pitch)
        rotation_y[..., 1, 1] = 1
        rotation_y[..., 2, 0] = -torch.sin(pitch)
        rotation_y[..., 2, 2] = torch.cos(pitch)

        # Z-axis rotation (yaw)
        rotation_z = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_z[..., 0, 0] = torch.cos(yaw)
        rotation_z[..., 0, 1] = -torch.sin(yaw)
        rotation_z[..., 1, 0] = torch.sin(yaw)
        rotation_z[..., 1, 1] = torch.cos(yaw)
        rotation_z[..., 2, 2] = 1

        # Apply rotations in ZYX order: Z * Y * X
        # This means: first roll (X), then pitch (Y), then yaw (Z)
        if extrinsic:
            rotation_matrix = torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))
        else:
            rotation_matrix = torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z))

        # Create homogeneous transformation matrix
        id = identity(batch_size, dim=4, dtype=param.dtype, device=param.device)
        id[..., :3, :3] = rotation_matrix
        return id

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 3D rotation matrix using Euler angles.

        Args:
            param: Tensor of shape (..., 3) with rotation angles around x, y, z axes

        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        return self.static_matrix(param)

    def param_size(self) -> int:
        return 3

    def orbit(self, n_samples: int, domain=2 * math.pi, extend: int = 0, shift: int = 0) -> None:
        """No orbit for multi-parameter transform."""
        return None

    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi


    def sample_param(self, batch_size, domain=None, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample Euler angles (yaw, pitch, roll) so that
        R_z(yaw) · R_y(pitch) · R_z(roll) is uniformly distributed in SO(3).

        - yaw   ~ Uniform(0, 2π)
        - cos(pitch) ~ Uniform(-1, 1)  → pitch = arccos(u)
        - roll  ~ Uniform(0, 2π)
        """
        # draw three independent U(0,1)
        u = torch.rand(batch_size, 3, device=device, dtype=dtype)
        # yaw in [0,2π)
        yaw  = 2 * math.pi * u[:, 0]
        # pitch via cosθ ∼ U(-1,1)
        pitch = torch.acos(2 * u[:, 1] - 1)
        # roll in [0,2π)
        roll = 2 * math.pi * u[:, 2]

        angles = torch.stack([yaw, pitch, roll], dim=-1)
        return angles







class RotationSkew3D(NormConstrainedTransform):
    """
    3D rotation using skew-symmetric matrix with direct axis-angle parameterization.

    This class provides a 3D-specific implementation that uses the standard mapping
    between axis-angle vectors and skew-symmetric matrices, avoiding the parameter
    reordering issues present in the general RotationSkew class.

    The parameters [vx, vy, vz] directly represent an axis-angle rotation vector:
    - Direction: axis of rotation (when normalized)
    - Magnitude: angle of rotation in radians

    Mathematical Background:
    - Uses the standard cross-product skew-symmetric matrix mapping
    - For vector v = [vx, vy, vz], the skew matrix [v]× is:
      [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
    - Rotation matrix: R = exp([v]×)

    """

    def __init__(self):
        """Initialize 3D skew-symmetric rotation transform."""
        self.dims = 3
        self.n_params = 3

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create rotation matrix using matrix exponential of skew-symmetric matrix.

        Constructs the skew-symmetric matrix [v]× from axis-angle vector v = [vx, vy, vz]
        using the standard cross-product mapping, then computes R = exp([v]×).

        The skew-symmetric matrix is:
        [v]× = [[0,  -vz,  vy],
                [vz,  0,  -vx],
                [-vy, vx,  0 ]]

        Args:
            param: Tensor of shape (..., 3) containing axis-angle vector [vx, vy, vz]

        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        batch_size = param.shape[:-1]
        vx, vy, vz = param.unbind(-1)

        # Create skew-symmetric matrix using standard cross-product mapping
        skew_matrix = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)

        # Fill skew-symmetric matrix with standard mapping
        # [v]× = [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
        skew_matrix[..., 0, 1] = -vz
        skew_matrix[..., 1, 0] = vz
        skew_matrix[..., 0, 2] = vy
        skew_matrix[..., 2, 0] = -vy
        skew_matrix[..., 1, 2] = -vx
        skew_matrix[..., 2, 1] = vx

        # Compute rotation matrix using matrix exponential
        rotation_matrix = torch.linalg.matrix_exp(skew_matrix)

        # Create homogeneous matrix (4×4)
        homogeneous_matrix = identity(batch_size, 4, dtype=param.dtype, device=param.device)
        homogeneous_matrix[..., :3, :3] = rotation_matrix

        return homogeneous_matrix



    def param_size(self) -> int:
        """Return the number of parameters (3 for 3D axis-angle)."""
        return self.n_params

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0) -> None:
        """
        Orbit computation not implemented for multi-parameter transforms_old.
        """
        return None

    def sample_param(self,batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        q = np.random.randn(batch_size,4)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)  # Normalize to unit quaternion
        return quaternion_to_skew_3d(q)





class RotationQuaternion(Transform):
    """3D rotation using quaternions."""
    def __init__(self):
        self.dims = 3

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 3D rotation matrix using quaternion.
        
        Args:
            param: Tensor of shape (..., 4) with quaternion components [w, x, y, z]
            
        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        # Normalize quaternion to ensure unit quaternion
        param = param / torch.norm(param, dim=-1, keepdim=True)

        # Expects param with shape (..., 4) in [w, x, y, z] format.
        w = param[..., 0]
        x = param[..., 1]
        y = param[..., 2]
        z = param[..., 3]

        batch_size = param.shape[:-1]
        r00 = 1 - 2 * (y ** 2 + z ** 2)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x ** 2 + z ** 2)
        r12 = 2 * (y * z - x * w)
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x ** 2 + y ** 2)
        row0 = torch.stack([r00, r01, r02], dim=-1)
        row1 = torch.stack([r10, r11, r12], dim=-1)
        row2 = torch.stack([r20, r21, r22], dim=-1)
        R = torch.stack([row0, row1, row2], dim=-2)
        H = identity(batch_size, dim=4, dtype=param.dtype, device=param.device)
        H[..., :3, :3] = R
        return H

    def param_size(self) -> int:
        return 4

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        """Project quaternion parameters by normalizing to unit quaternion."""
        # For quaternions, we just normalize to the unit hypersphere
        return param / torch.norm(param, dim=-1, keepdim=True)

    def calc_bounds(self, domain=None, dtype=torch.float32, device="cpu"):
        # Quaternion components lie in [-1,1] after normalization
        if domain is None or isinstance(domain, (int, float)):
            return torch.full((self.param_size(),), -1.0, dtype=dtype, device=device), torch.full((self.param_size(),), 1.0, dtype=dtype, device=device)
        # If domain is tensor-like, produce per-component bounds
        low = torch.full((self.param_size(),), -1.0,
                         dtype=dtype, device=device)
        high = torch.full((self.param_size(),),  1.0,
                          dtype=dtype, device=device)
        return low, high

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        # Normalize quaternions and account for double cover
        eps = 1e-8
        param1_norm = param1 / (torch.norm(param1, dim=-1, keepdim=True) + eps)
        param2_norm = param2 / (torch.norm(param2, dim=-1, keepdim=True) + eps)

        # Compute absolute dot product (handles q ≡ -q equivalence)
        dot = torch.abs(torch.sum(param1_norm * param2_norm, dim=-1))

        # Calculate geodesic distance on SO(3)
        dot = torch.clamp(dot, 0.0, 1.0 - eps)  # Avoid acos(>1)
        return 2 * torch.arccos(dot)

    def sample_param(self,batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        q = np.random.randn(batch_size,4)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)  # Normalize to unit quaternion
        return q


class DirectedRotation3D(PeriodicTransform):
        """Rotation in a specific axis direction in 3D."""
        def __init__(self, axis: int):
            self.dims = 3
            self.axis = axis  # 0: Z, 1: Y, 2: X

        def matrix(self, param: torch.Tensor) -> torch.Tensor:
            """
            Create a 3D rotation matrix that rotates around a specific axis.

            Args:
                param: Tensor of shape (..., 1) with rotation angle

            Returns:
                Homogeneous rotation matrix of shape (..., 4, 4)
            """
            # Create a full parameter vector with zeros except at the specified axis
            expanded_param = torch.zeros(param.shape[:-1] + (3,), dtype=param.dtype, device=param.device)
            expanded_param[..., self.axis] = param.squeeze(-1)

            # Use the Euler rotation matrix function
            return Rotation3DEuler.static_matrix(expanded_param)

        def param_size(self) -> int:
            return 1

        def interval(self):
            """Return the natural interval for this periodic transform."""
            return -math.pi, math.pi





class Rotation2Vec(BoundedTransform):
    """3D rotation using 2 vector representation."""
    def __init__(self):
        self.dims = 3

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Args:
            param: Tensor of shape (..., 6), split into two 3D vectors.
        Returns:
            Homogeneous (...,4,4) rotation matrices.
        """
        batch = param.shape[:-1]
        a1, a2 = param[..., :3], param[..., 3:6]
        # Normalize vectors
        c1 = a1 / torch.clamp(a1.norm(dim=-1, keepdim=True), min=1e-8)
        c2 = a2 - (c1 * a2).sum(-1, keepdim=True) * c1  # Orthogonalize a2 to a1

        # Gram–Schmidt
        b1 = c1 / torch.clamp(c1.norm(dim=-1, keepdim=True), min=1e-8)
        proj = (b1 * c2).sum(-1, keepdim=True) * b1
        b2 = (c2 - proj) / (c2 - proj)
        b2 = b2 / torch.clamp(b2.norm(dim=-1, keepdim=True), min=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)

        # Assemble 3×3 and embed into 4×4
        R = torch.stack([b1, b2, b3], dim=-1)  # (...,3,3)
        H = identity(batch, 4, dtype=param.dtype, device=param.device)
        H[..., :3, :3] = R
        return H

    def param_size(self) -> int:
        return 6

    def orbit(self, n_samples: int, domain=math.pi, extend: int = 0, shift: int = 0) -> torch.Tensor:
        return None

class Rotation3Vec(BoundedTransform):
    """3D rotation using a raw 9D matrix with Gram–Schmidt orthonormalization."""
    def __init__(self):
        self.dims = 3

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Args:
            param: Tensor of shape (..., 9), reshaped into (...,3,3).
        Returns:
            Homogeneous (...,4,4) rotation matrices.
        """
        batch = param.shape[:-1]
        M = param.view(*batch, 3, 3)

        return roma.mappings.procrustes(M)


    def param_size(self) -> int:
        return 9
    def orbit(self, n_samples: int, domain=math.pi, extend: int = 0, shift: int = 0) -> torch.Tensor:
        return None

class RotationVector(BoundedTransform):
        """3D rotation using axis–angle rotation vector (Rodrigues’ formula).
        Uses same parameterization as RotationSkew3D, but with a different
        mathematical formulation.
        """

        def __init__(self):
            self.dims = 3

        def matrix(self, param: torch.Tensor) -> torch.Tensor:
            """
            Args:
                param: Tensor of shape (..., 3), axis–angle vector (axis * angle)
            Returns:
                Homogeneous (...,4,4) rotation matrix.
            """
            batch = param.shape[:-1]
            # angle = ||v||, axis = v / angle
            eps = 1e-8
            angle = param.norm(dim=-1, keepdim=True)
            axis = param / (angle + eps)

            # build skew-symmetric K
            zero = torch.zeros(batch + (3, 3), dtype=param.dtype, device=param.device)
            K = zero.clone()
            K[..., 0, 1] = -axis[..., 2]
            K[..., 0, 2] = axis[..., 1]
            K[..., 1, 0] = axis[..., 2]
            K[..., 1, 2] = -axis[..., 0]
            K[..., 2, 0] = -axis[..., 1]
            K[..., 2, 1] = axis[..., 0]

            I3 = identity(batch, 3, dtype=param.dtype, device=param.device)
            sin_t = torch.sin(angle)[..., None]
            cos_t = torch.cos(angle)[..., None]
            R = I3 + sin_t * K + (1 - cos_t) * (K.matmul(K))

            H = identity(batch, 4, dtype=param.dtype, device=param.device)
            H[..., :3, :3] = R
            return H

        def param_size(self) -> int:
            return 3

        def orbit(self, n_samples: int, domain=math.pi, extend: int = 0, shift: int = 0) -> torch.Tensor:
            return None

        def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
            q = np.random.randn(batch_size, 4)
            q = q / np.linalg.norm(q, axis=-1, keepdims=True)  # Normalize to unit quaternion
            return quaternion_to_skew_3d(q)


#general do not use only for comparison/completness. Not really used. Not even slightly tested.
#Note that some use different conventions so they do NOT match the 3d specific cases.
class _RotationEuler(PeriodicTransform):
    """General N-dimensional rotation using Euler angles."""
    def __init__(self, dims: int,extrinsic=True):
        self.dims = dims
        self.n_params = dims * (dims - 1) // 2
        self.extrinsic = extrinsic  # Use extrinsic ZYX convention (yaw-pitch-roll)

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create an N-dimensional rotation matrix using Euler angles.
        Uses revese lexicographic upper-triangular ordering.
        In 3d this should be equivalent if passed Z-YX.

        Args:
            param: Tensor of shape (..., n_params) with Euler angles in
                   inverse-lexicographic order (e.g. for N=3,
                   param[...,0]=angle_x, param[...,1]=angle_y, param[...,2]=angle_z).

        Returns:
            Homogeneous rotation matrix of shape (..., N+1, N+1).
        """
        batch = param.shape[:-1]
        num_angles = param.shape[-1]
        # Solve N*(N-1)/2 = num_angles for N
        N = int((1 + (1 + 8 * num_angles) ** 0.5) / 2)

        # 1) All upper-triangular index pairs (i < j) in lexicographic order
        idx = torch.triu_indices(N, N, offset=1)  # shape=(2, N*(N-1)//2)
        i_indices = idx[0]  # shape: (num_angles,)
        j_indices = idx[1]  # shape: (num_angles,)

        rotation = identity(batch, N, dtype=param.dtype, device=param.device)
        for k in range(num_angles-1, -1, -1):
            i = i_indices[k].item()
            j = j_indices[k].item()
            angle_ij = param[..., k]

            c = torch.cos(angle_ij)
            s = torch.sin(angle_ij)

            local = identity(batch, N, dtype=param.dtype, device=param.device)
            local[..., i, i] = c
            local[..., j, j] = c
            local[..., i, j] = -s
            local[..., j, i] = s
            if self.extrinsic:
                rotation = torch.matmul(local, rotation)
            else:
                rotation = torch.matmul(local, rotation)

        # 5) Embed into homogeneous (N+1)×(N+1)
        H = identity(batch, N + 1, dtype=param.dtype, device=param.device)
        H[..., :N, :N] = rotation
        return H

    def param_size(self) -> int:
        return self.n_params

    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi

#general cases.
class _DirectedRotation(PeriodicTransform):
    """
    Rotation in a specific plane in N-dimensional space.
    For dimensions > 3, this specifies a rotation in a 2D subspace defined by two basis vectors.
    """

    def __init__(self, dims: int, plane: tuple):
        """
        Initialize a directed rotation in an N-dimensional space.

        Args:
            dims: The total number of dimensions of the space
            plane: A tuple of two integers specifying the plane of rotation (i, j) where 0 ≤ i, j < dims
        """
        self.dims = dims
        self.plane = plane  # The (i,j) indices defining the 2D plane of rotation
        if not (0 <= plane[0] < dims and 0 <= plane[1] < dims and plane[0] != plane[1]):
            raise ValueError(f"Invalid plane indices {plane} for {dims}-dimensional space")

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create an N-dimensional rotation matrix that rotates in a specific plane. Does not necessarily match the 3d specific case.

        Args:
            param: Tensor of shape (..., 1) with rotation angle

        Returns:
            Homogeneous rotation matrix of shape (..., dims+1, dims+1)
        """
        batch_size = param.shape[:-1]
        angle = param.squeeze(-1)

        # Create identity matrix as the base
        rotation = identity(batch_size, self.dims, dtype=param.dtype, device=param.device)

        # Extract plane indices
        i, j = self.plane

        # Fill in the rotation components for the specified plane
        cos_val = torch.cos(angle)
        sin_val = torch.sin(angle)

        rotation[..., i, i] = cos_val
        rotation[..., i, j] = -sin_val
        rotation[..., j, i] = sin_val
        rotation[..., j, j] = cos_val

        # Convert to homogeneous coordinates
        H = identity(batch_size, self.dims + 1, dtype=param.dtype, device=param.device)
        H[..., :self.dims, :self.dims] = rotation

        return H

    def param_size(self) -> int:
        return 1


    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi


class _RotationSkewGeneral(NormConstrainedTransform):
    """Rotation using skew-symmetric matrix."""

    def __init__(self, dims: int):
        self.dims = dims
        self.n_params = dims * (dims - 1) // 2

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a rotation matrix using the matrix exponential of a skew-symmetric matrix.

        Args:
            param: Tensor of shape (..., n_params) with entries for the skew-symmetric matrix

        Returns:
            Homogeneous rotation matrix
        """
        num_params = param.shape[-1]
        # calculate the dimension we rewuire n(n-1)/2 angles per dimension
        dim = int((1 + (1 + 8 * num_params) ** 0.5) / 2)
        batch_size = param.shape[:-1]
        # Create a rotation matrix using matrix exponentiation
        rotation_matrix = torch.zeros(batch_size + (dim, dim), dtype=param.dtype, device=param.device)
        # Fill the rotation matrix
        indices = torch.triu_indices(dim, dim, offset=1)
        rotation_matrix[..., indices[0], indices[1]] = param
        rotation_matrix[..., indices[1], indices[0]] = -param

        # use exp
        rotation_matrix = torch.linalg.matrix_exp(rotation_matrix)

        id = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        # copy rotation matrix to id
        id[..., :-1, :-1] = rotation_matrix
        return id

    def param_size(self) -> int:
        return self.n_params

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0) -> None:
        """No orbit for multi-parameter transform."""
        return None
    #TODO this only works in 3d.
    def sample_param(self,batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        q = np.random.randn(batch_size,4)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)  # Normalize to unit quaternion
        return quaternion_to_skew_general(q)



# 2d
Rotation2D = Rotation2D()
# New directed rotation instances
RotationSkew2DGeneral = _RotationSkewGeneral(2)  # 2D skew-symmetric rotation
RotationComplex2D = RotationComplex()
_RotationEuler2D = _RotationEuler(2) #this should match Rotation2D

DirectedRotation2D = Rotation2D
#3d
Rotation3DEulerIn = Rotation3DEuler(extrinsic=False)  # Intrinsic ZYX Euler angles
Rotation3DEuler = Rotation3DEuler()
RotationSkew3D = RotationSkew3D()  # 3D skew-symmetric rotation

RoationSkewGeneral3D = _RotationSkewGeneral(3)  # General N-dimensional skew-symmetric rotation
#skew should not match skew general

_RotationEulerGenereal3D = _RotationEuler(3) #should not match as plane xz is negative y rotation

RotationQuaternion3D = RotationQuaternion()
RotationVector3D = RotationVector()  # 3D axis-angle rotation vector
Rotation2Vec3D = Rotation2Vec()  # 3D rotation using two vectors
Rotation3Vec3D = Rotation3Vec()  # 3D rotation using a raw 9D matrix



RotationZ3D = DirectedRotation3D(0)
RotationY3D = DirectedRotation3D(1)
RotationX3D = DirectedRotation3D(2)

# Examples of directed rotations in higher dimensions
_DirectedRotation3D_XY = _DirectedRotation(3, (0, 1))  # Rotation in XY plane (equivalent to Z-axis rotation)
_DirectedRotation3D_XZ = _DirectedRotation(3, (0, 2))  # Rotation in XZ plane (equivalent to Y-axis rotation)
_DirectedRotation3D_YZ = _DirectedRotation(3, (1, 2))  # Rotation in YZ plane (equivalent to X-axis rotation)




def euler_to_quaternion(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3D Euler angles (roll, pitch, yaw) to quaternions (w, x, y, z).
    Expects `param` to have shape (..., 3) and returns a tensor of shape (..., 4).
    """
    y,p,r = param.unbind(-1)
    hr, hp, hy = 0.5 * r, 0.5 * p, 0.5 * y
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)

    w  = cr * cp * cy + sr * sp * sy
    x  = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z  = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, yq, z], dim=-1)

def quaternion_to_euler(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions (w, x, y, z) to 3D Euler angles (roll, pitch, yaw).
    Expects `param` to have shape (..., 4) and returns a tensor of shape (..., 3).
    """
    param = param / torch.norm(param, dim=-1, keepdim=True)  # Normalize quaternion
    w, x, yq, z = param.unbind(-1)

    t0 = 2 * (w * x + yq * z)
    t1 = 1 - 2 * (x * x + yq * yq)
    roll = torch.atan2(t0, t1)

    t2 = torch.clamp(2 * (w * yq - z * x), -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2 * (w * z + x * yq)
    t4 = 1 - 2 * (yq * yq + z * z)
    yaw = torch.atan2(t3, t4)

    angles= torch.stack([yaw,pitch,roll], dim=-1)
    return angles



def angle_to_complex(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 2D rotation angles θ to complex numbers (cos θ, sin θ).
    Expects `param` to have shape (..., 1) and returns a tensor of shape (..., 2).
    """
    θ = param.squeeze(-1)
    return torch.stack([torch.cos(θ), torch.sin(θ)], dim=-1)

def complex_to_angle(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 2D complex rotations (cos θ, sin θ) to angles θ.
    Expects `param` to have shape (..., 2) and returns a tensor of shape (..., 1).
    """
    param = param / torch.norm(param, dim=-1, keepdim=True)  # Normalize complex number
    θ = torch.atan2(param[..., 1], param[..., 0])
    return θ.unsqueeze(-1)


def quaternion_to_skew_general(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions (w, x, y, z) to skew parameters [p01, p02, p12]
    matching RotationSkew3D.matrix ordering.
    Expects `param` shape (...,4), returns (...,3).
    """
    param = param / torch.norm(param, dim=-1, keepdim=True)  # Normalize quaternion
    w, x, y, z = param.unbind(-1)

    # enforce w >= 0
    mask = w < 0
    w = torch.where(mask, -w, w)
    x = torch.where(mask, -x, x)
    y = torch.where(mask, -y, y)
    z = torch.where(mask, -z, z)

    eps = 1e-7
    half = torch.acos(torch.clamp(w, -1.0 + eps, 1.0 - eps))
    angle = 2.0 * half

    sin_half = torch.sin(half)
    scale = torch.where(
        angle < eps,
        torch.full_like(angle, 2.0),
        angle / sin_half
    )

    # axis-angle vector a = [a1, a2, a3]
    a1 = x * scale
    a2 = y * scale
    a3 = z * scale

    # map to skew params [p01, p02, p12] ugly
    p01 = -a3
    p02 =  a2
    p12 = -a1

    return torch.stack([p01, p02, p12], dim=-1)

def skew_general_to_skew_3d(param: torch.Tensor) -> torch.Tensor:
    """
    Convert skew parameters [p01, p02, p12] to axis-angle parameters [vx, vy, vz].
    Expects `param` shape (...,3), returns (...,3).
    """
    p01, p02, p12 = param.unbind(-1)

    # Reconstruct axis-angle vector a = [a1, a2, a3]
    # mapping: p01 = −a3, p02 = a2, p12 = −a1
    vx = -p12
    vy = p02
    vz = -p01

    return torch.stack([vx, vy, vz], dim=-1)

def skew_3d_to_skew_general(param: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle parameters [vx, vy, vz] to skew parameters [p01, p02, p12].
    Expects `param` shape (...,3), returns (...,3).
    """
    vx, vy, vz = param.unbind(-1)

    # map to skew params [p01, p02, p12] ugly
    p01 = -vz
    p02 =  vy
    p12 = -vx

    return torch.stack([p01, p02, p12], dim=-1)



def skew_general_to_quaternion(param: torch.Tensor) -> torch.Tensor:
    """
    Convert skew‐vector [p01, p02, p12] to quaternion [w, x, y, z],
    matching RotationSkew3D.matrix ordering.
    """
    # Unpack skew params
    p01, p02, p12 = param.unbind(-1)
    # Reconstruct axis-angle vector a = [a1, a2, a3]
    # mapping: p01 = −a3, p02 = a2, p12 = −a1
    vx = -p12
    vy = p02
    vz = -p01
    v = torch.stack([vx, vy, vz], dim=-1)

    # Compute angle and half-angle
    angle = torch.norm(v, dim=-1)
    half = 0.5 * angle
    cos_half = torch.cos(half)
    sin_half = torch.sin(half)

    # Avoid division by zero for small angles
    eps = 1e-7
    sin_half_over_angle = torch.where(
        angle < eps,
        torch.full_like(angle, 0.5),
        sin_half / angle
    )

    # Build quaternion and enforce w>=0
    w = cos_half
    x = v[..., 0] * sin_half_over_angle
    y = v[..., 1] * sin_half_over_angle
    z = v[..., 2] * sin_half_over_angle
    quat = torch.stack([w, x, y, z], dim=-1)

    mask = quat[..., 0] < 0
    quat[mask] = -quat[mask]
    return quat


def quaternion_to_skew_3d(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion [w, x, y, z] to axis-angle parameters [vx, vy, vz].

        This conversion is direct without parameter inversions since we use
        the standard axis-angle to skew-symmetric matrix mapping.

        Args:
            quat: Tensor of shape (..., 4) containing quaternions [w, x, y, z]

        Returns:
            Tensor of shape (..., 3) containing axis-angle parameters [vx, vy, vz]
        """
        w, x, y, z = quat.unbind(-1)

        # Ensure w >= 0 (choose shorter rotation path)
        mask = w < 0
        w = torch.where(mask, -w, w)
        x = torch.where(mask, -x, x)
        y = torch.where(mask, -y, y)
        z = torch.where(mask, -z, z)

        # Convert to axis-angle
        eps = 1e-7
        half_angle = torch.acos(torch.clamp(w, -1.0 + eps, 1.0 - eps))
        angle = 2.0 * half_angle
        sin_half = torch.sin(half_angle)

        # Compute scale factor for axis recovery
        scale = torch.where(
            angle < eps,
            torch.full_like(angle, 2.0),  # Small angle approximation
            angle / sin_half
        )

        # Axis-angle vector (no inversions needed!)
        vx = x * scale
        vy = y * scale
        vz = z * scale

        return torch.stack([vx, vy, vz], dim=-1)

def skew_3d_to_quaternion(param: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle parameters [vx, vy, vz] to quaternion [w, x, y, z].

        Args:
            param: Tensor of shape (..., 3) containing axis-angle parameters

        Returns:
            Tensor of shape (..., 4) containing quaternions [w, x, y, z]
        """
        vx, vy, vz = param.unbind(-1)
        v = torch.stack([vx, vy, vz], dim=-1)

        # Compute angle and half-angle
        angle = torch.norm(v, dim=-1)
        half_angle = 0.5 * angle
        cos_half = torch.cos(half_angle)
        sin_half = torch.sin(half_angle)

        # Avoid division by zero for small angles
        eps = 1e-7
        sin_half_over_angle = torch.where(
            angle < eps,
            torch.full_like(angle, 0.5),  # Small angle approximation
            sin_half / angle
        )

        # Build quaternion
        w = cos_half
        x = vx * sin_half_over_angle
        y = vy * sin_half_over_angle
        z = vz * sin_half_over_angle
        quat = torch.stack([w, x, y, z], dim=-1)

        # Ensure w >= 0 for canonical representation
        mask = quat[..., 0] < 0
        quat[mask] = -quat[mask]

        return quat





if __name__ == "__main__":
    print("Testing class-based rotation transforms_old...")
    
    # Create test data
    x_img = torch.randn(1, 1, 28, 28)  # 2D image
    x_pc = torch.randn(1, 1024, 3)    # 3D point cloud
    x_img_d = x_img.to(torch.double)  # Double precision for numeric gradient checks
    x_pc_d = x_pc.to(torch.double)

    # Test orbit with different domain formats
    print("\n12. Testing orbit with various domain formats:")

    # Test with scalar domain
    orbit_scalar = Rotation2D.orbit(6, math.pi / 2)
    print(f"Scalar domain orbit shape: {orbit_scalar.shape}")
    assert orbit_scalar.shape == (6, 1), f"Scalar domain orbit has wrong shape: {orbit_scalar.shape}"
    angles_scalar = orbit_scalar.squeeze().numpy()
    range_scalar = angles_scalar.max() - angles_scalar.min()
    assert abs(range_scalar - math.pi) < 1e-5, f"Expected range π/2, got {range_scalar}"
    print(f"✓ Scalar domain orbit covers correct range: {range_scalar:.3f}")

    # Test with vector domain
    domain_vec = torch.tensor([0.0, math.pi])
    orbit_vec = RotationY3D.orbit(5, domain_vec)
    print(f"Vector domain orbit shape: {orbit_vec.shape}")
    angles_vec = orbit_vec.squeeze().numpy()
    assert abs(angles_vec.min() - 0.0) < 1e-5, f"Expected min 0.0, got {angles_vec.min()}"
    assert abs(angles_vec.max() - math.pi) < 1e-5, f"Expected max π, got {angles_vec.max()}"
    print(f"✓ Vector domain orbit covers correct range: {angles_vec.min():.3f} to {angles_vec.max():.3f}")

    # Test with matrix domain
    domain_mat = torch.tensor([[-math.pi / 4, math.pi / 4]])
    orbit_mat = RotationZ3D.orbit(7, domain_mat)
    print(f"Matrix domain orbit shape: {orbit_mat.shape}")
    angles_mat = orbit_mat.squeeze().numpy()
    assert abs(angles_mat.min() - (-math.pi / 4)) < 1e-5, f"Expected min -π/4, got {angles_mat.min()}"
    assert abs(angles_mat.max() - math.pi / 4) < 1e-5, f"Expected max π/4, got {angles_mat.max()}"
    print(f"✓ Matrix domain orbit covers correct range: {angles_mat.min():.3f} to {angles_mat.max():.3f}")

    # Test full circle orbits without duplicating endpoints
    print("\n13. Testing full circle orbit sampling:")
    n_samples = 8
    orbit_full = Rotation2D.orbit(n_samples, math.pi)
    angles_full = orbit_full.squeeze().numpy()

    # Calculate angles from 0 to just under 2π
    expected_angles = torch.linspace(-math.pi, math.pi * (1 - 2 / n_samples), n_samples).numpy()
    assert np.allclose(angles_full, expected_angles), "Full circle orbit not sampled correctly"
    print("✓ Full circle orbit samples without duplicating endpoints")

    # Test 1: 2D Rotation
    print("\n1. Testing 2D rotation:")
    param_2d = torch.tensor([[0.5]], requires_grad=True)
    matrix_2d = Rotation2D.matrix(param_2d)
    out_2d = grid_resample(x_img, matrix_2d)
    out_2d.sum().backward()
    assert param_2d.grad is not None and param_2d.grad.abs().sum() > 0, "2D rotation gradient failed"
    print("✓ 2D rotation gradient check passed")
    
    # Test 2: 3D Euler Rotation
    print("\n2. Testing 3D Euler rotation:")
    param_3d_euler = torch.randn(1, 3, requires_grad=True)
    matrix_3d_euler = Rotation3DEuler.matrix(param_3d_euler)
    out_3d_euler = transform_3d_point_cloud(x_pc, matrix_3d_euler)
    out_3d_euler.sum().backward()
    assert param_3d_euler.grad is not None and param_3d_euler.grad.abs().sum() > 0, "3D Euler rotation gradient failed"
    print("✓ 3D Euler rotation gradient check passed")
    
    # Test 3: 3D Single-Axis Rotation (X-axis)
    print("\n3. Testing 3D X-axis rotation:")
    param_x = torch.tensor([[0.7]], requires_grad=True)
    matrix_x = RotationX3D.matrix(param_x)
    out_x = transform_3d_point_cloud(x_pc, matrix_x)
    out_x.sum().backward()
    assert param_x.grad is not None and param_x.grad.abs().sum() > 0, "X-axis rotation gradient failed"
    print("✓ X-axis rotation gradient check passed")
    
    # Test 4: 3D Y-axis and Z-axis rotations
    print("\n4. Testing 3D Y-axis and Z-axis rotations:")
    param_y = torch.tensor([[0.6]], requires_grad=True)
    matrix_y = RotationY3D.matrix(param_y)
    out_y = transform_3d_point_cloud(x_pc, matrix_y)
    out_y.sum().backward()
    assert param_y.grad is not None and param_y.grad.abs().sum() > 0, "Y-axis rotation gradient failed"
    print("✓ Y-axis rotation gradient check passed")
    
    param_z = torch.tensor([[0.4]], requires_grad=True)
    matrix_z = RotationZ3D.matrix(param_z)
    out_z = transform_3d_point_cloud(x_pc, matrix_z)
    out_z.sum().backward()
    assert param_z.grad is not None and param_z.grad.abs().sum() > 0, "Z-axis rotation gradient failed"
    print("✓ Z-axis rotation gradient check passed")
    
    # Test 5: Skew Rotation
    print("\n5. Testing skew rotation (3D):")
    param_skew = torch.randn(1, 3, requires_grad=True)
    matrix_skew = RotationSkew3D.matrix(param_skew)
    out_skew = transform_3d_point_cloud(x_pc, matrix_skew)
    out_skew.sum().backward()
    assert param_skew.grad is not None and param_skew.grad.abs().sum() > 0, "Skew rotation gradient failed"
    print("✓ Skew rotation gradient check passed")
    
    # Test 6: Quaternion Rotation
    print("\n6. Testing quaternion rotation:")
    param_quat = torch.randn(1, 4, requires_grad=True)
    # Normalize quaternion
    param_quat_norm = param_quat / torch.norm(param_quat, dim=-1, keepdim=True)
    matrix_quat = RotationQuaternion3D.matrix(param_quat_norm)
    out_quat = transform_3d_point_cloud(x_pc, matrix_quat)
    out_quat.sum().backward()
    assert param_quat.grad is not None and param_quat.grad.abs().sum() > 0, "Quaternion rotation gradient failed"
    print("✓ Quaternion rotation gradient check passed")
    
    # Test 7: Complex Number Rotation
    print("\n7. Testing complex number rotation:")
    param_complex = torch.randn(1, 2, requires_grad=True)
    # Normalize complex number
    param_complex_norm = param_complex / torch.norm(param_complex, dim=-1, keepdim=True)
    matrix_complex = RotationComplex2D.matrix(param_complex_norm)
    out_complex = grid_resample(x_img, matrix_complex)
    out_complex.sum().backward()
    assert param_complex.grad is not None and param_complex.grad.abs().sum() > 0, "Complex rotation gradient failed"
    print("✓ Complex rotation gradient check passed")
    
    # Test 8: Parameter Bounds and Projections
    print("\n8. Testing parameter bounds and projections:")
    
    # 2D rotation
    angle_over = torch.tensor([[4.0]])  # Outside of [-π, π]
    angle_proj = Rotation2D.project_parameters(angle_over, 2*math.pi)
    print(f"2D angle {angle_over[0][0]:.3f} projected to {angle_proj[0][0]:.3f}")
    assert -math.pi <= angle_proj[0][0] <= math.pi, "2D angle projection failed"
    
    # 3D Euler rotation
    angles_over = torch.tensor([[4.0, -4.0, 7.0]])  # Outside of [-π, π]
    angles_proj = Rotation3DEuler.project_parameters(angles_over, 2*math.pi)
    print(f"3D Euler angles {angles_over[0].tolist()} projected to {angles_proj[0].tolist()}")
    assert torch.all((angles_proj >= -math.pi) & (angles_proj <= math.pi)), "3D Euler angles projection failed"
    
    # Quaternion normalization
    quat_unnorm = torch.tensor([[2.0, 3.0, 4.0, 5.0]])  # Not normalized
    quat_norm = RotationQuaternion3D.project_parameters(quat_unnorm, None)
    norm = torch.norm(quat_norm, dim=-1)
    print(f"Quaternion normalized to unit norm: {norm.item():.6f}")
    assert torch.allclose(norm, torch.tensor(1.0)), "Quaternion normalization failed"
    
    # Complex number normalization
    complex_unnorm = torch.tensor([[3.0, 4.0]])  # Not normalized
    complex_norm = RotationComplex2D.project_parameters(complex_unnorm, None)
    norm = torch.norm(complex_norm, dim=-1)
    print(f"Complex number normalized to unit norm: {norm.item():.6f}")
    assert torch.allclose(norm, torch.tensor(1.0)), "Complex number normalization failed"
    
    # Test 9: Orbit Generation
    print("\n9. Testing orbit generation:")
    
    # 2D rotation orbit
    orbit_2d = Rotation2D.orbit(10, 2*math.pi)
    print(f"2D rotation orbit shape: {orbit_2d.shape}")
    assert orbit_2d.shape == (10, 1), f"2D rotation orbit has wrong shape: {orbit_2d.shape}"
    
    # Single-axis rotation orbit
    orbit_x = RotationX3D.orbit(8, 2*math.pi)
    print(f"X-axis rotation orbit shape: {orbit_x.shape}")
    assert orbit_x.shape == (8, 1), f"X-axis rotation orbit has wrong shape: {orbit_x.shape}"
    
    # Complex rotation orbit
    orbit_complex = RotationComplex2D.orbit(12, 2*math.pi)
    print(f"Complex rotation orbit shape: {orbit_complex.shape}")
    assert orbit_complex.shape == (12, 2), f"Complex rotation orbit has wrong shape: {orbit_complex.shape}"
    norms = torch.norm(orbit_complex, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms)), "Complex rotation orbit points are not unit norm"

    # Test 10: DirectedRotation2D
    print("\n10. Testing DirectedRotation2D:")
    param_2d_dir = torch.tensor([[0.5]], requires_grad=True)
    matrix_2d_dir = DirectedRotation2D.matrix(param_2d_dir)
    out_2d_dir = grid_resample(x_img, matrix_2d_dir)
    out_2d_dir.sum().backward()
    assert param_2d_dir.grad is not None and param_2d_dir.grad.abs().sum() > 0, "DirectedRotation2D gradient failed"
    print("✓ DirectedRotation2D gradient check passed")

    # Test 11: DirectedRotation in 3D
    print("\n11. Testing DirectedRotation in 3D:")
    param_3d_xy = torch.tensor([[0.7]], requires_grad=True)
    matrix_3d_xy = _DirectedRotation3D_XY.matrix(param_3d_xy)
    out_3d_xy = transform_3d_point_cloud(x_pc, matrix_3d_xy)
    out_3d_xy.sum().backward()
    assert param_3d_xy.grad is not None and param_3d_xy.grad.abs().sum() > 0, "DirectedRotation3D_XY gradient failed"
    print("✓ DirectedRotation3D_XY gradient check passed")

    # Compare with equivalent RotationZ3D
    matrix_z = RotationZ3D.matrix(param_3d_xy)
    assert torch.allclose(matrix_3d_xy, matrix_z, atol=1e-6), "DirectedRotation3D_XY should be equivalent to RotationZ3D"
    print("✓ DirectedRotation3D_XY is equivalent to RotationZ3D")

    # Test orbit generation for DirectedRotation
    orbit_dir_3d = _DirectedRotation3D_XY.orbit(10, 2*math.pi)
    print(f"DirectedRotation3D_XY orbit shape: {orbit_dir_3d.shape}")
    assert orbit_dir_3d.shape == (10, 1), f"DirectedRotation3D_XY orbit has wrong shape: {orbit_dir_3d.shape}"
    print("✓ DirectedRotation orbit generation works correctly")

    print("\nAll rotation tests passed!")
