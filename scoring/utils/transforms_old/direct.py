import torch
import math
import torch.linalg as la
from utils.transforms_old.base import Transform
from utils.helper import identity
from utils.transforms_old.bounded_transform import BoundedTransform


class Direct(BoundedTransform):
    """Generic direct affine: params shape (..., d*(d+1))."""
    def __init__(self, dims: int):
        self.dims = dims

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        if not isinstance(param, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        d = self.dims

        # Start with identity matrix (more efficient)
        result = identity(param.shape[:-1], d + 1, dtype=param.dtype, device=param.device)

        # reshape flat params → (..., d, d+1)
        mat = param.reshape(*param.shape[:-1], d, d + 1)

        # Add parameter values to the matrix (this will modify the identity values where needed)
        result[..., :d, :d + 1] += mat

        return result


    def param_size(self) -> int:
        return self.dims*(self.dims+1)

    def orbit(self, *args, **kwargs) -> None:
        return None

class Direct2D(Direct):
    def __init__(self):
        super().__init__(2)

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        # Override to use the original function
        if not isinstance(param, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        if param.shape[-1] != 6:
            raise ValueError(f"Last dim must be 6 for 2D, got {param.shape[-1]}")

        # Start with identity matrix (more efficient)
        result = identity(param.shape[:-1], 3, dtype=param.dtype, device=param.device)

        # reshape to (...,2,3) and add to the appropriate part of the result
        mat_view = param.reshape(*param.shape[:-1], 2, 3)
        result[..., :2, :3] += mat_view

        return result

class Direct3D(Direct):
    def __init__(self):
        super().__init__(3)

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        if not isinstance(param, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        if param.shape[-1] != 12:
            raise ValueError(f"Last dim must be 12 for 3D, got {param.shape[-1]}")

        # Start with identity matrix (more efficient)
        result = identity(param.shape[:-1], 4, dtype=param.dtype, device=param.device)

        # reshape to (...,3,4) and add to the appropriate part of the result
        mat_view = param.reshape(*param.shape[:-1], 3, 4)
        result[..., :3, :4] += mat_view

        return result


class ExpAffineParams2D(Transform):
    """2D Lie‐exp affine: params [...,6] → 3×3."""
    def matrix(self, params: torch.Tensor) -> torch.Tensor:
        # ...existing code from affine_from_exp_params_2d...
        batch = params.shape[0]
        L = torch.zeros(batch, 3, 3, device=params.device, dtype=params.dtype)
        theta, log_sx, log_sy, sh = params[:,2], params[:,3], params[:,4], params[:,5]
        L_lin = torch.zeros(batch,2,2, device=params.device, dtype=params.dtype)
        L_lin[:,0,0], L_lin[:,1,1] = log_sx, log_sy
        L_lin[:,0,1], L_lin[:,1,0] = sh - theta, theta
        L[:,:2,:2] = L_lin
        L[:,0,2], L[:,1,2] = params[:,0], params[:,1]
        return la.matrix_exp(L)

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        N = self.param_size()  # 6
        if dom.ndim == 0:
            dom = dom.expand(N)
            low, up = -dom, dom
        elif dom.ndim == 1:
            low, up = -dom, dom
        elif dom.ndim == 2:
            if dom.shape[0] != N:
                dom = dom.expand(N, -1)
            low, up = dom[:, 0], dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return low, up

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = False) -> torch.Tensor:
        low, up = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, low, up)
        span = up - low
        x = param - low
        period = 2 * span
        mod = torch.remainder(x, period)
        return torch.where(mod <= span, mod, period - mod) + low

    def param_size(self) -> int:
        return 6

    def orbit(self, *args, **kwargs) -> None:
        return None

    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Returns a nonnegative tensor of the same shape as `param`, where each entry
        is how far that coordinate lies outside [lower_i, upper_i]. If inside, returns 0.
        """
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)

        # violation_below = max(0, lower - param)
        violation_below = torch.relu(lower - param)
        # violation_above = max(0, param - upper)
        violation_above = torch.relu(param - upper)

        return violation_below + violation_above


class ExpAffineParams3D(Transform):
    """3D Lie‐exp affine: params [...,12] → 4×4."""
    def matrix(self, params: torch.Tensor) -> torch.Tensor:
        B = params.shape[0]
        L = torch.zeros(B, 4, 4, device=params.device, dtype=params.dtype)

        # unpack
        rx, ry, rz = params[:, 3], params[:, 4], params[:, 5]
        log_sx, log_sy, log_sz = params[:, 6], params[:, 7], params[:, 8]
        sh_xy, sh_xz, sh_yz = params[:, 9], params[:, 10], params[:, 11]

        # build 3×3 linear block
        A = torch.zeros(B, 3, 3, device=params.device, dtype=params.dtype)
        A[:, 0, 0] = log_sx
        A[:, 1, 1] = log_sy
        A[:, 2, 2] = log_sz
        A[:, 0, 1] += -rz + sh_xy
        A[:, 0, 2] += ry + sh_xz
        A[:, 1, 0] += rz
        A[:, 1, 2] += -rx + sh_yz
        A[:, 2, 0] += -ry
        A[:, 2, 1] += rx

        L[:, :3, :3] = A
        L[:, 0, 3] = params[:, 0]
        L[:, 1, 3] = params[:, 1]
        L[:, 2, 3] = params[:, 2]

        return la.matrix_exp(L)

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        N = self.param_size()  # 12
        if dom.ndim == 0:
            dom = dom.expand(N)
            low, up = -dom, dom
        elif dom.ndim == 1:
            low, up = -dom, dom
        elif dom.ndim == 2:
            if dom.shape[0] != N:
                dom = dom.expand(N, -1)
            low, up = dom[:, 0], dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return low, up

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = False) -> torch.Tensor:
        low, up = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, low, up)
        span = up - low
        x = param - low
        period = 2 * span
        mod = torch.remainder(x, period)
        return torch.where(mod <= span, mod, period - mod) + low

    def param_size(self) -> int:
        return 12

    def orbit(self, *args, **kwargs) -> None:
        return None

    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Returns a nonnegative tensor of the same shape as `param`, where each entry
        is how far that coordinate lies outside [lower_i, upper_i]. If inside, returns 0.
        """
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)

        # violation_below = max(0, lower - param)
        violation_below = torch.relu(lower - param)
        # violation_above = max(0, param - upper)
        violation_above = torch.relu(param - upper)

        return violation_below + violation_above


class ExpAffineFullShear3D(Transform):
    """3D full‐shear Lie‐exp affine: params [...,15] → 4×4."""
    def matrix(self, params: torch.Tensor) -> torch.Tensor:
        B = params.shape[0]
        L = torch.zeros(B, 4, 4, device=params.device, dtype=params.dtype)

        # unpack all
        tx, ty, tz = params[:, 0], params[:, 1], params[:, 2]
        rx, ry, rz = params[:, 3], params[:, 4], params[:, 5]
        log_sx, log_sy, log_sz = params[:, 6], params[:, 7], params[:, 8]
        sh_xy, sh_yx = params[:, 9], params[:, 10]
        sh_xz, sh_zx = params[:, 11], params[:, 12]
        sh_yz, sh_zy = params[:, 13], params[:, 14]

        # build 3×3 block
        A = torch.zeros(B, 3, 3, device=params.device, dtype=params.dtype)
        A[:, 0, 0] = log_sx
        A[:, 1, 1] = log_sy
        A[:, 2, 2] = log_sz
        A[:, 0, 1] = -rz + sh_xy
        A[:, 1, 0] = rz + sh_yx
        A[:, 0, 2] = ry + sh_xz
        A[:, 2, 0] = -ry + sh_zx
        A[:, 1, 2] = -rx + sh_yz
        A[:, 2, 1] = rx + sh_zy

        L[:, :3, :3] = A
        L[:, 0, 3] = tx
        L[:, 1, 3] = ty
        L[:, 2, 3] = tz

        return la.matrix_exp(L)

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        N = self.param_size()  # 15
        if dom.ndim == 0:
            dom = dom.expand(N)
            low, up = -dom, dom
        elif dom.ndim == 1:
            low, up = -dom, dom
        elif dom.ndim == 2:
            if dom.shape[0] != N:
                dom = dom.expand(N, -1)
            low, up = dom[:, 0], dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return low, up

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = False) -> torch.Tensor:
        low, up = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, low, up)
        span = up - low
        x = param - low
        period = 2 * span
        mod = torch.remainder(x, period)
        return torch.where(mod <= span, mod, period - mod) + low

    def param_size(self) -> int:
        return 15

    def orbit(self, *args, **kwargs) -> None:
        return None

    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Returns a nonnegative tensor of the same shape as `param`, where each entry
        is how far that coordinate lies outside [lower_i, upper_i]. If inside, returns 0.
        """
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)

        # violation_below = max(0, lower - param)
        violation_below = torch.relu(lower - param)
        # violation_above = max(0, param - upper)
        violation_above = torch.relu(param - upper)

        return violation_below + violation_above


# common instances
Direct2d2         = Direct(2)
Direct3d2         = Direct(3)

Direct2D       = Direct2D()
Direct3D       = Direct3D()
Exp2D          = ExpAffineParams2D()
Exp3D          = ExpAffineParams3D()

FullShear3D    = ExpAffineFullShear3D()


if __name__ == "__main__":
    # verify 2D
    p2 = torch.zeros(1, 6)
    p_2_clone = p2.clone()
    m_gen2 = Direct2d2.matrix(p_2_clone)
    m_exp2 = Direct2D.matrix(p_2_clone)
    print(m_gen2)
    print(m_exp2)
    assert torch.allclose(p2, p_2_clone)

    assert torch.allclose(m_gen2, m_exp2), "2D affine mismatch"

    # verify 3D
    p3 = torch.randn(1, 12)
    m_gen3 = Direct3d2.matrix(p3)
    m_exp3 = Direct3D.matrix(p3)
    assert torch.allclose(m_gen3, m_exp3), "3D affine mismatch"

    # Test zero parameters produce identity for direct methods
    zero2 = torch.zeros(1, 6)
    id3 = identity(1, 3, dtype=zero2.dtype, device=zero2.device)
    assert torch.allclose(Direct2D.matrix(zero2), id3), "Zero params not identity 2D direct"

    zero3 = torch.zeros(1, 12)
    id4 = identity(1, 4, dtype=zero3.dtype, device=zero3.device)
    assert torch.allclose(Direct3D.matrix(zero3), id4), "Zero params not identity 3D direct"

    # Test zero parameters for exp methods
    zero_exp_2d = torch.zeros(1, 6)
    assert torch.allclose(Exp2D.matrix(zero_exp_2d), id3), "Zero not identity 2D exp"

    zero_exp_3d = torch.zeros(1, 12)
    assert torch.allclose(Exp3D.matrix(zero_exp_3d), id4), "Zero not identity 3D exp"

    zero_exp_full = torch.zeros(1, 15)
    assert torch.allclose(FullShear3D.matrix(zero_exp_full), id4), "Zero not identity full shear"

    print("All direct transformation tests passed!")

    # Add gradient checks similar to scale.py
    from utils.transforms_old.apply import grid_resample, transform_3d_point_cloud

    print("\nTesting gradient flow through direct transformation functions...")

    # Tests for zero parameters producing identity were already included above
    print("✓ Zero parameter tests already verified!")

    # ------------ Test 1: Direct 2D matrix with images ------------
    print("\n1. Testing direct 2D matrix with images:")
    param_img = torch.randn(1, 6, requires_grad=True)
    matrix_img = Direct2D.matrix(param_img)
    x_img = torch.randn(1, 1, 28, 28)
    res_img = grid_resample(x_img, matrix_img)
    res_img.sum().backward()
    assert param_img.grad is not None, "Image gradient is None"
    assert param_img.grad.abs().sum().item() > 0, "Image gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_img_d = torch.randn(1, 6, dtype=torch.double, requires_grad=True)
    x_img_d = x_img.to(torch.double)
    fn_img = lambda p: grid_resample(x_img_d, Direct2D.matrix(p))
    assert torch.autograd.gradcheck(fn_img, (param_img_d,), eps=1e-6, atol=1e-4), "Image gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 2: Direct 3D matrix with point cloud ------------
    print("\n2. Testing direct 3D matrix with point cloud:")
    param_pc = torch.randn(1, 12, requires_grad=True)
    matrix_pc = Direct3D.matrix(param_pc)
    x_pc = torch.randn(1, 1024, 3)
    out_pc = transform_3d_point_cloud(x_pc, matrix_pc)
    out_pc.sum().backward()
    assert param_pc.grad is not None, "Point cloud gradient is None"
    assert param_pc.grad.abs().sum().item() > 0, "Point cloud gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_pc_d = torch.randn(1, 12, dtype=torch.double, requires_grad=True)
    x_pc_d = x_pc.to(torch.double)
    fn_pc = lambda p: transform_3d_point_cloud(x_pc_d, Direct3D.matrix(p))
    assert torch.autograd.gradcheck(fn_pc, (param_pc_d,), eps=1e-6, atol=1e-4), "Point cloud gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 3: Exp params 2D with images ------------
    print("\n3. Testing exp params 2D with images:")
    param_exp_2d = torch.randn(1, 6, requires_grad=True)
    matrix_exp_2d = Exp2D.matrix(param_exp_2d)
    res_exp_2d = grid_resample(x_img, matrix_exp_2d)
    res_exp_2d.sum().backward()
    assert param_exp_2d.grad is not None, "Exp 2D image gradient is None"
    assert param_exp_2d.grad.abs().sum().item() > 0, "Exp 2D image gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_exp_2d_d = torch.randn(1, 6, dtype=torch.double, requires_grad=True)
    fn_exp_2d = lambda p: grid_resample(x_img_d, Exp2D.matrix(p))
    assert torch.autograd.gradcheck(fn_exp_2d, (param_exp_2d_d,), eps=1e-6, atol=1e-4), "Exp 2D gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 4: Exp params 3D with point cloud ------------
    print("\n4. Testing exp params 3D with point cloud:")
    param_exp_3d = torch.randn(1, 12, requires_grad=True)
    matrix_exp_3d = Exp3D.matrix(param_exp_3d)
    out_exp_3d = transform_3d_point_cloud(x_pc, matrix_exp_3d)
    out_exp_3d.sum().backward()
    assert param_exp_3d.grad is not None, "Exp 3D point cloud gradient is None"
    assert param_exp_3d.grad.abs().sum().item() > 0, "Exp 3D point cloud gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_exp_3d_d = torch.randn(1, 12, dtype=torch.double, requires_grad=True)
    fn_exp_3d = lambda p: transform_3d_point_cloud(x_pc_d, Exp3D.matrix(p))
    assert torch.autograd.gradcheck(fn_exp_3d, (param_exp_3d_d,), eps=1e-6, atol=1e-4), "Exp 3D gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 5: Full shear 3D with point cloud ------------
    print("\n5. Testing full shear 3D with point cloud:")
    param_full_3d = torch.randn(1, 15, requires_grad=True)
    matrix_full_3d = FullShear3D.matrix(param_full_3d)
    out_full_3d = transform_3d_point_cloud(x_pc, matrix_full_3d)
    out_full_3d.sum().backward()
    assert param_full_3d.grad is not None, "Full shear 3D point cloud gradient is None"
    assert param_full_3d.grad.abs().sum().item() > 0, "Full shear 3D point cloud gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_full_3d_d = torch.randn(1, 15, dtype=torch.double, requires_grad=True)
    fn_full_3d = lambda p: transform_3d_point_cloud(x_pc_d, FullShear3D.matrix(p))
    assert torch.autograd.gradcheck(fn_full_3d, (param_full_3d_d,), eps=1e-6, atol=1e-4), "Full shear 3D gradcheck failed"
    print("✓ Numeric gradient check passed")

    print("\nAll gradient checks passed!")
