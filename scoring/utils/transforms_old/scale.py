import torch
import math

from utils.helper import identity
from utils.transforms_old.base import Transform
from utils.transforms_old.bounded_transform import BoundedTransform


class Scale(Transform):
    """Non-uniform scaling in D dimensions."""
    def __init__(self, dims: int, log: bool = True):
        self.dims = dims
        self.log = log

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        dim = param.shape[-1]
        batch_size = param.shape[:-1]
        # Create a scaling matrix
        scaling_matrix = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        # Fill the scaling matrix with param + 1 to ensure identity for zero input
        scaling_matrix[..., :-1, :-1] = torch.diag_embed(param + 1.0)
        return scaling_matrix

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = self.dims

        if dom.ndim == 0:
            # single value → same interval for each dim
            dom = dom.expand(param_size)
            lower = 1.0 / (1.0 + dom) -1.0
            upper = dom
        elif dom.ndim == 1:
            # vector → interval per dim
            lower = dom[0].unsqueeze(0).expand(param_size)
            upper = dom[1].unsqueeze(0).expand(param_size)
        elif dom.ndim == 2:
            # each row [low_i, high_i]; expand if needed
            if dom.shape[0] != param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain tensor shape: {dom.shape}")

        # ensure lower ≤ upper
        min_vals = torch.min(lower, upper)
        max_vals = torch.max(lower, upper)
        return min_vals, max_vals

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, lower, upper)
        # choose log or linear per self.log
        if self.log:
            # log‐space reflection per dim
            s = param + 1.0
            lm = torch.log(lower + 1.0)
            um = torch.log(upper + 1.0)
            span = um - lm
            x = torch.log(s) - lm
            period = 2 * span
            mod = torch.remainder(x, period)
            refl_log = torch.where(mod <= span, mod, period - mod) + lm
            return torch.exp(refl_log) - 1.0
        else:
            # linear reflection
            span = upper - lower
            x = param - lower
            period = 2 * span
            mod = torch.remainder(x, period)
            return torch.where(mod <= span, mod, period - mod) + lower

    def param_size(self) -> int:
        return self.dims

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0) -> None:
        # multi-param transform: no single-parameter orbit
        return None



    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Return a nonnegative tensor of shape (dims,) measuring how far each
        scale-parameter is outside [lower, upper].  If self.log=True, measure
        distance in log‐space on (param+1); else measure in linear param‐space.

        In detail:
          • If log=True, let s_i = param_i + 1.0, and compute
            violation_i = max(0, log(lower_i+1) − log(s_i)) + max(0, log(s_i) − log(upper_i+1)).
          • If log=False, simply
            violation_i = max(0, lower_i − param_i) + max(0, param_i − upper_i).
        """
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)

        if self.log:
            s = param + 1.0
            llog = torch.log(lower + 1.0)
            ulog = torch.log(upper + 1.0)
            slog = torch.log(s)
            violation_below = torch.relu(llog - slog)
            violation_above = torch.relu(slog - ulog)
            return violation_below + violation_above

        else:
            violation_below = torch.relu(lower - param)
            violation_above = torch.relu(param - upper)
            return violation_below + violation_above

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance between two scale parameter tensors.

        If log=True, computes distance in the log-scale space, which is more
        appropriate for scaling factors. For example, scaling by 0.5x and 2x
        should have the same distance from 1x.

        Args:
            param1: First parameter tensor
            param2: Second parameter tensor

        Returns:
            Distance between the parameters (scalar tensor)
        """
        if self.log:
            # Convert to scale space (add 1.0) and compute distance in log space
            s1 = param1 + 1.0
            s2 = param2 + 1.0
            log_s1 = torch.log(s1)
            log_s2 = torch.log(s2)
            return torch.norm(log_s1 - log_s2, p=2, dim=-1)
        else:
            # Use standard Euclidean distance in parameter space
            return torch.norm(param1 - param2, p=2, dim=-1)


class ScaleAllSame(Transform):
    """Uniform scaling in all D dimensions."""
    def __init__(self, dims: int, log: bool = True):
        self.dims = dims
        self.log = log

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        scales = param.expand(param.shape[:-1] + (self.dims,))
        batch_size = scales.shape[:-1]
        matrix = identity(batch_size, self.dims + 1, dtype=param.dtype, device=param.device)
        # Add 1.0 to ensure identity when param is zero
        matrix[..., :-1, :-1] = torch.diag_embed(scales + 1.0)
        return matrix

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = 1
        if dom.ndim == 0:
            dom = dom.expand(param_size)
            lower = 1.0 / (1.0 + dom) -1.0
            upper = dom
        elif dom.ndim == 1:
            lower = dom[0].unsqueeze(0).expand(param_size)
            upper = dom[1].unsqueeze(0).expand(param_size)
        elif dom.ndim == 2:
            if dom.shape[0] != param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return lower, upper

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, lower, upper)
        # log‐space or linear
        if self.log:
            s = param + 1.0
            lm = torch.log(lower + 1.0)
            um = torch.log(upper + 1.0)
            span = um - lm
            x = torch.log(s) - lm
            period = 2 * span
            mod = torch.remainder(x, period)
            refl = torch.where(mod <= span, mod, period - mod) + lm
            return (torch.exp(refl) - 1.0)
            span = upper - lower
            x = param - lower
            period = 2 * span
            mod = torch.remainder(x, period)
            refl = torch.where(mod <= span, mod, period - mod) + lower
            return refl.unsqueeze(-1)

    def param_size(self) -> int:
        return 1

    def orbit(self, n_samples: int, domain, dim=0, extend: int = 0, shift: int = 0) -> torch.Tensor:
        # Reuse calc_bounds to properly parse domain
        low_p, high_p = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_p.item(), high_p.item()

        if self.log:
            # shift param‐domain by +1 to get scale factors >0
            s_min, s_max = low_p + 1.0, high_p + 1.0
            log_min, log_max = math.log(s_min), math.log(s_max)
            total = n_samples + 2 * extend
            spacing = (log_max - log_min) / (n_samples - 1) if n_samples > 1 else 0
            start = log_min - extend * spacing
            end = log_max + extend * spacing
            logs = torch.linspace(start, end, total) + shift * spacing
            scales = torch.exp(logs)
            params = scales - 1.0
            return params[..., None].expand(params.shape + (self.param_size(),))
        else:
            # linear sampling over positive param‐domain
            total = n_samples + 2 * extend
            rng = high_p - low_p
            spacing = rng / (n_samples - 1) if n_samples > 1 else 0
            start = low_p - extend * spacing
            end = high_p + extend * spacing
            lin = torch.linspace(start, end, total) + shift * spacing
            return lin[..., None].expand(lin.shape + (self.param_size(),))



    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Return a nonnegative tensor of shape (dims,) measuring how far each
        scale-parameter is outside [lower, upper].  If self.log=True, measure
        distance in log‐space on (param+1); else measure in linear param‐space.

        In detail:
          • If log=True, let s_i = param_i + 1.0, and compute
            violation_i = max(0, log(lower_i+1) − log(s_i)) + max(0, log(s_i) − log(upper_i+1)).
          • If log=False, simply
            violation_i = max(0, lower_i − param_i) + max(0, param_i − upper_i).
        """
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)

        if self.log:
            s = param + 1.0
            llog = torch.log(lower + 1.0)
            ulog = torch.log(upper + 1.0)
            slog = torch.log(s)
            violation_below = torch.relu(llog - slog)
            violation_above = torch.relu(slog - ulog)
            return violation_below + violation_above

        else:
            violation_below = torch.relu(lower - param)
            violation_above = torch.relu(param - upper)
            return violation_below + violation_above

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance between two uniform scale parameter tensors.

        If log=True, computes distance in the log-scale space, which is more
        appropriate for scaling factors.

        Args:
            param1: First parameter tensor
            param2: Second parameter tensor

        Returns:
            Distance between the parameters (scalar tensor)
        """
        if self.log:
            # Convert to scale space (add 1.0) and compute distance in log space
            s1 = param1 + 1.0
            s2 = param2 + 1.0
            log_s1 = torch.log(s1)
            log_s2 = torch.log(s2)
            return torch.abs(log_s1 - log_s2).squeeze(-1)
        else:
            # Use standard Euclidean distance in parameter space
            return torch.abs(param1 - param2).squeeze(-1)


class DirectedScale(Transform):
    """Scaling along a single axis in D dimensions."""
    def __init__(self, dims: int, axis: int, log: bool = True):
        self.dims = dims
        self.axis = axis
        self.log = log

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        res = torch.zeros(*param.shape[:-1], self.dims, dtype=param.dtype, device=param.device)
        res[..., self.axis] = param.squeeze(-1)

        batch_size = res.shape[:-1]
        dim = res.shape[-1]
        # Create a scaling matrix
        scaling_matrix = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        # Fill the scaling matrix with param + 1 to ensure identity for zero input
        scaling_matrix[..., :-1, :-1] = torch.diag_embed(res + 1.0)

        return scaling_matrix

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = 1
        if dom.ndim == 0:
            dom = dom.expand(param_size)
            lower = 1.0 / (1.0 + dom) -1.0
            upper = dom
        elif dom.ndim == 1:
            lower = dom[0].unsqueeze(0).expand(param_size)
            upper = dom[1].unsqueeze(0).expand(param_size)
        elif dom.ndim == 2:
            if dom.shape[0] != param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return lower, upper

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, lower, upper)
        # log‐space or linear
        if self.log:
            s = param + 1.0
            lm = torch.log(lower + 1.0)
            um = torch.log(upper + 1.0)
            span = um - lm
            x = torch.log(s) - lm
            period = 2 * span
            mod = torch.remainder(x, period)
            refl = torch.where(mod <= span, mod, period - mod) + lm
            return (torch.exp(refl) - 1.0)
        else:
            span = upper - lower
            x = param - lower
            period = 2 * span
            mod = torch.remainder(x, period)
            return (torch.where(mod <= span, mod, period - mod) + lower)

    def param_size(self) -> int:
        return 1

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0) -> torch.Tensor:
        # Reuse calc_bounds to properly parse domain
        low_p, high_p = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_p.item(), high_p.item()
        
        if self.log:
            s_min, s_max = low_p + 1.0, high_p + 1.0
            log_min, log_max = math.log(s_min), math.log(s_max)
            total = n_samples + 2*extend
            spacing = (log_max - log_min)/(n_samples-1) if n_samples>1 else 0
            start = log_min - extend*spacing
            end   = log_max + extend*spacing
            logs  = torch.linspace(start, end, total) + shift*spacing
            scales = torch.exp(logs)
            params = scales - 1.0
            return params[..., None]
        else:
            total = n_samples + 2*extend
            rng   = high_p - low_p
            spacing = rng/(n_samples-1) if n_samples>1 else 0
            start = low_p - extend*spacing
            end   = high_p + extend*spacing
            lin = torch.linspace(start, end, total) + shift*spacing
            return lin[..., None]



    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Return a nonnegative tensor of shape (dims,) measuring how far each
        scale-parameter is outside [lower, upper].  If self.log=True, measure
        distance in log‐space on (param+1); else measure in linear param‐space.

        In detail:
          • If log=True, let s_i = param_i + 1.0, and compute
            violation_i = max(0, log(lower_i+1) − log(s_i)) + max(0, log(s_i) − log(upper_i+1)).
          • If log=False, simply
            violation_i = max(0, lower_i − param_i) + max(0, param_i − upper_i).
        """
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)

        if self.log:
            s = param + 1.0
            llog = torch.log(lower + 1.0)
            ulog = torch.log(upper + 1.0)
            slog = torch.log(s)
            violation_below = torch.relu(llog - slog)
            violation_above = torch.relu(slog - ulog)
            return violation_below + violation_above

        else:
            violation_below = torch.relu(lower - param)
            violation_above = torch.relu(param - upper)
            return violation_below + violation_above

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance between two directed scale parameter tensors.

        If log=True, computes distance in the log-scale space, which is more
        appropriate for scaling factors.

        Args:
            param1: First parameter tensor
            param2: Second parameter tensor

        Returns:
            Distance between the parameters (scalar tensor)
        """
        if self.log:
            # Convert to scale space (add 1.0) and compute distance in log space
            s1 = param1 + 1.0
            s2 = param2 + 1.0
            log_s1 = torch.log(s1)
            log_s2 = torch.log(s2)
            return torch.abs(log_s1 - log_s2).squeeze(-1)
        else:
            # Use standard Euclidean distance in parameter space
            return torch.abs(param1 - param2).squeeze(-1)


class Reflection(BoundedTransform):
    """Reflection across a hyperplane in D dimensions."""
    def __init__(self, dims: int, axis: int):
        super().__init__()
        self.dims = dims
        self.axis = axis

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        batch_size = param.shape[:-1]
        dim = self.dims
        reflection_matrix = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        p = torch.sign(param)
        reflection_matrix[..., self.axis, self.axis] = 1.0 * p.squeeze(-1)
        return reflection_matrix

    def param_size(self) -> int:
        return 1

    def project_parameters(self, param: torch.Tensor, domain=None, reflect: bool = False) -> torch.Tensor:
        # Force to exactly ±1
        return torch.sign(param).clamp(min=-1, max=1)

    def orbit(self, n_samples: int, domain=None, extend: int = 0, shift: int = 0) -> torch.Tensor:
        # Return the two possible reflections
        vals = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        return vals[:, None]




# Instantiate common transforms_old:
Scale2D         = Scale(2)
Scale3D         = Scale(3)
UniformScale2D  = ScaleAllSame(2)
UniformScale3D  = ScaleAllSame(3)
ScaleX2D        = DirectedScale(2, 0)
ScaleY2D        = DirectedScale(2, 1)
ScaleX3D        = DirectedScale(3, 0)
ScaleY3D        = DirectedScale(3, 1)
ScaleZ3D        = DirectedScale(3, 2)





if __name__ == "__main__":
    import torch
    from utils.transforms_old.apply import grid_resample, transform_3d_point_cloud

    print("=== Class-based Scale Tests ===")

    # 1. Zero-parameter identity checks
    zero2d = torch.zeros(1, 2)
    id3 = torch.eye(3).unsqueeze(0)
    assert torch.allclose(Scale2D.matrix(zero2d), id3), "Scale2D zero-param ≠ identity"

    zero1d = torch.zeros(1, 1)
    assert torch.allclose(UniformScale2D.matrix(zero1d), id3), "UniformScale2D zero-param ≠ identity"
    assert torch.allclose(ScaleX2D.matrix(zero1d), id3), "ScaleX2D zero-param ≠ identity"
    assert torch.allclose(ScaleY2D.matrix(zero1d), id3), "ScaleY2D zero-param ≠ identity"
    print("✓ Zero-parameter identity tests passed")

    # 2. Gradient check with 2D image
    param_img = torch.randn(1, 2, requires_grad=True)
    mat_img = Scale2D.matrix(param_img)
    x_img = torch.randn(1, 1, 28, 28)
    out_img = grid_resample(x_img, mat_img)
    out_img.sum().backward()
    assert param_img.grad is not None and param_img.grad.abs().sum() > 0, "Scale2D image grad failed"
    print("✓ Scale2D image gradient test passed")

    # 3. Numeric gradcheck for Scale2D
    param_img_d = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
    x_img_d = x_img.to(torch.double)
    fn_img = lambda p: grid_resample(x_img_d, Scale2D.matrix(p))
    assert torch.autograd.gradcheck(fn_img, (param_img_d,), eps=1e-6, atol=1e-4), "Scale2D gradcheck failed"
    print("✓ Scale2D gradcheck passed")

    # 4. Gradient check with 3D point cloud
    param_pc = torch.randn(1, 3, requires_grad=True)
    mat_pc = Scale3D.matrix(param_pc)
    x_pc = torch.randn(1, 1024, 3)
    out_pc = transform_3d_point_cloud(x_pc, mat_pc)
    out_pc.sum().backward()
    assert param_pc.grad is not None and param_pc.grad.abs().sum() > 0, "Scale3D point-cloud grad failed"
    print("✓ Scale3D point-cloud gradient test passed")

    # 5. Numeric gradcheck for Scale3D
    param_pc_d = torch.randn(1, 3, dtype=torch.double, requires_grad=True)
    x_pc_d = x_pc.to(torch.double)
    fn_pc = lambda p: transform_3d_point_cloud(x_pc_d, Scale3D.matrix(p))
    assert torch.autograd.gradcheck(fn_pc, (param_pc_d,), eps=1e-6, atol=1e-4), "Scale3D gradcheck failed"
    print("✓ Scale3D gradcheck passed")

    # 6. Uniform scaling 2D and 3D gradient checks
    p2 = torch.randn(1,1, requires_grad=True)
    m2 = UniformScale2D.matrix(p2)
    out2 = grid_resample(x_img, m2)
    out2.sum().backward()
    assert p2.grad.abs().sum()>0, "UniformScale2D grad failed"
    print("✓ UniformScale2D gradient test passed")

    p3 = torch.randn(1,1, requires_grad=True)
    m3 = UniformScale3D.matrix(p3)
    out3 = transform_3d_point_cloud(x_pc, m3)
    out3.sum().backward()
    assert p3.grad.abs().sum()>0, "UniformScale3D grad failed"
    print("✓ UniformScale3D gradient test passed")

    print("\nTesting project_parameters reflect behavior:")
    # choose a domain and an offset beyond the domain
    domain = 0.5
    delta = -1
    p = torch.tensor([[domain + delta]])
    # UniformScale2D
    low_u, up_u = UniformScale2D.calc_bounds(domain, dtype=p.dtype, device=p.device)
    print(low_u, up_u)
    refl_u = UniformScale2D.project_parameters(p, domain, reflect=True)
    clip_u = UniformScale2D.project_parameters(p, domain, reflect=False)
    print(f"UniformScale2D: param={p.item()} → reflect={refl_u.item()}, clip={clip_u.item()}")
    # DirectedScale example: ScaleX2D
    low_d, up_d = ScaleX2D.calc_bounds(domain, dtype=p.dtype, device=p.device)
    refl_d = ScaleX2D.project_parameters(p, domain, reflect=True)
    clip_d = ScaleX2D.project_parameters(p, domain, reflect=False)
    print(f"ScaleX2D: param={p.item()} → reflect={refl_d.item()}, clip={clip_d.item()}")
    print("Reflect tests completed.")
