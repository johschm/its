import torch
from utils.transforms.base import Transform



class NormConstrainedTransform(Transform):
    """Base class for transforms_old with constraints on parameter norms rather than individual bounds."""

    def supports_sobol(self) -> bool:  # changed
        return True  # enable Sobol → we provide sobol_to_param mapping

    def sample_space_param_size(self):  # new
        # We use d dims for direction + 1 dim for radius
        return self.param_size() + 1

    def sobol_to_param(self, sparam: torch.Tensor, domain) -> torch.Tensor:  # new
        """
        Map Sobol samples to parameters while respecting the supplied domain.

        Cases:
          - Scalar domain (norm bound R): map to uniform samples in the L2-ball of radius R in R^d.
            Expected Sobol shape (..., d+1). If (..., d) is given, a fallback derives radius.
          - Bound domain ((low, high) or (d,2) tensor): linearly map the first d Sobol dims per coordinate.

        Args:
          sparam: (..., k) Sobol samples in [0,1], k == d+1 (preferred) or k == d.
          domain: scalar radius or bounds ((low, high) or tensor (d,2)).

        Returns:
          (..., d) parameter tensor.
        """
        d = self.param_size()
        last_dim = sparam.shape[-1]
        if last_dim not in (d, d + 1):
            raise ValueError(f"sparam last dim must be d ({d}) or d+1 ({d+1}), got {last_dim}")

        # Scalar → norm-constrained L2-ball of radius=domain
        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and domain.dim() == 0):
            # direction from first d components via inverse-normal
            u_dir = sparam[..., :d].clamp(1e-7, 1 - 1e-7)
            z = torch.sqrt(torch.tensor(2.0, dtype=sparam.dtype, device=sparam.device)) * torch.erfinv(2 * u_dir - 1)
            direction = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-12)

            # radius
            if last_dim == d + 1:
                u_r = sparam[..., -1].clamp(1e-7, 1 - 1e-7)
            else:
                # fallback: use the mean of u_dir as a surrogate for u_r
                u_r = u_dir.mean(dim=-1).clamp(1e-7, 1 - 1e-7)
            r = u_r ** (1.0 / d)

            R = torch.as_tensor(domain, dtype=sparam.dtype, device=sparam.device)
            return direction * (r.unsqueeze(-1) * R)

        # Otherwise: per-parameter bounds
        lower, upper = self.calc_bounds(domain, dtype=sparam.dtype, device=sparam.device)
        u = sparam[..., :d].clamp(0.0, 1.0)
        return lower + u * (upper - lower)

    def supports_orbit(self) -> bool:
        return False

    def support_calc_bounds(self) -> bool:
        return True

    def _project_norm(self, param: torch.Tensor, norm_bound: float) -> torch.Tensor:
        """
        Scale down any vector whose ℓ₂ norm exceeds `norm_bound`.
        """
        norm = param.norm(p=2, dim=-1, keepdim=True)
        scale = torch.where(norm > norm_bound,
                            norm_bound / (norm + 1e-12),
                            torch.ones_like(norm))
        return param * scale

    def _reflect_norm(self, param: torch.Tensor, norm_bound: float) -> torch.Tensor:
        """
        Reflect vectors whose norm exceeds `norm_bound` back into the ball by radial reflection.
        Vectors already inside (<= norm_bound) are unchanged.
        """
        rho = param.norm(p=2, dim=-1, keepdim=True)  # (...,1)
        # overflow > 0 only for out-of-bound vectors
        overflow = torch.relu(rho - norm_bound)
        # Periodic folding with period 2R
        overflow_mod = torch.remainder(overflow, 2 * norm_bound)
        # Reflected radius for outside points
        reflected_r = norm_bound - torch.abs(overflow_mod - norm_bound)
        # Desired radius: original rho if inside, reflected_r if outside
        desired_r = torch.where(overflow > 0, reflected_r, rho)
        scale = desired_r / (rho + 1e-12)
        return param * scale

    def _project_bounds(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Simple clamping into per-parameter bounds.
        """
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        return torch.clamp(param, min=lower, max=upper)

    def _reflect_bounds(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Reflect values into per-parameter bounds using modulo arithmetic.
        """
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        L = upper - lower
        mask = L > 0
        safe_L = torch.where(mask, L, torch.ones_like(L))
        z = param - lower
        z_mod = torch.remainder(z, 2 * safe_L)
        reflected_z = safe_L - torch.abs(z_mod - safe_L)
        reflected_param = torch.where(mask, lower + reflected_z, lower)
        return reflected_param

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        """
        Dispatch to norm-based projection/reflection for scalar `domain`, or to clamping/reflection for bounds.
        Uses param's dtype and device for bounds.
        """
        dtype = param.dtype
        device = param.device
        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and domain.dim() == 0):
            return self._reflect_norm(param, domain) if reflect else self._project_norm(param, domain)
        else:
            return self._reflect_bounds(param, domain) if reflect else self._project_bounds(param, domain)

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter vector within the specified domain.
        :param domain: The domain to sample from.
        :return: A sampled parameter vector.
        """
        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and domain.dim() == 0):
            # Sample uniformly inside the L2 norm ball (solid sphere)
            param_size = self.param_size()

            # First, sample direction uniformly (normalized Gaussian)
            gaussian = torch.randn(batch_size, param_size, device=device, dtype=dtype)
            direction = gaussian / (gaussian.norm(p=2, dim=-1, keepdim=True) + 1e-12)

            # For uniform sampling *inside* the ball (not just surface),
            # we need to scale by radius^(1/n) for uniformity by volume
            u = torch.rand(batch_size, device=device, dtype=dtype)  # Uniform sample in [0, 1)
            radius = domain * (u ** (1 / param_size))  # This creates uniform volume distribution

            return radius * direction
        else:
            lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
            # Sample uniformly within the bounds
            return torch.rand(batch_size, self.param_size(), device=device, dtype=dtype) * (upper - lower) + lower

    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        Return a nonnegative scalar (or vector) measuring how far **by norm** `param`
        lies outside the ℓ₂‐ball of radius=domain.  If `domain` is not a scalar,
        falls back to per‐coordinate “inside/outside” checks.
        """
        # If domain is a scalar → we care about ℓ₂‐norm violation
        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and domain.dim() == 0):
            # Compute ℓ₂‐norm of param
            rho = param.norm(p=2, dim=-1)  # shape (...)
            # If inside: violation = 0.  If outside: violation = rho - domain
            return torch.relu(rho - domain)

        # Otherwise domain is per‐coordinate, so reuse _reflect_bounds/_project_bounds style logic:
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        # Just do the same as BoundedTransform:
        violation_below = torch.relu(lower - param)
        violation_above = torch.relu(param - upper)
        return violation_below + violation_above

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Default neighbourhood for norm-constrained transforms:
          - If domain is scalar R: use per-parameter span = 2*R (i.e., [-R,R])
            Returns length param_size().
          - Otherwise delegate to calc_bounds span (upper-lower).
        """
        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and getattr(domain, "dim", lambda: 1)() == 0):
            R = float(domain) if not torch.is_tensor(domain) else float(domain.item())
            span = torch.ones(self.param_size(), dtype=dtype, device=device) * (2.0 * R)
            return torch.clamp(span, min=1e-8)
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        span = (upper - lower).to(dtype=dtype, device=device)
        return torch.clamp(span, min=1e-8)



    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        If `domain` is a scalar → treat as ℓ₂ norm constraint but return [-domain, domain] per param.
        Otherwise delegate to standard bound logic (like BoundedTransform).
        """
        param_size = self.param_size()

        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and domain.dim() == 0):
            # Fallback: allow each parameter in [-domain, domain]
            lower = -torch.ones(param_size, dtype=dtype, device=device) * domain
            upper = torch.ones(param_size, dtype=dtype, device=device) * domain
            return lower, upper

        # Otherwise interpret as per-parameter bounds
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        if dom.ndim == 1 and dom.numel() == 2:
            lower = torch.ones(param_size, dtype=dtype, device=device) * dom[0]
            upper = torch.ones(param_size, dtype=dtype, device=device) * dom[1]
        elif dom.ndim == 2:
            if dom.shape[0] < param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")

        return torch.min(lower, upper), torch.max(lower, upper)

if __name__ == "__main__":
    class DummyNormConstrainedTransform(NormConstrainedTransform):
        def param_size(self):
            return 2  # Example size

        def matrix(self, param):
            return torch.eye(self.param_size())  # simple homogeneous identity example

    transform = DummyNormConstrainedTransform()
    param = torch.tensor([6.0, 8.0])
    domain = 3.0

    projected = transform.project_parameters(param, domain, reflect=False)
    reflected = transform.project_parameters(param, domain, reflect=True)

    expected_proj = torch.tensor([1.8, 2.4])
    expected_ref = torch.tensor([0.6, 0.8])

    assert torch.allclose(projected, expected_proj, atol=1e-6), f"Projection failed: {projected} != {expected_proj}"
    assert torch.allclose(reflected, expected_ref, atol=1e-6), f"Reflection failed: {reflected} != {expected_ref}"

    param2 = torch.tensor([1.0, 1.0])
    for refl in (False, True):
        out = transform.project_parameters(param2, domain, reflect=refl)
        assert torch.allclose(out, param2), f"Unexpected change for in-bound vector: {out}"

    print("All main tests passed")