import torch
from utils.transforms_old.base import Transform

class NormConstrainedTransform(Transform):
    """Base class for transforms_old with constraints on parameter norms rather than individual bounds."""
    
    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        If `domain` is a scalar → treat as ℓ₂ norm constraint but return [-domain, domain] per param.
        Otherwise delegate to standard bound logic (like BoundedTransform).
        """
        param_size = self.param_size()

        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and domain.dim() == 0):
            # Fallback: allow each parameter in [-domain, domain]
            lower = -torch.ones(param_size, dtype=dtype, device=device) * domain
            upper =  torch.ones(param_size, dtype=dtype, device=device) * domain
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
        # Handle division by zero by using a safe_L where L is zero
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
            if reflect:
                return self._reflect_norm(param,domain)
            else:
                return self._project_norm(param, domain)
        else:
            if reflect:
                return self._reflect_bounds(param, domain)
            else:
                return self._project_bounds(param, domain)


    def sample_param(self,batch_size, domain,device="cpu",dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter vector within the specified domain.
        :param domain: The domain to sample from.
        :return: A sampled parameter vector.
        """
        if isinstance(domain, (int, float)) or (torch.is_tensor(domain) and domain.dim() == 0):
            # Sample uniformly inside the L2 norm ball (solid sphere)
            param_size = self.param_size()

            # First, sample direction uniformly (normalized Gaussian)
            gaussian = torch.randn(batch_size,param_size, device=device, dtype=dtype)
            direction = gaussian / (gaussian.norm(p=2, dim=-1, keepdim=True) + 1e-12)

            # For uniform sampling *inside* the ball (not just surface),
            # we need to scale by radius^(1/n) for uniformity by volume
            u = torch.rand(batch_size, device=device, dtype=dtype)  # Uniform sample in [0, 1)
            radius = domain * (u**(1/param_size))  # This creates uniform volume distribution

            return radius * direction
        else:
            lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
            # Sample uniformly within the bounds
            return torch.rand(batch_size,self.param_size(), device=device, dtype=dtype) * (upper - lower) + lower


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

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the  distance between two parameter tensors.


        Args:
            param1: First parameter tensor
            param2: Second parameter tensor

        Returns:
            Distance between the parameters (scalar tensor)
        """
        return torch.norm(param1 - param2, p=2, dim=-1)

if __name__ == "__main__":
    class DummyNormConstrainedTransform(NormConstrainedTransform):
        def param_size(self):
            return 2  # Example size

        def matrix(self):
            return torch.eye(self.param_size(), dtype=torch.float32)  # Example matrix

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
