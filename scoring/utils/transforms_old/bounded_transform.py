import math
from typing import Optional

import torch
from utils.transforms_old.base import Transform


class BoundedTransform(Transform):
    """Base class for transforms_old that require parameter bounds checking and projection."""
    
    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate lower and upper bounds for parameters based on the specified domain.
        
        Args:
            domain: Can be a scalar, a tuple (min, max), or a tensor with bound information
            dtype: dtype for the output tensors
            device: device for the output tensors

        Returns:
            A tuple of (lower_bounds, upper_bounds) as tensors
        """
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = self.param_size()
        
        if dom.ndim == 0:
            lower = -torch.ones(param_size, dtype=dtype, device=device) * dom
            upper = torch.ones(param_size, dtype=dtype, device=device) * dom
        elif dom.ndim == 1 and len(dom) == 2:
            lower = torch.ones(param_size, dtype=dtype, device=device) * dom[0]
            upper = torch.ones(param_size, dtype=dtype, device=device) * dom[1]
        elif dom.ndim == 2:
            if dom.shape[0] < param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain tensor shape: {dom.shape}")
        
        min_vals = torch.min(lower, upper)
        max_vals = torch.max(lower, upper)
        return min_vals, max_vals

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        """
        Project parameters to stay within the specified domain.

        Args:
            param: Parameter tensor to project
            domain: Domain specification (scalar, tuple, or tensor)
            reflect: If True, reflect parameters at the boundaries; if False, clamp them

        Returns:
            Projected parameter tensor
        """
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)

        if not reflect:
            return torch.clamp(param, min=lower, max=upper)

        span = upper - lower
        x = param - lower
        period = 2 * span
        mod = torch.remainder(x, period)
        return torch.where(mod <= span, mod, period - mod) + lower


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

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Euclidean distance between two parameter tensors.

        Args:
            param1: First parameter tensor
            param2: Second parameter tensor

        Returns:
            Distance between the parameters (scalar tensor)
        """
        # For standard bounded transforms_old, use regular Euclidean distance
        return torch.norm(param1 - param2, p=2, dim=-1)


    def __init__(self, log: bool = False):
        """
        Initialize the BoundedTransform.

        Args:
            log: If True, use logarithmic scaling for parameters; otherwise, use linear scaling.
        """
        super().__init__()
        self.log = log

    def orbit(self,
              n_samples: int,
              domain,
              dim=0,
              extend: int = 0,
              shift: int = 0) -> Optional[torch.Tensor]:

        # get per-dimension bounds and determine parameter dimension
        low_all, high_all = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        param_dim = low_all.numel()
        low_p, high_p = low_all[dim].item(), high_all[dim].item()

        # total samples including padding
        total = n_samples + 2 * extend

        if self.log:
            # logarithmic spacing in [low_p, high_p]
            s_min, s_max = low_p + 1.0, high_p + 1.0
            log_min, log_max = math.log(s_min), math.log(s_max)
            spacing = (log_max - log_min) / (n_samples - 1) if n_samples > 1 else 0
            start = log_min - extend * spacing
            logs = torch.linspace(start,
                                  log_max + extend * spacing,
                                  total) + shift * spacing
            values = torch.exp(logs) - 1.0
        else:
            # linear spacing in [low_p, high_p]
            rng = high_p - low_p
            spacing = rng / (n_samples - 1) if n_samples > 1 else 0
            start = low_p - extend * spacing
            values = torch.linspace(start,
                                    high_p + extend * spacing,
                                    total) + shift * spacing

        # embed into full parameter vectors
        params = torch.zeros((total, param_dim), dtype=torch.float32)
        params[:, dim] = values

        return params

