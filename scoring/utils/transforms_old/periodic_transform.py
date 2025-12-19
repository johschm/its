import torch
import math
from utils.transforms_old.base import Transform


def _generate_orbit_samples(low: torch.Tensor, high: torch.Tensor, n_samples: int,
                            extend: int = 0, shift: int = 0, is_full_circle: bool = False) -> torch.Tensor:
    """
    Helper function to generate orbit samples between low and high bounds.

    Args:
        low: Lower bound
        high: Upper bound
        n_samples: Number of samples to generate
        extend: Number of additional samples to extend on each side
        shift: Amount to shift the samples
        is_full_circle: If True, generate one extra sample and discard the last one to avoid duplicating start/end

    Returns:
        Tensor of shape (n_samples + 2*extend, 1) containing orbit samples
    """
    total_samples = n_samples + 2 * extend

    if is_full_circle:
        # For full circle, generate one extra point and discard the last one
        start_angle = low
        angles = torch.linspace(start_angle, start_angle + 2 * math.pi, total_samples + 1)[:-1]
        if shift != 0:
            angles = angles + shift * (2 * (high - low) / (n_samples - 1))
        return angles
    else:
        # For partial arcs
        spacing = (high - low) / (n_samples - 1) if n_samples > 1 else 0
        start = low - extend * spacing
        end = high + extend * spacing
        orbit = torch.linspace(start, end, total_samples)
        if shift != 0:
            orbit = orbit + shift * (2 * (high - low) / n_samples)
        return orbit

class PeriodicTransform(Transform):
    """Base class for transforms_old with periodic parameters (like rotations)"""
    
    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
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

        return lower, upper

    def reproject_to_interval(self, param: torch.Tensor) -> torch.Tensor:
        """
        Reprojects the parameters into the interval defined by the type of transformation.
        For periodic transformations, this will wrap the parameters into the specified domain.

        Args:
            param: The parameters to reproject.

        Returns:
            The reprojected parameters.
        """
        default_lower, default_upper = self.interval()

        # Wrap the parameters into the interval
        span_interval = default_upper - default_lower
        shifted = param - default_lower
        wrapped = torch.remainder(shifted, span_interval) + default_lower

        return wrapped

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        dtype = param.dtype
        device = param.device
        lower_bounds, upper_bounds = self.calc_bounds(domain, dtype=dtype, device=device)
        wrapped = self.reproject_to_interval(param)
        default_lower, default_upper = self.interval()

        # Wrap the parameters into the interval
        span_interval = default_upper - default_lower

        if reflect:
            # 1) detect if [ℓ, u] is “inverted” (i.e. ℓ > u means it wraps around the period boundary)
            invert = lower_bounds > upper_bounds

            # 2) compute a “continuous” upper bound u_mod = (u + span_interval) when inverted, else u
            u_mod = torch.where(invert, upper_bounds + span_interval, upper_bounds)

            # 3) shift any wrapped < ℓ (when inverted) into the second half of the real‐line interval
            w_mod = torch.where(
                invert & (wrapped < lower_bounds),
                wrapped + span_interval,
                wrapped
            )

            # 4) now the “allowed domain” lives as a single contiguous segment [ℓ, u_mod]
            span_domain = u_mod - lower_bounds
            period_domain = 2 * span_domain

            # 5) reflect w_mod around [ℓ, u_mod]
            y = w_mod - lower_bounds
            m = torch.remainder(y, period_domain)
            folded = torch.where(m <= span_domain, m, period_domain - m) + lower_bounds

            # 6) map anything ≥ β back into [α, β)
            reflected = torch.where(
                folded >= default_upper,
                folded - span_interval,
                folded
            )
            return reflected

        else:
            inside = (wrapped >= lower_bounds) & (wrapped <= upper_bounds)
            #ivert where lower_bounds is larger than upper_bounds
            invert = lower_bounds > upper_bounds
            inside2= (wrapped >= lower_bounds) | (wrapped <= upper_bounds)
            inside = torch.where(invert, inside2, inside)


            diff_l = torch.abs(wrapped - lower_bounds)
            diff_u = torch.abs(wrapped - upper_bounds)

            dist_l = torch.minimum(diff_l, span_interval - diff_l)
            dist_u = torch.minimum(diff_u, span_interval - diff_u)


            clamp_to_lower = dist_l < dist_u

            proj = torch.where(
                inside,
                wrapped,
                torch.where(clamp_to_lower, lower_bounds, upper_bounds)
            )
            return proj


    def boundary_violation(self, param: torch.Tensor, domain) -> torch.Tensor:
        """
        For each coordinate, compute the shortest circular distance from the wrapped value w
        into the allowed sub‐interval [ℓ, u]. If w ∈ [ℓ,u], violation = 0.
        Otherwise:
            if w < ℓ:
                d_direct = ℓ - w
                d_wrap   = (w - u) + span_interval
            else:  # w > u
                d_direct = w - u
                d_wrap   = (ℓ - w) + span_interval
            violation = min(d_direct, d_wrap).
        Returns a nonnegative tensor of the same shape as param.
        """
        dtype = param.dtype
        device = param.device

        # 1) compute per‐coordinate [ℓ_i, u_i]
        lower_bounds, upper_bounds = self.calc_bounds(domain, dtype=dtype, device=device)

        # 2) wrap each param into [α, β)
        default_lower, default_upper = self.interval()
        span_interval = default_upper - default_lower   # e.g. 2π if [-π,+π)
        shifted = param - default_lower
        wrapped = torch.remainder(shifted, span_interval) + default_lower

        # 3) figure out where wrapped is inside [ℓ,u]
        inside = (wrapped >= lower_bounds) & (wrapped <= upper_bounds)

        # 4) compute direct distance vs wrapped‐around distance
        #    – if wrapped < ℓ:  d_direct = ℓ - wrapped;  d_wrap = (wrapped - u) + span_interval
        #    – if wrapped > u:  d_direct = wrapped - u;  d_wrap = (ℓ - wrapped) + span_interval
        #    – if inside:      violation = 0
        # We can vectorize with torch.where:
        #   a) mask_left  = (wrapped < lower_bounds)
        #   b) mask_right = (wrapped > upper_bounds)
        mask_left  = wrapped < lower_bounds

        # direct distance:
        d_direct = torch.where(
            mask_left,
            lower_bounds - wrapped,     # ℓ - w  when w < ℓ
            wrapped - upper_bounds      # w - u  when w > u (zero if inside, but we'll override inside later)
        )

        # wrapped distance (go “around the other side” of the circle):
        d_wrap = torch.where(
            mask_left,
            (wrapped - upper_bounds) + span_interval,  # (w - u)+span
            (lower_bounds - wrapped) + span_interval   # (ℓ - w)+span
        )

        # If a coordinate is already inside, we force violation=0.  Otherwise take min(d_direct, d_wrap).
        d_min = torch.min(d_direct, d_wrap)
        violation = torch.where(inside, torch.zeros_like(d_min), d_min)
        return violation

    def boundary_violation_without_interval(self, param: torch.Tensor, domain) -> torch.Tensor:
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
        Calculate the shortest distance between two parameter tensors, taking periodicity into account.


        Args:
            param1: First parameter tensor
            param2: Second parameter tensor

        Returns:
            Distance between the parameters (scalar tensor)
        """
        # Get the period interval
        default_lower, default_upper = self.interval()
        span_interval = default_upper - default_lower  # e.g., 2π for [-π, π)

        # Calculate direct distance
        direct_diff = param2 - param1

        # Wrap the difference to account for periodicity
        wrapped_diff = torch.remainder(direct_diff + span_interval/2, span_interval) - span_interval/2

        # Calculate the Euclidean norm of the wrapped differences
        return torch.norm(wrapped_diff, p=2, dim=-1)

    def distance_domain(self, param1: torch.Tensor, param2: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Calculate the distance between two parameters, taking into account the bounds of the transformation.
        It is assumed that the param are already projected into the domain.

        Args:
            param1: First parameter tensor
            param2: Second parameter tensor
            domain: Optional domain specification (ignored in standard Euclidean distance)

        Returns:
            Distance between the parameters (scalar tensor)
        """
        #project both parameters to interval
        projected_param1 = self.project_parameters(param1, domain)
        projected_param2 = self.project_parameters(param2, domain)
        #TODO

    def orbit(self, n_samples: int, domain=2 * math.pi, dim: int = 0, extend: int = 0, shift: int = 0) -> torch.Tensor:
        """Generate an orbit of parameters, supporting wrapped domains where lower > upper."""
        low_vec, high_vec = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_vec[dim].item(), high_vec[dim].item()
        interval_low, interval_high = self.interval()
        # NEW: robust scalar extraction
        if torch.is_tensor(interval_low):
            interval_low = interval_low.item() if interval_low.ndim == 0 else interval_low.view(-1)[0].item()
        if torch.is_tensor(interval_high):
            interval_high = interval_high.item() if interval_high.ndim == 0 else interval_high.view(-1)[0].item()
        period = interval_high - interval_low
        epsilon = 1e-4

        wrap = low_p > high_p  # wrapped arc crossing boundary
        if wrap:
            high_mod = high_p + period  # make continuous ascending segment
            arc_len = high_mod - low_p
            is_full_circle = abs(arc_len - period) < epsilon
            if is_full_circle:
                samples = _generate_orbit_samples(low_p, low_p + period, n_samples, extend, shift, True)
            else:
                samples = _generate_orbit_samples(low_p, high_mod, n_samples, extend, shift, False)
        else:
            arc_len = high_p - low_p
            is_full_circle = abs(abs(arc_len) - period) < epsilon
            if is_full_circle:
                samples = _generate_orbit_samples(low_p, low_p + period, n_samples, extend, shift, True)
            else:
                samples = _generate_orbit_samples(low_p, high_p, n_samples, extend, shift, False)

        params = torch.zeros((n_samples + 2 * extend, self.param_size()), dtype=torch.float32)
        # Wrap samples back into fundamental interval
        span_interval = period
        wrapped = torch.remainder(samples - interval_low, span_interval) + interval_low
        params[:, dim] = wrapped
        params = self.project_parameters(params, domain, reflect=False)
        return params

    def sample_param(self,batch_size,domain,device="cpu",dtype=torch.float32) -> torch.Tensor:
        """
        Sample a batch of parameters from the periodic transform's domain.
        This method generates random parameters within the specified domain.

        Args:
            batch_size: Number of parameters to sample
            domain: Domain over which to sample the parameters
            device: Device to place the sampled parameters on
            dtype: Data type of the sampled parameters

        Returns:
            A tensor of shape (batch_size, param_size) containing sampled parameters
        """
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        default_lower, default_upper = self.interval()
        span_interval = default_upper - default_lower
        p = self.param_size()

        # clamp arc length into one period
        span_raw = torch.remainder(upper - lower, span_interval)
        span = torch.where(span_raw > 0, span_raw, span_interval)

        u = torch.rand(batch_size, p, dtype=dtype, device=device)
        raw = lower.unsqueeze(0) + u * span.unsqueeze(0)
        return self.reproject_to_interval(raw)





if __name__ == '__main__':
    import math
    import torch
    from utils.transforms_old.periodic_transform import PeriodicTransform

    class DummyPeriodic(PeriodicTransform):
        def matrix(self, param: torch.Tensor) -> torch.Tensor:
            # Dummy implementation for testing
            return param
        def param_size(self):
            return 1
        def interval(self):
            return torch.tensor([-1.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32)

    tp = DummyPeriodic()

    #now we have a domain over the interval from 0.9 to -0.9
    values = [-0.125, 0.125, -1.125, 1.05]
    domain = [0.9, -0.9]
    for reflect in (False, True):
        print(f"\nTesting with reflect={reflect}")
        for v in values:
            x = torch.tensor([v], dtype=torch.float32)
            y = tp.project_parameters(x, domain=domain, reflect=reflect)
            print(f" value={v: .2f} → projected = {y.item(): .2f}")


    values = [-0.125,0.125,-1.125,1.125]
    domain = [-0.1, 0.1]
    expected = [-0.075, 0.075, 0.075, -0.075]


    for reflect in (False, True):
        print(f"\nTesting with reflect={reflect}")
        for i,v in enumerate(values):
            x = torch.tensor([v], dtype=torch.float32)
            y = tp.project_parameters(x, domain=domain, reflect=reflect)
            if reflect:
                assert torch.allclose(y, torch.ones_like(y)*expected[i], atol=1e-6), f"Reflection failed: {y} != {expected[i]}"


    values = [-0.125,0.125,-1.125,1.125]
    domain = [-2, 2]
    for reflect in (False, True):
        print(f"\nTesting with reflect={reflect}")
        for v in values:
            x = torch.tensor([v], dtype=torch.float32)
            y = tp.project_parameters(x, domain=domain, reflect=reflect)
            print(f" value={v: .2f} → projected={y.item(): .2f}")


    # NEW TEST: orbit with wrapped domain (low > high)
    wrap_domain = [0.9, -0.9]
    orb = tp.orbit(n_samples=9, domain=wrap_domain, dim=0)
    inside = (orb[:, 0] >= wrap_domain[0]) | (orb[:, 0] <= wrap_domain[1])
    assert inside.all(), f"Orbit produced out-of-domain samples for wrapped domain: {orb[:,0]}"
    # Ensure coverage of arc endpoints
    assert (orb[:, 0] >= 0.9).any() and (orb[:, 0] <= -0.9).any(), "Wrapped arc endpoints missing in orbit."

    print("Wrapped domain orbit test passed.")


