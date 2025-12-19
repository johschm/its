import torch
from copy import deepcopy
from utils.transforms_old.apply import grid_resample

class TransformSequenceDiscrete:
    """
    Discrete transform sequence driven by integer indices into each transform's orbit.
    Each transform supplies an orbit: (K_i, param_size_i). Indices shape: (B, n_transforms).
    Requested n_samples may be larger than actually returned by a transform's orbit.
    """
    def __init__(self,
                 transformations,
                 domains,
                 n_samples,
                 application_method=grid_resample,
                 device="cpu",
                 dtype=torch.float32,
                 invert=False):
        self.device = device
        self.dtype = dtype
        self.transforms = transformations
        self.domains = domains
        self.invert = invert
        self.application_method = application_method

        if isinstance(n_samples, int):
            n_samples = [n_samples] * len(transformations)
        self.n_samples_req = n_samples  # keep requested
        # parameter sizes & total
        self.param_sizes = [t.param_size() for t in transformations]
        self.total_dim = sum(self.param_sizes)
        # build per-transform orbits
        self.orbits = []
        self.orbit_lengths = []
        for t, dom, nreq in zip(transformations, domains, n_samples):
            # let transform decide how many it returns (may be < nreq)
            # generic fallback: first dimension varied (dim=0)
            orbit = t.orbit(nreq, dom, dim=0)
            if orbit is None:
                raise ValueError(f"Transform {t} does not provide an orbit for discrete sequence.")
            #check that orbit shape is correct
            if orbit.dim() != 2 or orbit.size(1) != t.param_size():
                raise ValueError(f"Transform {t} orbit has invalid shape {orbit.shape}, expected (K, {t.param_size()})")
            orbit = orbit.to(device=device, dtype=dtype)
            self.orbits.append(orbit)
            self.orbit_lengths.append(orbit.shape[0])

    def total_param_dim(self):
        return self.total_dim

    def n_transforms(self):
        return len(self.transforms)

    def sample_indices(self, batch_size):
        """
        Returns (B, n_transforms) with per-transform orbit indices.
        """
        cols = [torch.randint(0, L, (batch_size,), device=self.device) for L in self.orbit_lengths]
        return torch.stack(cols, dim=1)

    def indices_to_param(self, indices: torch.Tensor):
        """
        Map indices (B, n_transforms) -> concatenated param tensor (B, total_dim)
        """
        if indices.dim() != 2 or indices.size(1) != self.n_transforms():
            raise ValueError(f"indices must have shape (B, {self.n_transforms()})")
        parts = []
        for idx_col, orbit in zip(indices.t(), self.orbits):
            parts.append(orbit[idx_col])  # (B, param_size_i)
        return torch.cat(parts, dim=-1)

    def __call__(self, indices: torch.Tensor, layer: int | None = None):
        """
        Build and return batched transformation matrices from orbit indices.
        indices: (B, n_transforms)
        layer: if not None, apply only transforms up to and including this layer (0-based).
        """
        params = self.indices_to_param(indices)  # (B, total_dim)
        splits = torch.split(params, self.param_sizes, dim=-1)

        if layer is not None:
            if layer < 0 or layer >= self.n_transforms():
                raise ValueError(f"layer {layer} out of range 0..{self.n_transforms()-1}")
            max_layers = layer + 1
        else:
            max_layers = self.n_transforms()

        T = None
        for i, (transform, p) in enumerate(zip(self.transforms, splits)):
            if i >= max_layers:
                break
            T = transform.apply(T, p)

        if self.invert:
            T = torch.linalg.inv(T)
        return T

    def sample_params(self, batch_size):
        """
        Convenience: returns (B, total_dim) by sampling indices.
        """
        idx = self.sample_indices(batch_size)
        return self.indices_to_param(idx)

    def sample_neighbors(self, indices,neighbourhood_size=None, wrap=True):
        """
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        # determine fraction of domain to cover
        if neighbourhood_size is None:
            frac = 1.0
        else:
            frac = float(neighbourhood_size)
        frac = max(0.0, min(1.0, frac))
        # simplified radius: 1/2 * frac * L_min, scaled by mult, with minimum 1
        L_min = int(min(self.orbit_lengths)) if len(self.orbit_lengths) > 0 else 1
        base = 0.5 * frac * L_min
        radius = max(1, int(base))
        # build neighborhood
        shifts = torch.arange(-radius, radius + 1, device=self.device).view(1, 1, -1)
        idx = indices.unsqueeze(-1)  # (B, T, 1)
        lengths = torch.tensor(self.orbit_lengths, device=self.device).view(1, -1, 1)
        neigh = idx + shifts
        if wrap:
            neigh = neigh % lengths
        else:
            neigh = torch.clamp(neigh, min=0)
            neigh = torch.minimum(neigh, lengths - 1)
        return neigh

    def get_inverted(self):
        inv = deepcopy(self)
        inv.invert = not inv.invert
        return inv

    def transform(self, x, indices, layer: int | None = None):
        """
        Apply transform(s) given orbit indices, optionally truncated at layer.
        """
        T = self(indices, layer=layer)
        return self.application_method(x, T)

    # -------- NEW: extended-index aware transform ----------
    def _call_with_extend(self, indices: torch.Tensor, extend: int = 0, layer: int | None = None):
        """
        Internal: like __call__ but allows negative / oversized indices via optional 'extend'.
        For each layer:
          - If extend == 0: indices wrapped modulo core orbit length.
          - If extend > 0 : rebuild extended orbit (core_len + 2*extend) via transform.orbit(..., extend=extend),
                            map idx_full = (idx + extend) % (core_len + 2*extend),
                            select parameter from extended orbit (keeping core mapping stable).
        Only transforms up to 'layer' (inclusive) are composed if layer is not None.
        Returns transformation matrices (B, d, d).
        """
        if indices.dim() != 2 or indices.size(1) != self.n_transforms():
            raise ValueError(f"indices must have shape (B, {self.n_transforms()})")
        if layer is not None and (layer < 0 or layer >= self.n_transforms()):
            raise ValueError(f"layer {layer} out of range 0..{self.n_transforms()-1}")

        max_layers = self.n_transforms() if layer is None else (layer + 1)
        B = indices.size(0)
        T = None
        for l in range(max_layers):
            core_len = self.orbit_lengths[l]
            if extend == 0:
                idx_l = indices[:, l] % core_len
                params_l = self.orbits[l][idx_l]  # (B,param_size_l)
            else:
                # build extended orbit on demand
                t = self.transforms[l]
                dom = self.domains[l]
                ext_orbit = t.orbit(self.n_samples_req[l], dom, dim=0, extend=extend, shift=0)
                if ext_orbit is None:
                    raise RuntimeError(f"Transform {t} did not return orbit for extend={extend}")
                if ext_orbit.dim() != 2 or ext_orbit.size(1) != t.param_size():
                    raise ValueError(f"Extended orbit shape mismatch got {ext_orbit.shape}")
                ext_orbit = ext_orbit.to(device=self.device, dtype=self.dtype)
                full_len = ext_orbit.shape[0]  # should be core_len + 2*extend
                idx_full = (indices[:, l] + extend) % full_len
                params_l = ext_orbit[idx_full]
            T = self.transforms[l].apply(T, params_l)
        if self.invert:
            T = torch.linalg.inv(T)
        return T

    def transform_with_extend(self,
                              x: torch.Tensor,
                              indices: torch.Tensor,
                              extend: int = 0,
                              layer: int | None = None):
        """
        Apply transformation sequence to images with support for negative / oversized indices
        through an 'extend' padding mechanism (mirrors original orbit padding logic).
        """
        # Compute core length for this layer
        if layer is None:
            layer = slice(None)
        core_len = self.orbit_lengths[layer] if isinstance(layer, int) else max(self.orbit_lengths)
        full_len = core_len + 2 * extend

        # Wrap indices into [0, full_len-1]
        indices_safe = (indices + extend) % full_len

        T = self._call_with_extend(indices_safe, extend=extend, layer=layer)
        return self.application_method(x, T)

    # -------- New: conversion to continuous TransformSequence --------
    def to_continuous(self,
                      neighbour_hood_size=None,
                      application_method=None,
                      init_method="individual",
                      reflect=False):
        """
        Convert this discrete sequence to a continuous TransformSequence.

        Note: This assumes your transform objects are compatible with TransformSequence
        (i.e., support the interface expected there). If not, this may raise at construction.
        """
        from utils.transform_sequence import TransformSequence
        app = application_method if application_method is not None else self.application_method
        return TransformSequence(
            transformations=self.transforms,
            domains=self.domains,
            neighbour_hood_size=neighbour_hood_size,
            application_method=app,
            device=self.device,
            dtype=self.dtype,
            init_method=init_method,
            use_individual_param_correction=False,
            reflect=reflect,
            invert=self.invert
        )


# ---------------- Test / Example (updated) ----------------
from utils.transforms_old.base import Transform  # assuming old base path; adjust if needed

class DummyScale(Transform):
    """
    Simple 2D scale (uniform) around identity (3x3 homogeneous).
    Domain: (low, high)
    """
    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        # param shape (B,1); produce (B,3,3)
        B = param.shape[0]
        I = torch.eye(3, device=param.device, dtype=param.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        I[:, 0, 0] = param.squeeze(-1)
        I[:, 1, 1] = param.squeeze(-1)
        I[:, 2, 2] = 1.0
        return I

    def param_size(self) -> int:
        return 1

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32):
        low, high = domain
        return torch.rand(batch_size, 1, device=device, dtype=dtype) * (high - low) + low

    def supports_sobol(self) -> bool:
        return False

    def supports_orbit(self) -> bool:
        return True

    def calc_bounds(self, domain, dtype=None, device=None):
        if dtype is None: dtype = torch.float32
        if device is None: device = "cpu"
        low = torch.tensor([domain[0]], dtype=dtype, device=device)
        high = torch.tensor([domain[1]], dtype=dtype, device=device)
        return low, high

    def orbit(self, n_samples: int, domain, dim=0, extend: int = 0, shift: int = 0):
        """
        Override: evenly spaced in actual domain with optional extend padding.
        """
        low, high = domain
        if n_samples < 1:
            raise ValueError("n_samples must be >=1")
        if extend == 0:
            vals = torch.linspace(low, high, n_samples)
        else:
            # build extended uniformly (include endpoints for padding)
            core = torch.linspace(low, high, n_samples)
            spacing = (core[1] - core[0]) if n_samples > 1 else torch.tensor(0., dtype=core.dtype)
            left = core[0] - torch.arange(extend, 0, -1, dtype=core.dtype) * spacing
            right = core[-1] + torch.arange(1, extend + 1, dtype=core.dtype) * spacing
            vals = torch.cat([left, core, right], dim=0)
        return vals.unsqueeze(-1)

def main():
    transforms = [DummyScale()]
    # changed domain so that identity (scale=1.0) is the middle sample
    domains = [(0.5, 1.5)]
    n_samples = 3
    seq = TransformSequenceDiscrete(transforms, domains, n_samples)

    # total_param_dim / param_sizes
    assert seq.total_param_dim() == 1
    assert seq.param_sizes == [1]
    print("✔ sizes")

    # orbit correctness
    orbit_vals = seq.orbits[0].squeeze(-1)
    # updated expected values
    assert torch.allclose(orbit_vals, torch.tensor([0.5, 1.0, 1.5]))
    print("✔ orbit values")

    # NEW: identity in the middle (lower if even length)
    mid_index = (seq.orbit_lengths[0] - 1) // 2
    T_mid = seq(torch.tensor([[mid_index]]))
    assert torch.allclose(T_mid[0], torch.eye(3)), "Middle orbit element must be identity transform."
    print("✔ identity at middle")

    # sample_indices / indices_to_param
    batch = 4
    idx = seq.sample_indices(batch)
    assert idx.shape == (batch, 1)
    params = seq.indices_to_param(idx)
    expected = orbit_vals[idx.squeeze(1)]
    assert torch.allclose(params.squeeze(1), expected)
    print("✔ index->param mapping")

    # build matrices
    all_idx = torch.tensor([[0], [1], [2]])
    T = seq(all_idx)
    for i in range(3):
        expected_mat = torch.diag(torch.tensor([orbit_vals[i], orbit_vals[i], 1.0]))
        assert torch.allclose(T[i], expected_mat)
    print("✔ matrix build")

    # transform application (dummy image)
    x = torch.randn(1, 3, 8, 8)
    out = seq.transform(x, all_idx[:1])
    assert out.shape == x.shape
    print("✔ transform apply")

    # neighbors
    nei = seq.sample_neighbors(torch.tensor([[0], [2]]), radius=1)
    exp = torch.tensor([[[2, 0, 1]], [[1, 2, 0]]])
    assert torch.equal(nei, exp)
    print("✔ neighbors")

    # inversion (use non-identity index = 2 -> scale 1.5)
    inv = seq.get_inverted()
    assert inv.invert
    inv_T = inv(torch.tensor([[2]]))
    scale = 1.0 / 1.5
    expected_inv = torch.diag(torch.tensor([scale, scale, 1.0])).unsqueeze(0)
    assert torch.allclose(inv_T, expected_inv)
    print("✔ inversion")

    # sample_params
    sp = seq.sample_params(5)
    assert sp.shape == (5, 1)
    print("✔ sample_params")

    # --- NEW layer tests with two transforms ---
    transforms2 = [DummyScale(), DummyScale()]
    domains2 = [(0.5, 1.5), (0.5, 1.5)]
    seq2 = TransformSequenceDiscrete(transforms2, domains2, n_samples)
    # pick deterministic indices
    idx2 = torch.tensor([[0, 2]])  # scales: 0.5 and 1.5
    full_T = seq2(idx2)            # expects scale 0.75 (0.5*1.5)
    layer0_T = seq2(idx2, layer=0) # expects scale 0.5
    layer1_T = seq2(idx2, layer=1) # same as full (since 2 transforms)
    s_full = 0.5 * 1.5
    expect_full = torch.diag(torch.tensor([s_full, s_full, 1.0]))
    expect_l0 = torch.diag(torch.tensor([0.5, 0.5, 1.0]))
    assert torch.allclose(full_T[0], expect_full)
    assert torch.allclose(layer0_T[0], expect_l0)
    assert torch.allclose(layer1_T[0], expect_full)
    try:
        _ = seq2(idx2, layer=5)
        raise AssertionError("Out-of-range layer did not raise.")
    except ValueError:
        pass
    print("✔ layered execution")

    # QUICK test transform_with_extend negative / oversized
    base_x = torch.randn(2, 3, 8, 8)
    idx_core = torch.tensor([[1], [0]])  # within core
    out_core = seq.transform_with_extend(base_x, idx_core, extend=0)
    idx_ext = torch.tensor([[-1], [3]])  # outside core; require extend
    out_ext = seq.transform_with_extend(base_x, idx_ext, extend=1)  # should not crash
    assert out_ext.shape == out_core.shape
    print("✔ transform_with_extend")

    print("\nAll tests passed.")

if __name__ == "__main__":
    main()