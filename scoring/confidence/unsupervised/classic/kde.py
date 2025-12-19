import torch
import numpy as np
import math
from typing import Optional, Callable, Union

from sklearn.neighbors import KernelDensity
from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform  # if you have an InputTransform implementation
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase


class KDE(torch.nn.Module):
    """
    PyTorch implementation of Kernel Density Estimation (KDE) intended to match
    sklearn.neighbors.KernelDensity’s behavior as closely as possible, using float64.
    Supports Gaussian, tophat, epanechnikov, linear, and cosine kernels.
    """
    def __init__(self, bandwidth: float = 1.0, kernel: str = "gaussian", eps: float = 1e-12):
        super().__init__()
        self.bandwidth = float(bandwidth)
        self.kernel = kernel
        self.eps = float(eps)
        self.fitted = False

    def fit(self, x: torch.Tensor):
        """
        Fit the KDE on data x.
        Expects x to be a 2D tensor of shape (N, D). Internally uses float64.
        """
        pts = x.clone().detach().to(dtype=torch.float64)
        N, D = pts.shape
        self.D = D
        self.register_buffer("data", pts)

        if self.kernel == "gaussian":
            log_norm_const = (D / 2) * math.log(2 * math.pi) + D * math.log(self.bandwidth)
            self.register_buffer("log_norm_const", torch.tensor(log_norm_const, dtype=torch.float64))
        else:
            # Volume of unit D-ball, calculated using log-gamma for stability
            log_c_d = (D / 2) * math.log(math.pi) - math.lgamma(D / 2 + 1)
            c_d = math.exp(log_c_d)
            c_d = max(c_d, 1e-8)  # Clip to avoid division by zero in high dimensions

            if self.kernel == "tophat":
                kernel_norm = 1.0 / c_d
            elif self.kernel == "epanechnikov":
                kernel_norm = (D + 2) / (2 * c_d)
            elif self.kernel == "linear":
                kernel_norm = (D + 1) / c_d
            elif self.kernel == "cosine":
                # Adjusted to match sklearn’s cosine normalization
                kernel_norm = (math.pi / 2) / c_d
            else:
                raise NotImplementedError(f"Kernel '{self.kernel}' not implemented")

            self.register_buffer("kernel_norm", torch.tensor(kernel_norm, dtype=torch.float64))

        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-density estimates for input x of shape (..., D).
        Returns a tensor of shape (...), corresponding to log density.
        """
        if not self.fitted:
            raise RuntimeError("KDE not fitted. Call fit() before forward().")

        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1)).to(dtype=torch.float64)

        d2 = torch.cdist(flat, self.data, p=2).pow(2)  # (M, N)
        N = float(self.data.size(0))
        D = self.D

        if self.kernel == "gaussian":
            logK = -0.5 * d2 / (self.bandwidth ** 2) - self.log_norm_const
            log_density = torch.logsumexp(logK, dim=1) - math.log(N)
        else:
            u = torch.sqrt(d2) / self.bandwidth  # (M, N)

            if self.kernel == "tophat":
                # Differentiable approximation using sigmoid for smooth indicator
                K = self.kernel_norm * torch.sigmoid((1 - u) * 10)

            elif self.kernel == "epanechnikov":
                # Differentiable approximation using softplus for max(0, 1 - u**2)
                K = self.kernel_norm * torch.nn.functional.softplus(1 - u**2)

            elif self.kernel == "linear":
                # Differentiable approximation using softplus for max(0, 1 - u)
                K = self.kernel_norm * torch.nn.functional.softplus(1 - u)

            elif self.kernel == "cosine":
                # Only apply cosine within |u| <= 1, zero elsewhere (already differentiable)
                K = torch.where(
                    u <= 1.0,
                    self.kernel_norm * torch.cos((math.pi / 2) * u),
                    torch.zeros_like(u),
                )

            else:
                raise NotImplementedError(f"Kernel '{self.kernel}' not implemented")

            sum_K = K.sum(dim=1)  # (M,)
            normalizer = N * (self.bandwidth ** D)

            # Differentiable log-density using eps to avoid log(0)
            log_density = torch.log(sum_K / normalizer + self.eps)

        return log_density.reshape(*orig_shape).to(dtype=x.dtype).to(device=x.device)


class KDEConfidence(ClassicConfidenceBase):
    """
    Wraps the above KDE into a confidence module. Optionally applies an InputTransform,
    and a map_function on the log-density output.
    """
    def __init__(
        self,
        bandwidth: float = 0.1,
        kernel: str = "gaussian",
        map_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform: Optional[InputTransform] = None):
        super().__init__(input_transform=input_transform)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.input_transform = input_transform
        self.map_fn = map_function or (lambda x: x)
        self.kde: Optional[KDE] = None
        self.fitted = False

    def _fit(self, x: Union[np.ndarray, torch.Tensor], y: torch.Tensor = None) -> None:
        """
        Fit the KDE on data x (optionally transform inputs first).
        x can be a NumPy array or Torch tensor of shape (N, D).
        """
        pts = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        pts = pts.to(dtype=torch.float64).contiguous()
        self.kde = KDE(self.bandwidth, self.kernel)
        self.kde.fit(pts)
        self.fitted = True


    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute confidence scores = map_fn(log-density) for each row of x.
        Input x: tensor of shape (..., D).
        Returns: tensor of shape (...) (same leading shape as x without last dim).
        """
        if not self.fitted:
            raise RuntimeError("KDEConfidence not fitted. Call fit() before forward().")

        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1))
        log_density = self.kde(flat)
        conf = self.map_fn(log_density)
        return conf.reshape(*orig_shape)


# -----------------------------------------------------
# SKLearn-based KDE & Confidence wrapper
# -----------------------------------------------------
class SKLearnKDE(torch.nn.Module):
    """
    Wrapper around sklearn.neighbors.KernelDensity to produce a Torch-friendly interface.
    """
    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.bandwidth = float(bandwidth)
        self.kernel = kernel
        self.eps = float(eps)
        self.kde: Optional[KernelDensity] = None
        self.fitted = False

    def fit(self, x: torch.Tensor):
        """
        Fit sklearn’s KernelDensity on data x (Tensor of shape (N, D)).
        Converts to numpy float64 internally.
        """
        data_np = x.detach().cpu().double().numpy()  # ensure float64
        self.kde = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            atol=0,
            rtol=0,
        )
        self.kde.fit(data_np)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log-density for x of shape (..., D). Returns shape (...).
        """
        if not self.fitted or self.kde is None:
            raise RuntimeError("SKLearnKDE not fitted. Call fit() first.")

        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1)).detach().cpu().double().numpy()
        logdens = self.kde.score_samples(flat)
        logdens = np.maximum(logdens, math.log(self.eps))
        result = torch.from_numpy(logdens).to(x.dtype).to(x.device)
        return result.reshape(*orig_shape)


class SKKDEConfidence(ClassicConfidenceBase):
    """
    Confidence wrapper for sklearn-based KDE. Optionally applies an InputTransform
    and a map_function on the log-density output.
    """
    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        map_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform: Optional[InputTransform] = None,
    ):
        super().__init__(input_transform=input_transform)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.map_fn = map_function or (lambda x: x)
        self.kde: Optional[SKLearnKDE] = None
        self.fitted = False

    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> "SKKDEConfidence":
        """
        Fit the sklearn-based KDE on x.
        """
        pts = x.to(dtype=torch.float64).contiguous()
        self.kde = SKLearnKDE(self.bandwidth, self.kernel)
        self.kde.fit(pts)
        self.fitted = True
        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute confidence = map_fn(log-density) for each row of x. Input x shape (..., D).
        Returns shape (...).
        """
        if not self.fitted:
            raise RuntimeError("SKKDEConfidence not fitted. Call fit() before forward().")

        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1))
        logdens = self.kde(flat)
        return self.map_fn(logdens).reshape(*orig_shape)


# -----------------------------------------------------
# Example/Test in main: loop over multiple kernels
# -----------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate sample 1D data: mixture of two Gaussians
    N = 200
    X1 = np.random.normal(loc=-2.0, scale=0.5, size=(N // 2, 1))
    X2 = np.random.normal(loc=2.0, scale=0.8, size=(N // 2, 1))
    X = np.vstack([X1, X2]).astype(np.float32)  # shape (200, 1)

    # Convert to Torch tensor
    X_tensor = torch.from_numpy(X)

    # Define evaluation grid
    grid_np = np.linspace(-6.0, 6.0, 200).reshape(-1, 1).astype(np.float32)
    grid_tensor = torch.from_numpy(grid_np)

    # List of kernels to test
    kernels = ["gaussian", "tophat", "epanechnikov", "linear", "cosine"]

    for kernel in kernels:
        print(f"\n=== Testing kernel: {kernel} ===")

        # ----------------------------
        # 1. PyTorch KDE
        # ----------------------------
        pt_kde = KDE(bandwidth=0.5, kernel=kernel)
        pt_kde.fit(X_tensor)
        with torch.no_grad():
            log_dens_pt = pt_kde(grid_tensor).cpu().numpy()
        dens_pt = np.exp(log_dens_pt)

        # ----------------------------
        # 2. sklearn KDE
        # ----------------------------
        sk_kde = KernelDensity(bandwidth=0.5, kernel=kernel)
        sk_kde.fit(X.astype(np.float64))
        log_dens_sk = sk_kde.score_samples(grid_np.astype(np.float64))
        dens_sk = np.exp(log_dens_sk)

        # ----------------------------
        # 3. Compare via wrapped modules
        # ----------------------------
        pt_conf = KDEConfidence(bandwidth=0.5, kernel=kernel)
        pt_conf.fit(X_tensor)
        with torch.no_grad():
            out_pt_conf = torch.exp(pt_conf(grid_tensor)).cpu().numpy()

        sk_conf = SKKDEConfidence(bandwidth=0.5, kernel=kernel)
        sk_conf.fit(X_tensor)
        with torch.no_grad():
            out_sk_conf = torch.exp(sk_conf(grid_tensor)).cpu().numpy()

        # ----------------------------
        # Print first few values and summary differences
        # ----------------------------
        print("First 5 log-density values (PyTorch KDE):")
        print(np.round(log_dens_pt[:5], 5))
        print("First 5 log-density values (sklearn KDE):")
        print(np.round(log_dens_sk[:5], 5))

        print("Max absolute difference in log-density (PyTorch vs sklearn):")
        print(np.round(np.nanmax(np.abs(log_dens_pt - log_dens_sk)), 8))

        print("Max absolute difference in density (PyTorch vs sklearn):")
        print(np.round(np.nanmax(np.abs(dens_pt - dens_sk)), 8))

        print("Max absolute difference in confidence (wrapped PyTorch vs sklearn):")
        print(np.round(np.nanmax(np.abs(out_pt_conf - out_sk_conf)), 8))

        # ----------------------------
        # Plot densities for this kernel
        # ----------------------------
        plt.figure(figsize=(6, 3))
        plt.plot(grid_np.squeeze(), dens_pt, label="PyTorch KDE", lw=2)
        plt.plot(grid_np.squeeze(), dens_sk, "--", label="sklearn KDE", lw=2)
        plt.title(f"1D KDE Density Comparison ({kernel})")

        plt.xlabel("x")
        plt.ylabel("density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()