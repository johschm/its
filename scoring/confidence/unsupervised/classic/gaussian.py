from typing import Optional, Callable, Union

import torch
from torch_geometric.data import DataLoader

from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase


class ImageGaussianConfidence(ClassicConfidenceBase):
    """
    “One‐class Gaussian” confidence module over D‐dimensional embeddings.
    Assumes each image embedding x ∈ ℝᵈ is drawn from 𝒩(μ, Σ), fit on “normal” data.
    At inference, computes Mahalanobis distance d(x) = sqrt((x−μ)ᵀ Σ⁻¹ (x−μ))
    and maps it to confidence via map_fn (default: 1/(1 + d)).
    """

    def __init__(
        self,
        map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reg_epsilon: float = 1e-6,
        input_transform: Optional[InputTransform] = None,
    ):
        """
        Args:
          input_transform:   (Optional) transforms_old applied to embeddings before fitting/forward.
          map_fn:            Callable(distance_tensor) → confidence_tensor. Default: 1/(1 + d).
          reg_epsilon:       ε added to diagonal of covariance for numerical stability.
        """
        super().__init__(input_transform=input_transform)
        self.reg_epsilon = reg_epsilon
        self.map_fn = map_fn or (lambda d: 1.0 / (1.0 + d))

        # Will be set during fit()
        self.mean_: Optional[torch.Tensor] = None       # shape: (D,)
        self.invcov_: Optional[torch.Tensor] = None     # shape: (D, D)
        self.fitted = False

    def _fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> "ImageGaussianConfidence":
        """
        Compute μ and Σ⁻¹ from “normal” embeddings.

        Args:
          data: either a torch.Tensor of shape (N, D), or a DataLoader yielding batches of embeddings.
        """
        # 3) Compute μ ∈ ℝᵈ
        N, D = X.shape
        mu = X.mean(dim=0)               # shape: (D,)

        # 4) Center data
        Xc = X - mu.unsqueeze(0)         # shape: (N, D)

        # 5) Compute covariance Σ = (Xcᵀ Xc) / (N − 1)
        cov = (Xc.t() @ Xc) / (N - 1)    # shape: (D, D)
        cov = cov + self.reg_epsilon * torch.eye(D, device=X.device)

        # 6) Invert Σ
        invcov = torch.linalg.inv(cov)   # shape: (D, D)

        # 7) Store μ and Σ⁻¹
        self.mean_ = mu.detach()
        self.invcov_ = invcov.detach()
        self.fitted = True
        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute confidence scores for a batch of embeddings x ∈ ℝ^{B×D}.

        Returns:
          Tensor of shape (B,), each in (0, 1]: higher → “more normal.”
        """
        if not self.fitted:
            raise RuntimeError("Call fit(...) before forward(...)")

        if x.ndim != 2:
            raise ValueError(f"Expected x ∈ ℝ^(B×D), but got shape {x.shape}")

        # 1) Optional input normalization/transform
        if self.input_transform:
            x = self.input_transform.transform(x)

        # Ensure tensors are on the same device as x
        if self.mean_.device != x.device:
            self.mean_ = self.mean_.to(x.device)
            self.invcov_ = self.invcov_.to(x.device)

        # 2) Mahalanobis distance: d_i = sqrt((x_i − μ)ᵀ Σ⁻¹ (x_i − μ))
        #    Let diff ∈ ℝ^{B×D} be (x − μ)
        diff = x - self.mean_.unsqueeze(0)           # (B, D)
        invcov = self.invcov_                        # (D, D)
        temp = diff @ invcov                         # (B, D)
        sqr  = (temp * diff).sum(dim=1)              # (B,)
        dist = torch.sqrt(sqr + 1e-12)               # (B,)

        # 3) Map distances to confidences via map_fn (monotonic decreasing)
        conf = self.map_fn(dist)                     # (B,)
        return conf