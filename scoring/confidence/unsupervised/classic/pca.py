import torch
from typing import Optional, Callable
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase


class PCATorchConfidence(ClassicConfidenceBase):
    def __init__(
        self,
        n_components: Optional[int] = None,
        map_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        vim_scaling: bool = False,
        square: bool = False,
        use_residual: bool = False,
        input_transform: Optional[InputTransform] = None,
    ):
        super().__init__(input_transform=input_transform)
        self.n_components = n_components
        self.map_fn = map_function or (lambda err: 1.0 / (1.0 + err))
        self.vim_scaling = vim_scaling
        self.square = square
        self.use_residual = use_residual  # NEW
        self.mean_: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self.alpha: Optional[float] = None
        self.fitted = False

    def _fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> "PCATorchConfidence":
        N, D = X.shape
        if self.n_components is None:
            self.n_components = max(D // 2, 1)
        if not (0 < self.n_components < D):
            raise ValueError(f"n_components must be in [1, {D-1}], got {self.n_components}")

        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)
        self.components_ = eigvecs[:, idx][:, : self.n_components].T  # shape: [n_components, D]
        self.fitted = True

        if self.vim_scaling:
            flat = X - self.mean_
            if self.use_residual:
                proj = flat @ self.components_.T @ self.components_
                resid = flat - proj
            else:
                recon = (flat @ self.components_.T) @ self.components_ + self.mean_
                resid = flat - recon

            err = resid.pow(2).sum(dim=1) if self.square else resid.norm(dim=1)
            self.alpha = 1.0 / (err.mean().item() + 1e-12)

        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Call fit() before forward()")

        flat = x.reshape(-1, x.size(-1)) - self.mean_

        # Distance from PCA subspace (residual)
        if self.use_residual:
            proj = flat @ self.components_.T @ self.components_
            resid = flat - proj
        else:
            # Reconstruction-based
            recon = (flat @ self.components_.T) @ self.components_ + self.mean_
            resid = flat - recon

        err = resid.pow(2).sum(dim=1) if self.square else resid.norm(dim=1)

        if self.vim_scaling:
            if self.alpha is None:
                raise RuntimeError("alpha not set; call fit() first")
            err = err * self.alpha

        return self.map_fn(err).reshape(*x.shape[:-1])
if __name__ == '__main__':
    import torch
    from torch import nn
    from confidence.unsupervised.classic.VIM import ViMTorchConfidence

    # synthetic data
    N, D, C = 100, 16, 4
    X = torch.randn(N, D)
    y = torch.randint(0, C, (N,))

    # PCA reconstruction-based
    pca_recon = PCATorchConfidence(n_components=D // 2, map_function=lambda err: -err, use_residual=False)
    pca_recon.fit(X, y)
    scores_recon = pca_recon(X)

    # PCA residual-distance-based
    pca_resid = PCATorchConfidence(n_components=D // 2, map_function=lambda err: -err, use_residual=True)
    pca_resid.fit(X, y)
    scores_resid = pca_resid(X)

    # ViM baseline
    model = nn.Sequential(nn.Linear(D, C))
    vim = ViMTorchConfidence(model=model, n_dim=D // 2, use_energy=False)
    vim.fit(X, y)
    scores_vim = vim(X)

    # Compare
    print("Reconstruction PCA scores:", scores_recon)
    print("Residual-based PCA scores:", scores_resid)
    print("ViM scores:", scores_vim)