import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import MinCovDet
from typing import Union, Literal, Optional, Tuple


class InputTransform(nn.Module):
    def __init__(
            self,
            standardize: bool = False,
            whiten: bool = False,
            robust_cov: bool = False,
            eps: float = 1e-8
    ):
        super().__init__()
        self.standardize = standardize or robust_cov
        self.whiten = whiten
        self.robust_cov = robust_cov
        self.eps = eps

        # register buffers so they move with .to(device)
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        self.register_buffer('cov_mean', None)
        self.register_buffer('whitening_matrix', None)

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        # Make data 2D: [N, D] while remembering original prefix shape
        if isinstance(data, torch.Tensor):
            orig_shape = data.shape
            if data.dim() >= 2:
                data2d = data.reshape(-1, data.shape[-1])
            else:
                data2d = data.view(-1, 1)
            # --- torch branch ---
            if self.standardize:
                m = data2d.mean(dim=0)
                s = data2d.std(dim=0)
                s = torch.where(s < self.eps, torch.ones_like(s), s)
                self.mean, self.std = m, s
                data2d = (data2d - m) / s
            if self.whiten:
                arr = data2d.cpu().numpy()
                if self.robust_cov:
                    mcd = MinCovDet().fit(arr)
                    cov = torch.from_numpy(mcd.covariance_).to(data2d.dtype)
                    cm = torch.from_numpy(mcd.location_).to(data2d.dtype)
                else:
                    cm = data2d.mean(dim=0)
                    cov = torch.from_numpy(np.cov(arr, rowvar=False)).to(data2d.dtype)
                eigv, eigvec = torch.linalg.eigh(cov)
                inv_sqrt = eigv.clamp(min=self.eps).rsqrt()
                W = eigvec @ torch.diag(inv_sqrt) @ eigvec.T
                self.cov_mean, self.whitening_matrix = cm, W
            return  # no need to return transformed during fit
        else:
            # numpy branch
            orig_shape = data.shape
            if data.ndim >= 2:
                data2d = data.reshape(-1, data.shape[-1])
            else:
                data2d = data.reshape(-1, 1)
            if self.standardize:
                m = data2d.mean(axis=0)
                s = data2d.std(axis=0)
                s[s < self.eps] = 1.0
                data2d = (data2d - m) / s
                self.mean = torch.from_numpy(m.astype(np.float32))
                self.std = torch.from_numpy(s.astype(np.float32))
            if self.whiten:
                if self.robust_cov:
                    mcd = MinCovDet().fit(data2d)
                    cov = mcd.covariance_
                    cm = mcd.location_
                else:
                    cm = data2d.mean(axis=0)
                    cov = np.cov(data2d, rowvar=False)
                eigv, eigvec = np.linalg.eigh(cov)
                inv_sqrt = 1.0 / np.sqrt(np.maximum(eigv, self.eps))
                W = eigvec @ np.diag(inv_sqrt) @ eigvec.T
                self.cov_mean = torch.from_numpy(cm.astype(np.float32))
                self.whitening_matrix = torch.from_numpy(W.astype(np.float32))
            return  # no need to return transformed during fit

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            x = data
            if self.standardize and self.mean is not None:
                x = (x - self.mean) / self.std
            if self.whiten and self.whitening_matrix is not None:
                x = (x - self.cov_mean) @ self.whitening_matrix
            return x
        else:
            x = data
            if self.standardize and self.mean is not None:
                m = self.mean.numpy()
                s = self.std.numpy()
                x = (x - m) / s
            if self.whiten and self.whitening_matrix is not None:
                cm = self.cov_mean.numpy()
                W = self.whitening_matrix.numpy()
                x = (x - cm) @ W
            return x

    def forward(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        return self.transform(data)


class InputTransformImage(nn.Module):
    def __init__(self, reduce_dims=(2, 2),reshape_image_shape=None,average_pool=True,
                 rp_dim: Optional[int] = None, rp_method: str = "gaussian", rp_seed: Optional[int] = None,
                 rp_normalize_rows: bool = True):
        super().__init__()
        self.reduce_dims = reduce_dims
        self.avg_pool = nn.AdaptiveAvgPool2d(reduce_dims) if average_pool else nn.AdaptiveMaxPool2d(reduce_dims)
        self.reshape_image_shape = reshape_image_shape

        # Random projection optional config
        self.rp = None
        self.rp_dim = rp_dim
        # store constructor metadata so we can reconstruct on load
        self.rp_method = rp_method
        self.rp_seed = rp_seed
        self.rp_normalize_rows = rp_normalize_rows
        self._rp_skipped = False  # whether RP was intentionally skipped because pooled dim <= rp_dim
        if rp_dim is not None:
            # create RP submodule (unfitted) so its state is carried in state_dict()
            self.rp = RandomProjectionModule(
                n_components=rp_dim,
                method=rp_method,
                seed=rp_seed,
                normalize_rows=rp_normalize_rows,
            )

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        was_tensor = torch.is_tensor(data)
        # reshape image if requested
        if self.reshape_image_shape is not None:
            if was_tensor:
                data = data.view(-1, *self.reshape_image_shape)
            else:
                data = data.reshape(-1, *self.reshape_image_shape)

        # convert numpy -> torch for pooling
        if not was_tensor:
            data_t = torch.from_numpy(data.astype(np.float32))
        else:
            data_t = data

        pooled = self.avg_pool(data_t).flatten(start_dim=1)  # [B, F]
        pooled_dim = int(pooled.shape[1])

        # apply random projection only if pooled_dim > requested rp_dim
        if self.rp is not None and pooled_dim > int(self.rp.n_components):
            # auto-fit RP if not fitted yet (convenience)
            # keep output type consistent with input
            if was_tensor:
                out = self.rp.transform(pooled.float())
                return out
            else:
                out = self.rp.transform(pooled.detach().cpu().numpy())
                return out
        else:
            # RP not configured or pooled dim too small -> skip RP
            return pooled if was_tensor else pooled.detach().cpu().numpy()

    def forward(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        return self.transform(data)

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Fit the internal random projection (if configured) using pooled features
        computed from the provided image data. RP is fitted only when pooled_dim > rp_dim.
        """
        if self.reshape_image_shape is not None:
            if isinstance(data, torch.Tensor):
                data = data.view(-1, *self.reshape_image_shape)
            else:
                data = data.reshape(-1, *self.reshape_image_shape)

        # convert to torch for pooling
        if isinstance(data, torch.Tensor):
            data_t = data
        else:
            data_t = torch.from_numpy(data.astype(np.float32))

        pooled = self.avg_pool(data_t).flatten(start_dim=1)  # [N, F]
        pooled_dim = int(pooled.shape[1])

        if self.rp is not None:
            if pooled_dim > int(self.rp.n_components):
                # let RandomProjectionModule decide device/dtype from pooled
                self.rp.fit(pooled)
                self._rp_skipped = False
            else:
                # pooled dim is not larger than requested rp_dim -> skip RP
                self._rp_skipped = True
        return self

    # NEW: reconstruct RP submodule if saved weights are present
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Ensure that an RP submodule exists before delegating to super().load_state_dict.
        This allows saved state with keys like 'rp.proj.weight' to be loaded even if the
        current instance was constructed without rp kwargs.
        """
        # If saved state contains rp.* keys and self.rp is missing, reconstruct a placeholder module
        if any(k.startswith("rp.") for k in state_dict.keys()) and self.rp is not None:
            # try to infer output dim from saved weight (most likely at 'rp.proj.weight')
            self.rp.load_state_dict({k[3:]: v for k, v in state_dict.items() if k.startswith("rp.")}, strict=False)

        # delegate to base loader (be permissive so missing keys don't error)
        res = super().load_state_dict(state_dict, strict=False)
        # update skip flag if rp exists but not fitted
        if self.rp is not None:
            self._rp_skipped = not getattr(self.rp, "_fitted", False)
        return res


class PCAInputModule(nn.Module):
    def __init__(self, n_components: int = 2, keep_first: bool = True, backend: str = "sklearn"):
        super().__init__()
        self.n_components = n_components
        self.keep_first = keep_first
        self.backend = backend
        # always start with an empty buffer so state_dict() works
        self.register_buffer("proj_matrix", torch.empty(0))
        self._fitted = False

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None, device=None) -> "PCAInputModule":
        # Flatten everything except batch -> [B, F]
        if torch.is_tensor(data):
            x = data.detach()
            if x.dim() == 0:
                raise ValueError("PCAInputModule.fit: scalar input not supported")
            if x.dim() == 1:
                arr2d = x.view(1, -1)
            else:
                arr2d = x.contiguous().view(x.shape[0], -1)
            arr = arr2d.cpu().numpy() if self.backend == "sklearn" else arr2d
        else:
            x = data
            if x.ndim == 0:
                raise ValueError("PCAInputModule.fit: scalar input not supported")
            if x.ndim == 1:
                arr = x.reshape(1, -1)
            else:
                arr = x.reshape(x.shape[0], -1)

        n_features = arr.shape[1]

        if self.backend == "sklearn":
            from sklearn.decomposition import PCA
            pca = PCA(
                n_components=self.n_components if self.keep_first else None,
                svd_solver="auto"
            )
            pca.fit(arr)
            comps = pca.components_
        else:
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(arr, full_matrices=False)
            comps = Vh.cpu().numpy()

        # Select components
        if self.keep_first:
            selected = comps[: self.n_components]
        else:
            selected = comps[self.n_components: self.n_components * 2]

        device = device or (data.device if torch.is_tensor(data) else "cpu")
        pm = torch.tensor(selected, dtype=torch.float32, device=device)

        # ✅ update buffer in-place instead of overwriting
        self.proj_matrix.resize_as_(pm).copy_(pm)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten everything except batch; output [B, C] or [C] for 1D input
        if self.proj_matrix.numel() == 0:
            raise RuntimeError("PCAInputModule is not fitted. Call fit() first.")
        if x.dim() == 0:
            raise ValueError("PCAInputModule.forward: scalar input not supported")
        if x.dim() == 1:
            x2d = x.view(1, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj_matrix.shape[1]):
                raise ValueError(f"PCAInputModule.forward: input features {D_in} != fitted {int(self.proj_matrix.shape[1])}")
            out2d = x2d @ self.proj_matrix.T
            return out2d.view(-1)
        else:
            B = x.shape[0]
            x2d = x.contiguous().view(B, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj_matrix.shape[1]):
                raise ValueError(f"PCAInputModule.forward: input features {D_in} != fitted {int(self.proj_matrix.shape[1])}")
            out2d = x2d @ self.proj_matrix.T
            return out2d

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None):
        if torch.is_tensor(data):
            return self.forward(data.float())
        else:
            x = torch.from_numpy(data.astype(np.float32))
            out = self.forward(x).detach().cpu().numpy()
            return out

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "proj_matrix" in state_dict:
            ckpt_pm = state_dict["proj_matrix"]
            if self.proj_matrix.shape != ckpt_pm.shape:
                # re-register the buffer with correct shape
                self.register_buffer("proj_matrix", ckpt_pm.clone())
                # remove from state_dict so super() doesn't try again
                del state_dict["proj_matrix"]

        super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._fitted = self.proj_matrix.numel() > 0


import torch
import torch.nn as nn
from torch_pca import PCA  # differentiable PCA

class PCAInputModuleTorch(nn.Module):
    def __init__(self, n_components: int = 2, whiten=False):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.pca: Optional[PCA] = None
        self._fitted = False

    def fit(self, data: torch.Tensor, y=None) -> "PCAInputModuleTorch":
        if data.dim() == 1:
            data = data.unsqueeze(0)
        else:
            data = data.view(data.shape[0], -1)

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(data)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("PCAInputModuleTorch is not fitted. Call fit() first.")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.shape[0], -1)
        return self.pca.transform(x)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self._fitted and self.pca is not None:
            # flatten PCA params into the dict
            sd["pca.components_"] = self.pca.components_
            sd["pca.mean_"] = self.pca.mean_
            sd["pca.explained_variance_"] = self.pca.explained_variance_
            sd["pca.explained_variance_ratio_"] = self.pca.explained_variance_ratio_
            sd["pca.singular_values_"] = self.pca.singular_values_
            sd["pca.noise_variance_"] = self.pca.noise_variance_
            sd["pca.n_components_"] = torch.tensor(self.pca.n_components_)
            sd["pca.n_samples_"] = torch.tensor(self.pca.n_samples_)
            sd["pca.n_features_in_"] = torch.tensor(self.pca.n_features_in_)
            sd["pca.whiten"] = torch.tensor(int(self.pca.whiten))
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # Extract PCA params
        pca_keys = [k for k in state_dict if k.startswith("pca.")]
        if pca_keys:
            self.pca = PCA(
                n_components=int(state_dict["pca.n_components_"].item()),
                whiten=bool(state_dict["pca.whiten"].item()),
            )
            self.pca.components_ = state_dict["pca.components_"]
            self.pca.mean_ = state_dict["pca.mean_"]
            self.pca.explained_variance_ = state_dict["pca.explained_variance_"]
            self.pca.explained_variance_ratio_ = state_dict["pca.explained_variance_ratio_"]
            self.pca.singular_values_ = state_dict["pca.singular_values_"]
            self.pca.noise_variance_ = state_dict["pca.noise_variance_"]
            self.pca.n_components_ = int(state_dict["pca.n_components_"].item())
            self.pca.n_samples_ = int(state_dict["pca.n_samples_"].item())
            self.pca.n_features_in_ = int(state_dict["pca.n_features_in_"].item())
            self._fitted = True
            for k in pca_keys:
                state_dict.pop(k)

        super().load_state_dict(state_dict, strict)

    def to(self, *args, **kwargs):
        super().to( *args, **kwargs)
        if self.pca is not None:
            self.pca.to( *args, **kwargs)
        return self



class TokenPooling(nn.Module):
    """
    Pool token representations to a single vector per sample.

    Methods:
        - 'mean': average all patch tokens (exclude CLS token)
        - 'max': max over patch tokens (exclude CLS token)
        - 'cls': return the CLS token only
        - 'cls+mean': concatenate CLS token + mean of patch tokens
        - 'cls+max': concatenate CLS token + max of patch tokens
        - integer string: take every nth token (for subsampling patch tokens)

    Args:
        method (str): pooling method
        attn_hidden (Optional[int]): hidden dim for learned attention (not implemented here)
    """

    def __init__(self, method: str = "mean", attn_hidden: Optional[int] = None):
        super().__init__()
        self.method = method
        self.attn_hidden = attn_hidden
        # Placeholder for attention-based pooling if needed in future
        # e.g., self.attn = nn.Sequential(nn.Linear(D, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1))

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D] or [L, D] (L = number of tokens including CLS)
        """
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)

        cls_token = x[:, 0:1, :]  # CLS token
        patch_tokens = x[:, 1:, :]  # All patch tokens

        if self.method == "mean":
            out = patch_tokens.mean(dim=1)
        elif self.method == "max":
            out, _ = patch_tokens.max(dim=1)
        elif self.method == "cls":
            out = cls_token.squeeze(1)
        elif self.method == "cls+mean":
            patch_mean = patch_tokens.mean(dim=1)
            out = torch.cat([cls_token.squeeze(1), patch_mean], dim=1)
        elif self.method == "cls+max":
            patch_max, _ = patch_tokens.max(dim=1)
            out = torch.cat([cls_token.squeeze(1), patch_max], dim=1)
        elif self.method.isdigit():
            n = int(self.method)
            out = patch_tokens[:, ::n, :]
            out = out.flatten(1)  # flatten token dim if needed
        else:
            raise ValueError(f"Unknown pooling method {self.method}")

        return out if batched else out.squeeze(0)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class ConvPatchReducer(nn.Module):
    """
    Reduce patch embeddings by reshaping tokens to [B, C, H, W] and optionally pooling or resizing.

    Args:
        patch_grid: original patch grid (H, W) used to reshape tokens.
        pool: 'avg'|'none' - if 'avg' uses AdaptiveAvgPool2d to reduce spatial dims.
        out_grid: optional target spatial output (H', W'). If provided:
            - pool=='avg' -> AdaptiveAvgPool2d(out_grid)
            - pool=='none' -> F.interpolate to out_grid
        cls_first: if True the first token is a CLS token and will be separated.
        keep_cls: if True the CLS token is preserved in output as the first token.
        flatten: if True return flattened tokens [B, N, C']; if False return spatial map [B, C, H', W'].
    """

    def __init__(
        self,
        patch_grid: Tuple[int, int],
        pool: str = "avg",
        out_grid: Optional[Tuple[int, int]] = None,
        cls_first: bool = True,
        keep_cls: bool = True,
        flatten: bool = True,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.pool = pool.lower()
        self.out_grid = out_grid
        self.cls_first = cls_first
        self.keep_cls = keep_cls
        self.flatten = flatten

    def forward(self, x: torch.Tensor):
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        B, L, D = x.shape
        H, W = self.patch_grid
        expected = H * W + (1 if self.cls_first else 0)
        if L != expected:
            raise ValueError(f"Expected L={expected} tokens (H*W + cls?), got {L}")

        # Separate CLS token
        cls_token = None
        if self.cls_first:
            cls_token = x[:, 0:1, :]
            x = x[:, 1:, :]
            B, L, D = x.shape

        # reshape patch tokens -> [B, D, H, W]
        x2 = x.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # resize / pool if requested
        if self.out_grid is not None:
            if self.pool == "avg":
                x2 = F.adaptive_avg_pool2d(x2, self.out_grid)
            else:
                x2 = F.interpolate(x2, size=self.out_grid, mode="bilinear", align_corners=False)
        else:
            if self.pool == "avg":
                x2 = F.adaptive_avg_pool2d(x2, (1, 1))

        if self.flatten:
            # flatten spatial map to tokens [B, N, C]
            Bc, Cc, Hc, Wc = x2.shape
            tokens = x2.permute(0, 2, 3, 1).reshape(Bc, Hc * Wc, Cc)
            if self.cls_first and self.keep_cls:
                cls_token = cls_token.to(tokens.dtype)
                tokens = torch.cat([cls_token, tokens], dim=1)
            out = tokens
        else:
            # keep spatial map [B, C, H, W]
            if self.cls_first and self.keep_cls:
                # prepend CLS token as an extra spatial position (H=W=1)
                cls_map = cls_token.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
                out = torch.cat([cls_map, x2], dim=-1)  # concatenate along W dimension
            else:
                out = x2

        return out.squeeze(0) if not batched else out


class L2Normalization(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + self.eps)


import math
from typing import Union
import numpy as np
import torch
import torch.nn as nn


class RandomProjectionModule(nn.Module):
    """
    Random linear projection using dense Gaussian or orthogonal random matrices.
    """
    def __init__(
        self,
        n_components: int,
        method: Literal["gaussian", "orthogonal"] = "gaussian",
        seed: int = None,
        normalize_rows: bool = True,
    ):
        super().__init__()
        self.n_components = n_components
        self.method = method.lower()
        self.seed = seed
        self.normalize_rows = normalize_rows
        self.proj = None
        self._fitted = False

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> "RandomProjectionModule":
        if isinstance(data, torch.Tensor):
            x = data
            if x.dim() == 0:
                raise ValueError("RandomProjectionModule.fit: scalar input not supported")
            if x.dim() == 1:
                data2d = x.view(1, -1)
            else:
                data2d = x.contiguous().view(x.shape[0], -1)
            n_features = int(data2d.shape[1])
            device = data2d.device
            dtype = data2d.dtype
        else:
            x = data.reshape(data.shape[0], -1) if data.ndim > 1 else data.reshape(1, -1)
            n_features = int(x.shape[1])
            device = torch.device("cpu")
            dtype = torch.float32

        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.method == "gaussian":
            W = torch.randn(self.n_components, n_features, device=device, dtype=dtype)
            W /= math.sqrt(self.n_components)
        elif self.method == "orthogonal":
            M = torch.randn(max(self.n_components, n_features), n_features, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(M, mode='reduced')
            W = Q[:self.n_components] * math.sqrt(n_features / self.n_components)
        else:
            raise ValueError(f"Unknown method '{self.method}'")

        if self.normalize_rows:
            norms = W.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            W = W / norms

        self.proj = nn.Linear(n_features, self.n_components, bias=False).to(device=device, dtype=dtype)
        self.proj.weight.data = W
        self.proj.weight.requires_grad = False
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten everything except batch; output [B, C] or [C] for 1D input
        if self.proj is None:
            raise RuntimeError("RandomProjectionModule not fitted. Call fit() first.")
        if x.dim() == 0:
            raise ValueError("RandomProjectionModule.forward: scalar input not supported")
        if x.dim() == 1:
            x2d = x.view(1, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj.weight.shape[1]):
                raise ValueError(f"RandomProjectionModule.forward: input features {D_in} != fitted {int(self.proj.weight.shape[1])}")
            out2d = self.proj(x2d)
            return out2d.view(-1)
        else:
            B = x.shape[0]
            x2d = x.contiguous().view(B, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj.weight.shape[1]):
                raise ValueError(f"RandomProjectionModule.forward: input features {D_in} != fitted {int(self.proj.weight.shape[1])}")
            out2d = self.proj(x2d)
            return out2d

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return self.forward(data.float())
        else:
            x = torch.from_numpy(data.astype(np.float32))
            out = self.forward(x).detach().cpu().numpy()
            return out

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Make loading robust even if proj isn't created yet by reconstructing it
        from the incoming 'proj.weight'.
        """
        # state_dict may be Tensors (e.g., from safetensors)
        w = state_dict.get("proj.weight", None)
        if w is not None:
            # Ensure plain Tensor
            if isinstance(w, torch.nn.Parameter):
                w = w.data
            n_out, n_in = int(w.shape[0]), int(w.shape[1])
            # Create proj if missing or mismatched
            if self.proj is None or self.proj.weight.shape != w.shape:
                self.proj = nn.Linear(n_in, n_out, bias=False)
                self.proj.weight.requires_grad = False
            # Keep metadata consistent
            self.n_components = n_out
        result = super().load_state_dict(state_dict, strict=False)  # be permissive
        self._fitted = self.proj is not None
        return result


import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from typing import Union


class SparseRandomProjectionModule(nn.Module):
    """
    Memory-efficient random projection using sparse random matrices.
    Uses CSR format for efficient storage and computation.
    """

    def __init__(
        self,
        n_components: int,
        density: Union[float, str] = 'auto',
        seed: int = None
    ):
        super().__init__()
        self.n_components = n_components
        self.density = density
        self.seed = seed

        # Initialize as None so state_dict can load real tensors without shape mismatch
        self.register_buffer('sparse_weight_values', torch.tensor([], dtype=torch.float32), persistent=True)
        self.register_buffer('sparse_weight_indices', torch.empty(2, 0, dtype=torch.long), persistent=True)
        self.register_buffer('sparse_weight_shape', torch.empty(2, dtype=torch.long), persistent=True)

        self._fitted = False

    def _make_sparse_random_matrix(
        self,
        n_components: int,
        n_features: int,
        density: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Expected number of non-zeros
        nnz = int(density * n_components * n_features)
        s = 1.0 / density

        # Random positions
        row_indices = np.random.randint(0, n_components, size=nnz)
        col_indices = np.random.randint(0, n_features, size=nnz)

        # Random values ±sqrt(s)/sqrt(n_components)
        values = np.random.choice(
            [-np.sqrt(s) / np.sqrt(n_components), np.sqrt(s) / np.sqrt(n_components)],
            size=nnz
        )

        # Build sparse matrix
        components = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_components, n_features),
            dtype=np.float32
        )
        components.eliminate_zeros()

        coo = components.tocoo()

        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data.copy()).float()

        return indices.to(device), values.to(device), torch.Size([n_components, n_features])

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> "SparseRandomProjectionModule":
        if isinstance(data, torch.Tensor):
            x = data.view(data.shape[0], -1) if data.dim() > 1 else data.view(1, -1)
            n_features = x.shape[1]
            device = x.device
            dtype = x.dtype
        else:
            x = data.reshape(data.shape[0], -1) if data.ndim > 1 else data.reshape(1, -1)
            n_features = x.shape[1]
            device = torch.device("cpu")
            dtype = torch.float32

        # Auto density from sklearn: 1 / sqrt(n_features)
        if self.density == 'auto':
            density = min(1.0, 1.0 / np.sqrt(n_features))
        else:
            density = float(self.density)

        indices, values, shape = self._make_sparse_random_matrix(
            self.n_components, n_features, density, dtype, device
        )

        # Update buffers safely
        self.sparse_weight_indices = indices
        self.sparse_weight_values = values
        self.sparse_weight_shape = torch.tensor(list(shape), dtype=torch.long, device=device)

        self._fitted = True
        return self

    def _get_sparse_weight(self, device=None):
        if not self._fitted or self.sparse_weight_indices is None:
            raise RuntimeError("SparseRandomProjectionModule not fitted.")
        indices = self.sparse_weight_indices.to(device)
        values = self.sparse_weight_values.to(device)
        shape = tuple(self.sparse_weight_shape.tolist())
        return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("SparseRandomProjectionModule not fitted. Call fit() first.")
        if x.dim() == 0:
            raise ValueError("SparseRandomProjectionModule.forward: scalar input not supported")

        x_2d = x.view(-1, int(self.sparse_weight_shape[1]))
        sparse_weight = self._get_sparse_weight(device=x.device)
        return torch.sparse.mm(sparse_weight, x_2d.t()).t()

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return self.forward(data.float())
        else:
            x = torch.from_numpy(data.astype(np.float32))
            return self.forward(x).detach().cpu().numpy()

    def load_state_dict(self, state_dict, strict: bool = True):
        # Replace buffers directly with checkpoint values (avoid size mismatch)
        for name in ["sparse_weight_values", "sparse_weight_indices", "sparse_weight_shape"]:
            if name in state_dict:
                tensor = state_dict[name]
                self.register_buffer(name, tensor)
        # Call parent loader (looser check for buffers)
        result = super().load_state_dict(state_dict, strict=False)
        # Mark as fitted if buffers are present
        if (self.sparse_weight_indices is not None
                and self.sparse_weight_values is not None
                and self.sparse_weight_shape is not None):
            self._fitted = True
        return result


def calc_auto_density(n_features: int,mult=1.0) -> float:
    """
    Calculate 'auto' density for SparseRandomProjection as 1/sqrt(n_features).
    """
    return min(1.0, 1.0 / np.sqrt(n_features) * mult)


class PointCloudGlobalPooling(nn.Module):
    """
    Global pooling for point cloud features from SA modules.
    Pools over the point dimension to create a single feature vector per batch.

    Args:
        pool_method: 'max' or 'mean' pooling over points
        keep_batch_dim: if True, preserve batch dimension [B, F]; if False flatten to [F]
    """

    def __init__(self, pool_method: str = "max", keep_batch_dim: bool = True):
        super().__init__()
        self.pool_method = pool_method.lower()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [N, F] point features where N is total points across batch
        batch: [N] batch assignment for each point (optional, assumes single batch if None)
        """
        if batch is None:
            # Single batch case - pool over all points
            if self.pool_method == "max":
                pooled = x.max(dim=0)[0]  # [F]
            elif self.pool_method == "mean":
                pooled = x.mean(dim=0)    # [F]
            else:
                raise ValueError(f"Unknown pool method: {self.pool_method}")

            if self.keep_batch_dim:
                pooled = pooled.unsqueeze(0)  # [1, F]
            return pooled
        else:
            # Multiple batch case - pool per batch
            try:
                from torch_geometric.nn import global_max_pool, global_mean_pool
                if self.pool_method == "max":
                    pooled = global_max_pool(x, batch)  # [B, F]
                elif self.pool_method == "mean":
                    pooled = global_mean_pool(x, batch)  # [B, F]
                else:
                    raise ValueError(f"Unknown pool method: {self.pool_method}")
            except ImportError:
                # Fallback implementation if torch_geometric not available
                unique_batch = batch.unique()
                batch_results = []
                for b in unique_batch:
                    mask = batch == b
                    batch_points = x[mask]  # [N_b, F]
                    if self.pool_method == "max":
                        batch_pooled = batch_points.max(dim=0)[0]
                    else:  # mean
                        batch_pooled = batch_points.mean(dim=0)
                    batch_results.append(batch_pooled)
                pooled = torch.stack(batch_results, dim=0)  # [B, F]

            if not self.keep_batch_dim and pooled.shape[0] == 1:
                pooled = pooled.squeeze(0)  # [F]
            return pooled

    def transform(self, data: Union[np.ndarray, torch.Tensor], batch=None) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        return self.forward(data, batch)

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> "PointCloudGlobalPooling":
        # No fitting needed for pooling
        return self



# ---------------------------------------------------------------------
# NEW: Simple collapse reducer for Mahalanobis / OOD detection
# ---------------------------------------------------------------------
class InputTransformCollapse(nn.Module):
    """
    Collapse the spatial H,W dimensions to a single point per channel.
    - No constructor parameters.
    - For input shapes:
        - [B, C, H, W] -> returns [B, C] (global average over H,W)
        - [C, H, W] -> returns [C]
        - [B, F] or [F] -> returned unchanged (already collapsed)
    - Supports torch.Tensor and numpy.ndarray inputs.
    - fit() is a no-op to match reducer API.
    """
    def __init__(self):
        super().__init__()

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> "InputTransformCollapse":
        # No fitting required for collapse reducer
        return self

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            x = data
            if x.dim() == 4:
                # [B, C, H, W] -> [B, C]
                return x.mean(dim=(2, 3))
            elif x.dim() == 3:
                # [C, H, W] -> [C]
                return x.mean(dim=(1, 2))
            else:
                # already collapsed or vector; return as-is
                return x
        else:
            x = data
            if x.ndim == 4:
                # numpy [B, C, H, W]
                return x.mean(axis=(2, 3))
            elif x.ndim == 3:
                # numpy [C, H, W]
                return x.mean(axis=(1, 2))
            else:
                return x

    def forward(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        return self.transform(data, y)


# Registry and factory for feature reducers (NEW)
FEATURE_REDUCER_REGISTRY = {
    "pca": PCAInputModule,
    "rp": RandomProjectionModule,
    "sparse_rp": SparseRandomProjectionModule,
    "conv_patch": ConvPatchReducer,
    "token_pool": TokenPooling,
    "image_transform": InputTransformImage,
    "collapse_image": InputTransformCollapse,   # << added key for simple collapse reducer
    "pca_torch": PCAInputModuleTorch,
    "pointcloud_pool": PointCloudGlobalPooling,
}

def create_feature_reducer(name: str, **kwargs) -> nn.Module:
    """
    Create a feature reducer by name. The reducer must provide:
      - fit(data: 2D torch.Tensor) -> self
      - forward(x: 2D torch.Tensor) -> 2D torch.Tensor
    """
    if name is None:
        return None
    #check if name is a torch nn module class
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name(**kwargs)

    key = name.lower()
    if key not in FEATURE_REDUCER_REGISTRY:
        raise ValueError(f"Unknown feature reducer '{name}'. Available: {list(FEATURE_REDUCER_REGISTRY)}")
    cls = FEATURE_REDUCER_REGISTRY[key]
    return cls(**kwargs)


if __name__ == "__main__":
    import numpy as np
    import torch
    from sklearn.random_projection import SparseRandomProjection
    import time

    # Test settings - use larger matrix to show sparsity benefits
    n_samples = 1000
    n_features = 50000  # Increased for better sparsity demonstration
    n_components = 100
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def run_tests():
        print("Testing SparseRandomProjectionModule...")
        print(f"Matrix size: {n_samples}x{n_features} -> {n_components}")

        # Generate random data
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        X_torch = torch.from_numpy(X).to(device)

        # Test 1: Basic initialization and fitting
        print("\nTest 1: Basic initialization and fitting")
        srp = SparseRandomProjectionModule(n_components=n_components)
        try:
            srp.fit(X_torch)
            print("✓ Fitting successful")

            # Check actual sparsity of projection matrix
            sparse_weight = srp._get_sparse_weight()
            total_elements = sparse_weight.shape[0] * sparse_weight.shape[1]
            nonzero_elements = sparse_weight._nnz()
            matrix_sparsity = nonzero_elements / total_elements
            print(f"Projection matrix sparsity: {matrix_sparsity:.4f} ({nonzero_elements}/{total_elements})")
        except Exception as e:
            print(f"✗ Fitting failed: {e}")

        # Test 2: Compare with sklearn implementation
        print("\nTest 2: Compare with sklearn implementation")
        sklearn_srp = SparseRandomProjection(n_components=n_components, random_state=42)
        srp = SparseRandomProjectionModule(n_components=n_components, seed=42)

        X_sklearn = sklearn_srp.fit_transform(X)
        srp.fit(X_torch)
        X_torch_proj = srp.transform(X_torch).cpu().numpy()

        print(f"Sklearn output shape: {X_sklearn.shape}")
        print(f"Torch output shape: {X_torch_proj.shape}")

        # Check sklearn matrix sparsity
        sklearn_matrix = sklearn_srp.components_
        sklearn_sparsity = (sklearn_matrix != 0).mean()
        print(f"Sklearn matrix density: {sklearn_sparsity:.4f}")

        # Test 3: Memory usage comparison
        print("\nTest 3: Memory usage")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            dense_start = torch.cuda.memory_allocated()
            dense_proj = RandomProjectionModule(n_components=n_components)
            dense_proj.fit(X_torch)
            dense_memory = torch.cuda.memory_allocated() - dense_start

            torch.cuda.empty_cache()

            sparse_start = torch.cuda.memory_allocated()
            sparse_proj = SparseRandomProjectionModule(n_components=n_components)
            sparse_proj.fit(X_torch)
            sparse_memory = torch.cuda.memory_allocated() - sparse_start

            print(f"Dense memory: {dense_memory / 1024 / 1024:.2f} MB")
            print(f"Sparse memory: {sparse_memory / 1024 / 1024:.2f} MB")
            print(f"Memory reduction: {(dense_memory - sparse_memory) / dense_memory * 100:.1f}%")

        # Test 4: Different density settings
        print("\nTest 4: Different density settings")
        densities = ['auto', 0.1, 0.01, 0.001]
        for density in densities:
            try:
                srp_test = SparseRandomProjectionModule(n_components=50, density=density, seed=42)
                srp_test.fit(X_torch[:100])

                # Check matrix sparsity
                sparse_mat = srp_test._get_sparse_weight()
                matrix_density = sparse_mat._nnz() / (sparse_mat.shape[0] * sparse_mat.shape[1])

                output = srp_test.transform(X_torch[:10])
                print(f"Density {density}: matrix density {matrix_density:.4f}, output shape {output.shape}")
            except Exception as e:
                print(f"Density {density} failed: {e}")

        # Test 5: State dict saving/loading
        print("\nTest 5: State dict saving/loading")
        state = srp.state_dict()
        new_srp = SparseRandomProjectionModule(n_components=n_components)

        new_srp.load_state_dict(state)
        test_output = new_srp.transform(X_torch[:5])
        print(f"✓ State dict save/load successful, output shape: {test_output.shape}")



    try:
        run_tests()
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback

        traceback.print_exc()