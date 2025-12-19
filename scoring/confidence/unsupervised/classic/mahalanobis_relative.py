# Improved Mahalanobis from "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection"
import torch
import torch.nn.functional as F
from confidence.unsupervised.classic.nn_pytorch import DistanceConfidence, _remap_labels_to_indices, _compute_global_mean_inv_cov, _compute_per_class_means_inv_covs


class RelativeMahalanobisConfidence(DistanceConfidence):
    """
    Relative Mahalanobis Distance (RMD) with optional diagonal or low-rank covariance.
    RMD_k(z') = MD_k(z') - MD_0(z')
    """

    def __init__(self,
                 mahalanobis_eps: float = 1e-6,
                 shared_covariance: bool = False,
                 use_raw_scatter: bool = False,
                 cov_mode: str = "full",     # "full" | "diag" | "lowrank"
                 low_rank_r: int = 64,
                 global_cov_mode: str = None,  # None means "use same as class cov_mode"
                 **kwargs):
        super().__init__(**kwargs)
        self.mahalanobis_eps = mahalanobis_eps
        self.shared_covariance = shared_covariance
        self.use_raw_scatter = use_raw_scatter
        self.cov_mode = cov_mode
        self.low_rank_r = low_rank_r
        self.global_cov_mode = global_cov_mode

    def _fit(self, x: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            labels_unique, y_idx = _remap_labels_to_indices(y)
            self.register_buffer('y_idx', y_idx)
            self.register_buffer('labels_unique', labels_unique)
            N, D = x.size()
            C = int(self.y_idx.max().item()) + 1

            one_hot = F.one_hot(self.y_idx, num_classes=C).float()
            counts = one_hot.sum(dim=0)
            class_means = (one_hot.T @ x) / counts.unsqueeze(1)
            self.register_buffer('class_means', class_means)
            self.register_buffer('global_mean', x.mean(0))

            # Global (background) covariance
            # Use explicit global mode if provided, otherwise fall back to class cov_mode
            g_mode = self.global_cov_mode if self.global_cov_mode is not None else self.cov_mode
            global_residuals = x - self.global_mean
            if g_mode == "diag":
                var = (global_residuals ** 2).sum(0) / (N if self.use_raw_scatter else max(1, N - 1))
                var = var + self.mahalanobis_eps
                global_inv = 1.0 / var
                self.register_buffer("global_inv_diag", global_inv)
            elif g_mode == "lowrank":
                cov = (global_residuals.T @ global_residuals) / (N if self.use_raw_scatter else max(1, N - 1))
                cov = cov + torch.eye(D, device=x.device) * self.mahalanobis_eps
                r = min(self.low_rank_r, D)
                U, S, _ = torch.linalg.svd(cov, full_matrices=False)
                self.register_buffer("global_U", U[:, :r])
                self.register_buffer("global_inv_S", 1.0 / S[:r])
            else:
                _, global_inv = _compute_global_mean_inv_cov(x, self.mahalanobis_eps)
                if self.use_raw_scatter:
                    global_inv /= float(max(1, N))
                self.register_buffer("global_inv_cov", global_inv)

            # Class-specific covariances
            if self.shared_covariance:
                residuals = x - class_means[y_idx]
                if self.cov_mode == "diag":
                    var = (residuals ** 2).sum(0) / (N if self.use_raw_scatter else max(1, N - C))
                    var = var + self.mahalanobis_eps
                    inv_diag = 1.0 / var
                    self.register_buffer("class_inv_diag", inv_diag.unsqueeze(0).repeat(C, 1))
                elif self.cov_mode == "lowrank":
                    cov = (residuals.T @ residuals) / (N if self.use_raw_scatter else max(1, N - C))
                    cov = cov + torch.eye(D, device=x.device) * self.mahalanobis_eps
                    r = min(self.low_rank_r, D)
                    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
                    self.register_buffer("class_U", U[:, :r].unsqueeze(0).repeat(C, 1, 1))
                    self.register_buffer("class_inv_S", (1.0 / S[:r]).unsqueeze(0).repeat(C, 1))
                else:
                    _, pooled_inv = _compute_global_mean_inv_cov(residuals, self.mahalanobis_eps)
                    if self.use_raw_scatter:
                        pooled_inv /= float(max(1, N))
                    self.register_buffer("class_inv_covs", pooled_inv.unsqueeze(0).repeat(C, 1, 1))
            else:
                residuals = x - class_means[y_idx]
                if self.cov_mode == "diag":
                    var_num = torch.einsum('nd,nk,nd->kd', residuals, one_hot, residuals)
                    denom = counts.unsqueeze(1) if self.use_raw_scatter else (counts).clamp_min(1).unsqueeze(1)
                    var = var_num / denom
                    var = var + self.mahalanobis_eps
                    inv_diag = 1.0 / var
                    self.register_buffer("class_inv_diag", inv_diag)
                elif self.cov_mode == "lowrank":
                    cov_num = torch.einsum('nd,nk,nm->kdm', residuals, one_hot, residuals)
                    denom = counts.view(C, 1, 1) if self.use_raw_scatter else (counts).clamp_min(1).view(C, 1, 1)
                    covs = cov_num / denom
                    eye = torch.eye(D, device=x.device).unsqueeze(0)
                    covs = covs + self.mahalanobis_eps * eye
                    
                    r = min(self.low_rank_r, D)
                    U_list, inv_S_list = [], []
                    for c in range(C):
                        U, S, _ = torch.linalg.svd(covs[c], full_matrices=False)
                        U_list.append(U[:, :r])
                        inv_S_list.append(1.0 / S[:r])
                    self.register_buffer("class_U", torch.stack(U_list))
                    self.register_buffer("class_inv_S", torch.stack(inv_S_list))
                else:
                    _, invcovs = _compute_per_class_means_inv_covs(x, self.y_idx, self.mahalanobis_eps)
                    if self.use_raw_scatter:
                        denom = counts.clamp_min(1).view(C, 1, 1)
                        invcovs /= denom
                    self.register_buffer("class_inv_covs", invcovs)
            return self

    def _compute_distance(self, x: torch.Tensor, y: torch.Tensor):
        diff_cls = x.unsqueeze(1) - self.class_means.unsqueeze(0)

        if self.cov_mode == "diag":
            m2_cls = (diff_cls ** 2 * self.class_inv_diag.unsqueeze(0)).sum(-1)
        elif self.cov_mode == "lowrank":
            proj = torch.einsum('ncd,cdr->ncr', diff_cls, self.class_U)
            m2_cls = (proj ** 2 * self.class_inv_S.unsqueeze(0)).sum(-1)
        else:
            inter = torch.einsum('ncd,cde->nce', diff_cls, self.class_inv_covs)
            m2_cls = (inter * diff_cls).sum(-1)

        diff_glob = x - self.global_mean.unsqueeze(0)
        # use the same resolution for global cov_mode as in _fit
        g_mode = self.global_cov_mode if self.global_cov_mode is not None else self.cov_mode
        if g_mode == "diag":
            m2_0 = (diff_glob ** 2 * self.global_inv_diag.unsqueeze(0)).sum(-1, keepdim=True)
        elif g_mode == "lowrank":
            proj = torch.matmul(diff_glob, self.global_U)
            m2_0 = (proj ** 2 * self.global_inv_S.unsqueeze(0)).sum(-1, keepdim=True)
        else:
            inter_glob = torch.matmul(diff_glob, self.global_inv_cov)
            m2_0 = torch.sum(inter_glob * diff_glob, dim=1, keepdim=True)

        rmd = m2_cls - m2_0
        min_rmd, _ = rmd.min(dim=1)
        return min_rmd


import torch
from confidence.unsupervised.classic.mahalanobis_relative import RelativeMahalanobisConfidence
from pytorch_ood.detector import RMD as PyTorchRMD


class PyTorchRMDWrapper:
    """
    Wrap pytorch_ood.detector.RMD so that:
      - fit_features takes (z, y) and returns self
      - score(z) returns the same raw RMD metric as RelativeMahalanobisConfidence._compute_distance
    """
    def __init__(self):
        # use identity model: features are precomputed
        self.detector = PyTorchRMD(model=lambda t: t)

    def fit(self, z: torch.Tensor, y: torch.Tensor):
        # directly fit on features
        self.detector.fit_features(z, y)
        return self

    def score(self, z: torch.Tensor) -> torch.Tensor:
        # pytorch_ood returns min(d_k - d_0), so invert sign to match ours
        return self.detector.predict_features(z)


if __name__ == "__main__":
    torch.manual_seed(0)

    # toy data
    N, D, C = 200, 2, 3
    z = torch.randn(N, D)
    y = torch.randint(0, C, (N,))

    # our implementation
    ours = RelativeMahalanobisConfidence(shared_covariance=True,confidence_function=lambda x: x,use_raw_scatter=True)
    ours._fit(z, y)

    #print cov
    print("Class means:", ours.class_means)
    print("Global mean:", ours.global_mean)
    print("Global inverse covariance:", ours.global_inv_cov)
    print("Class inverse covariances:", ours.class_inv_covs)


    # pytorch_ood implementation
    pt = PyTorchRMDWrapper().fit(z, y)
    print("Class means (PyTorch):", pt.detector.mu)
    print("Global mean (PyTorch):", pt.detector.background_mu)
    print("Global inverse covariance (PyTorch):", pt.detector.background_precision)
    print("Class inverse covariances (PyTorch):", pt.detector.precision)

    assert torch.allclose(ours.class_means, pt.detector.mu)
    assert torch.allclose(ours.global_mean, pt.detector.background_mu)
    assert torch.allclose(ours.global_inv_cov, pt.detector.background_precision)
    assert torch.allclose(ours.class_inv_covs, pt.detector.precision)


    ours_scores = ours.forward(z[:10])  # get scores for first 10 samples
    pt_scores = pt.score(z[:10])  # get scores for first 10 samples

    print("Ours scores:", ours_scores)
    print("PyTorch scores:", pt_scores)

    #check that score does not change when repeating z
    z_ten_times = z.repeat(100, 1)
    #now fit
    y_ten_times = y.repeat(100)
    ours_ten = RelativeMahalanobisConfidence(shared_covariance=True, confidence_function=lambda x: x, use_raw_scatter=True)
    ours_ten._fit(z_ten_times, y_ten_times)
    pt_ten = PyTorchRMDWrapper().fit(z_ten_times, y_ten_times)
    ours_ten_scores = ours_ten.forward(z_ten_times[:10])
    pt_ten_scores = pt_ten.score(z_ten_times[:10])
    print("Ours ten scores:", ours_ten_scores)
    print("PyTorch ten scores:", pt_ten_scores)
    assert torch.allclose(ours_ten_scores, pt_ten_scores)
    assert torch.allclose(ours_scores, pt_scores)