from typing import Any, Optional, Callable, Union

import sklearn
import torch
import numpy as np

from math import pi
from scipy.special import logsumexp

from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform  # new import
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase


class GaussianMixture(torch.nn.Module):
    """
    Gaussian Mixture Model (GMM) class for calculating log probabilities and sampling from the distribution.
    """

    def __init__(self, n_components: int, n_features: int, covariance_type="full", eps=1.e-6, init_params="kmeans", reg_covar=1e-6):
        """
        Initialize the GaussianMixture model.

        Args:
            n_components (int): Number of mixture components.
            n_features (int): Number of features in the data.
            covariance_type (str): Type of covariance to use. Options are "full", "diag", "spherical", "tied".
        """
        super(GaussianMixture, self).__init__()
        self.n_components = n_components
        self.n_features = n_features

        self.covariance_type = covariance_type

        self.eps = eps
        self.init_params = init_params
        self.reg_covar = reg_covar

        # Initialize parameters
        self.means = torch.nn.Parameter(torch.randn(n_components, n_features))
        self.weights = torch.nn.Parameter(torch.ones(n_components) / n_components)
        if self.covariance_type == "full":
            self.covariances = torch.nn.Parameter(torch.eye(n_features).expand(n_components, -1, -1))
        elif self.covariance_type == "diag":
            self.covariances = torch.nn.Parameter(torch.ones(n_components, n_features))
        elif self.covariance_type == "spherical":
            self.covariances = torch.nn.Parameter(torch.ones(n_components))
        elif self.covariance_type == "tied":
            self.covariances = torch.nn.Parameter(torch.eye(n_features))

        self.log_det_covariances = None
        self._precision = None

    def fit(self, x: torch.Tensor, max_iter: int = 100, tol: float = 1e-4) -> None:
        """
        Fit the Gaussian Mixture Model to the data.
        """
        sklearn_gmm = sklearn.mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            init_params=self.init_params,
            max_iter=max_iter,
            tol=tol,
            reg_covar=self.reg_covar
        )
        sklearn_gmm.fit(x.cpu().numpy())

        # copy sklearn params into torch
        self.means.data = torch.from_numpy(sklearn_gmm.means_).to(self.means.device)
        self.weights.data = torch.from_numpy(sklearn_gmm.weights_).to(self.weights.device)
        if self.covariance_type == "full":
            cov = sklearn_gmm.covariances_
            self.covariances.data = torch.from_numpy(cov).to(self.covariances.device)
        elif self.covariance_type == "diag":
            var = sklearn_gmm.covariances_
            self.covariances.data = torch.from_numpy(var).to(self.covariances.device)
        elif self.covariance_type == "spherical":
            var0 = sklearn_gmm.covariances_
            self.covariances.data = torch.from_numpy(var0).to(self.covariances.device)
        elif self.covariance_type == "tied":
            cov0 = sklearn_gmm.covariances_
            self.covariances.data = torch.from_numpy(cov0).to(self.covariances.device)

        # precompute precision & logdet
        self._compute_precision_and_logdet()

    def _compute_precision_and_logdet(self):
        D = self.n_features
        eps = self.eps
        K = self.n_components

        if self.covariance_type == "full":
            cov = self.covariances + eps * torch.eye(D, device=self.covariances.device)
            # cov: (K,D,D)
            # Add regularization here
            reg = 1e-6 * torch.eye(D, device=self.covariances.device)  # Adjust the regularization strength as needed
            cov = cov + reg
            precision = torch.linalg.inv(cov)
            sign, logdet = torch.linalg.slogdet(cov)
        elif self.covariance_type == "diag":
            var = self.covariances + eps
            precision = 1.0 / var
            logdet = torch.sum(torch.log(var), dim=1)
        elif self.covariance_type == "spherical":
            # expand spherical variances into per-feature diag form
            var = (self.covariances + eps).unsqueeze(1).expand(K, D)  # (K,D)
            precision = 1.0 / var                                    # (K,D)
            logdet = torch.sum(torch.log(var), dim=1)                # (K,)
        elif self.covariance_type == "tied":
            cov0 = self.covariances + eps * torch.eye(D, device=self.covariances.device)
            # Add regularization here
            reg = 1e-6 * torch.eye(D, device=self.covariances.device)  # Adjust the regularization strength as needed
            cov0 = cov0 + reg
            prec0 = torch.linalg.inv(cov0)
            sign0, logdet0 = torch.linalg.slogdet(cov0)
            precision = prec0.unsqueeze(0).expand(K, D, D)
            logdet = logdet0.expand(K)

        self._precision = torch.nn.Parameter(precision)  # (K,D,D) or (K,D)
        self.log_det_covariances = torch.nn.Parameter(logdet)  # (K,)


    def forward(self, x: torch.Tensor):
        # x: (N,D)
        N, D = x.shape
        K = self.n_components
        # diffs: (N,K,D)
        diffs = x.unsqueeze(1) - self.means.unsqueeze(0)

        # Mahalanobis term
        if self.covariance_type == "full" or self.covariance_type == "tied":
            m = torch.einsum("nkd,kde,nke->nk", diffs, self._precision, diffs)
        else:
            # diag or spherical: precision shape (K,D) or (K,)
            prec = self._precision.unsqueeze(0)  # (1,K,D) or (1,K,1)
            m = torch.sum(diffs * diffs * prec, dim=2)

        # constant term + logdet + log weights
        const = D * torch.log(torch.tensor(2 * pi, device=x.device))
        log_w = torch.log(self.weights).unsqueeze(0)             # (1,K)
        ld = self.log_det_covariances.unsqueeze(0)               # (1,K)
        comp_logprob = -0.5 * (const + ld + m) + log_w           # (N,K)

        # total log‐prob
        return torch.logsumexp(comp_logprob, dim=1)              # (N,)

    def score_samples(self, x: torch.Tensor):
        return self.forward(x)


class GaussianMixtureConfidence(ClassicConfidenceBase):
    """
    Confidence based on GMM log-density: higher log-prob ⇒ higher confidence.
    Optionally apply a mapping fn (e.g., sigmoid) to constrain to [0,1].
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        map_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform: Optional[InputTransform] = None,
        **gmm_kwargs: Any,
    ):
        super().__init__(input_transform=input_transform)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.map_fn = map_function or (lambda x: x)
        self.gmm_kwargs = gmm_kwargs
        self.gmm: Optional[GaussianMixture] = None

    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> "GaussianMixtureConfidence":
        pts = x.float()
        D = pts.shape[-1]
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            n_features=D,
            covariance_type=self.covariance_type,
            **self.gmm_kwargs,
        )
        self.gmm.fit(pts)
        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (*batch_dims, feature_dim)
        Returns confidence = map_fn(log_density), shape (*batch_dims).
        """
        if self.gmm is None:
            raise ValueError("Call fit() before forward()")
        
        orig = x.shape[:-1]
        D = x.shape[-1]
        flat = x.reshape(-1, D)
        logp = self.gmm.score_samples(flat)
        conf = self.map_fn(logp)
        return conf.reshape(*orig)  # (*batch_dims,)


if __name__ == "__main__":
    import numpy as np
    import torch
    from sklearn.mixture import GaussianMixture as SKGMM

    # generate synthetic 2-D data
    np.random.seed(0)
    X = np.vstack([
        np.random.randn(100, 2) + np.array([5.0, 5.0]),
        np.random.randn(100, 2) + np.array([-5.0, -5.0])
    ])
    x_torch = torch.from_numpy(X).float()

    # fit sklearn GMM and get log-probs
    sk_gmm = SKGMM(n_components=2, covariance_type="spherical", random_state=0).fit(X)
    sk_logprob = sk_gmm.score_samples(X)

    # instantiate our model and copy SKGMM params
    model = GaussianMixture(n_components=2, n_features=2, covariance_type="spherical")
    model.means.data       = torch.from_numpy(sk_gmm.means_).to(model.means.device)
    model.weights.data     = torch.from_numpy(sk_gmm.weights_).to(model.weights.device)
    model.covariances.data = torch.from_numpy(sk_gmm.covariances_).to(model.covariances.device)
    model._compute_precision_and_logdet()

    # compute torch log-probs
    torch_logprob = model.score_samples(x_torch).detach().cpu().numpy()

    print("Sklearn log-probabilities:", sk_logprob)
    print("Torch log-probabilities:", torch_logprob)

    # compare
    diff = np.abs(sk_logprob - torch_logprob)
    print("Max absolute difference:", diff.max())
    assert np.allclose(sk_logprob, torch_logprob, atol=1e-6), "Mismatch in log-probs!"
    print("Test passed.")