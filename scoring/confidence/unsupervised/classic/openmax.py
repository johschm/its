import numpy as np
import torch
from pytorch_ood.detector import OpenMax
from scipy.stats import exponweib
from torch import Tensor
from typing import Optional
from confidence.base_confidence import ConfidenceModule
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from confidence.input_transform import InputTransform


# Directly wrap the PyTorch-OOD OpenMax detector


class OpenMaxConfidenceNumpy(ClassicConfidenceBase):
    """
    Simple wrapper around PyTorch-OOD's OpenMax.
    Fits on model logits and returns unknown-class probability.
    """

    def __init__(
            self,
            model: Optional[torch.nn.Module] = None,
            tailsize: int = 25,
            alpha: int = 10,
            input_transform: Optional[InputTransform] = None,
            euclid_weight=0.5
    ):
        super().__init__(input_transform=input_transform)
        self.model = model
        self._openmax = OpenMax(model=model, tailsize=tailsize, alpha=alpha, euclid_weight=euclid_weight)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def _fit(self, x, y: Tensor) -> "OpenMaxConfidence":
        """
        Fit OpenMax on a DataLoader (extracts features via the wrapped model).
        """
        self._openmax.fit_features(x, y)
        return self
    @torch.no_grad()
    def _forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Compute unknown-class probability for input x.

        If using features directly, x should be logits/features; y is unused.
        If using raw inputs, x is passed through the wrapped model first.
        """
        res = 1 - self._openmax.predict_features(x)
        return res.to(x.device, x.dtype)

    @property
    def centers(self) -> Tensor:
        """
        Class centers (means) from the fitted OpenMax model.
        """
        return self._openmax._openmax.centers


import torch
import torch.nn.functional as F
from typing import Optional


class OpenMaxConfidence(ClassicConfidenceBase):
    """
    OpenMax with support for Euclidean, Cosine, or blended Euclidean-Cosine distances via euclid_weight.
    """

    def __init__(
            self,
            tail_size: float = 25,
            alpha: int = 10,
            euclid_weight: float = 0.5,
            input_transform: Optional[InputTransform] = None,
            input_is_logits: bool = True
    ):
        super().__init__(input_transform=input_transform)
        self.tail_size = tail_size
        self.alpha = alpha
        self.euclid_weight = euclid_weight
        self.cos_weight = 1.0 - euclid_weight
        self.translation = 10000.0
        self.register_buffer('class_means', None)
        self.register_buffer('shapes', None)
        self.register_buffer('scales', None)
        self.register_buffer('min_vals', None)
        self.n_classes = None

    def _get_dists(self, x: Tensor) -> Tensor:
        # x: [B, D], class_means: [C, D] -> output [B, C]
        if self.euclid_weight == 1.0:
            return torch.cdist(x, self.class_means, p=2)
        elif self.euclid_weight == 0.0:
            x_norm = F.normalize(x, dim=1)
            m_norm = F.normalize(self.class_means, dim=1)
            cos_sim = torch.matmul(x_norm, m_norm.t())
            return 1 - cos_sim
        else:
            # Euclidean
            euclid = torch.cdist(x, self.class_means, p=2)
            # Cosine
            x_norm = F.normalize(x, dim=1)
            m_norm = F.normalize(self.class_means, dim=1)
            cos_sim = torch.matmul(x_norm, m_norm.t())
            cos_dist = 1 - cos_sim
            # Weighted sum
            return self.euclid_weight * euclid + self.cos_weight * cos_dist

    def _fit(self, X: Tensor, y: Tensor) -> "OpenMaxConfidence":
        classes = torch.unique(y)
        means, shapes, scales, min_vals = [], [], [], []
        for c in classes.tolist():
            feats = X[y == c]
            mu = feats.mean(dim=0)
            dists = torch.norm(feats - mu, dim=1).cpu().numpy()
            k = int(self.tail_size) if self.tail_size >= 1 else max(1, int(len(dists) * self.tail_size))
            tail = np.sort(dists)[-k:]
            min_val = tail.min()
            tail_trans = tail + self.translation - min_val
            _, c_shape, _, scale = exponweib.fit(tail_trans, f0=1, floc=0)
            means.append(mu)
            shapes.append(c_shape)
            scales.append(scale)
            min_vals.append(min_val)
        device = X.device
        self.class_means = torch.stack(means, dim=0).to(device)
        self.shapes = torch.tensor(shapes, device=device, dtype=self.class_means.dtype)
        self.scales = torch.tensor(scales, device=device, dtype=self.class_means.dtype)
        self.min_vals = torch.tensor(min_vals, device=device, dtype=self.class_means.dtype)
        self.n_classes = len(means)
        return self

    def _forward_logits(self, x: Tensor, y=None) -> Tensor:
        assert self.n_classes is not None, "Model not fitted"
        assert x.dim() == 2, "Input must be 2D tensor [B, D]"
        batch = x.size(0)
        # get weighted distances
        dists = self._get_dists(x)  # [B, C]
        # translate/clamp
        tail = dists + self.translation - self.min_vals.unsqueeze(0)
        tail = torch.clamp(tail, min=0)
        # Weibull CDF
        c = self.shapes.unsqueeze(0)
        s = self.scales.unsqueeze(0)
        cdf = 1 - torch.exp(- (tail / s) ** c)
        raw_w = cdf
        effective_alpha = min(self.alpha, self.n_classes)
        topk_vals, topk_idx = torch.topk(x, effective_alpha, dim=1)
        idx = torch.arange(1, effective_alpha + 1, device=x.device, dtype=torch.float32)
        alpha_w = ((effective_alpha + 1) - idx) / float(effective_alpha)
        w = torch.zeros_like(x)
        for i in range(effective_alpha):
            cls = topk_idx[:, i]
            w[torch.arange(batch), cls] = raw_w[torch.arange(batch), cls] * alpha_w[i]
        rev = x * (1 - w)
        outlier = (x * w).sum(dim=1, keepdim=True)
        logits = torch.cat([outlier, rev], dim=1)
        return F.softmax(logits, dim=1)[:, 0]

    def _forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        return 1 - self._forward_logits(x, y)


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.datasets import make_blobs

    # --- Simple 2D blob test for OpenMaxConfidence and OpenMaxConfidenceNumpy ---

    # 1. Generate toy data
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 2. Simple linear classifier
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
        def forward(self, x):
            return self.fc(x)

    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # 3. Train classifier
    for epoch in range(100):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 4. Get logits for training data
    with torch.no_grad():
        train_logits = model(X)

    # 5. Fit both OpenMaxConfidence and OpenMaxConfidenceNumpy on logits
    openmax = OpenMaxConfidence(tail_size=10, alpha=1, euclid_weight=1.0)
    openmax.fit(train_logits, y)

    openmax_numpy = OpenMaxConfidenceNumpy(model=None, tailsize=10, alpha=1, euclid_weight=1.0)
    openmax_numpy.fit(train_logits, y)

    # 6. Test on in-distribution and OOD points
    test_points = torch.tensor([[0, 0], [2, 2], [10, 10], [-10, -10]], dtype=torch.float32)
    test_logits = model(test_points)

    conf_openmax = openmax(test_logits)
    conf_numpy = openmax_numpy(test_logits)

    print("OpenMaxConfidence unknown-class probabilities:", conf_openmax.detach().numpy())
    print("OpenMaxConfidenceNumpy unknown-class probabilities:", conf_numpy.detach().numpy())
    print("Difference:", (conf_openmax - conf_numpy).detach().numpy())

    # --- End of blob test ---

    # --- Original synthetic test for OpenMaxConfidence and OpenMaxConfidenceNumpy ---

    # Synthetic data
    logits = torch.randn(1000, 5) * 1  # 100 samples, 5 classes

    labels = torch.cat(
        [torch.zeros(200, dtype=torch.long), torch.ones(200, dtype=torch.long), torch.full((200,), 2, dtype=torch.long),
         torch.full((200,), 3, dtype=torch.long), torch.full((200,), 4, dtype=torch.long)])
    logits[500:, 1] += 5  # Class 0 centered around 5
    X = logits  # Use logits directly
    y = labels  # Corresponding labels

    # Wrap OpenMaxConfidence (no model): fit on features
    omc = OpenMaxConfidence(tail_size=25, alpha=1, euclid_weight=1.0)
    omc.fit(X, y)

    omc2 = OpenMaxConfidenceNumpy(model=None, tailsize=25, alpha=1, euclid_weight=1.0)
    omc2.fit(X, y)
    # Print fitted parameters
    print("OpenMax class means:", omc.class_means.detach().numpy())
    print("OpenMaxNumpy class means:", omc2.centers)

    # Test points
    xs = torch.randn(50, 5) * 1
    xs[0] = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0])  # Extreme OOD point
    xs[1] = torch.tensor([-100.0, -100.0, -100.0, -100.0, -100.0])  # Extreme OOD point
    conf2 = omc2(xs)  # unknown-class probs
    conf = omc(xs)  # unknown-class probs
    xs[2] = torch.tensor([5.0, 0.0, 0.0, 0.0, 5.0])  # Near class 0
    xs[3] = torch.tensor([0.0, 5.0, 0.0, 0.0, 5.0])  # Near class 1
    conf2 = omc2(xs)  # unknown-class probs
    conf = omc(xs)  # unknown-class probs

    xs[4] = torch.tensor([0.0, 0.0, 5.0, 0.0, 5.0])  # Near class 2
    xs[5] = torch.tensor([0.0, 2.0, 1.0, 5.0, 5.0])  # Near class

    # dif
    print("Difference in unknown-class probabilities:", (conf - conf2).detach().numpy())

    print("OpenMax unknown-class probabilities:", conf.detach().numpy())
    print("OpenMaxNumpy unknown-class probabilities:", conf2.detach().numpy())
