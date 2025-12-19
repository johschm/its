import torch
from typing import Optional, Union
from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase

class LOFTorchConfidence(ClassicConfidenceBase):
    """
    Pure PyTorch Local Outlier Factor (LOF) implementation.
    """
    def __init__(
        self,
        n_neighbors: int = 20,
        input_transform: Optional[InputTransform] = None,
    ):
        super().__init__(input_transform=input_transform)
        self.k = n_neighbors




    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> "LOFTorchConfidence":
        x = x
        dists = torch.cdist(x, x)
        knn_dists, knn_idx = torch.topk(dists, self.k + 1, largest=False)
        knn_dists = knn_dists[:, 1:]
        knn_idx = knn_idx[:, 1:]
        k_distances = knn_dists[:, -1]
        reach = torch.maximum(k_distances[knn_idx], knn_dists)
        lrd_train = self.k / (reach.sum(dim=1) + 1e-10)
        self.register_buffer("X_train",x)
        self.register_buffer("k_distances", k_distances)
        self.register_buffer("lrd_train", lrd_train)

        return self

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.X_train is None:
            raise ValueError("Call fit() before forward().")
        x = x
        d_xt = torch.cdist(x, self.X_train)
        d_knn, idx_knn = torch.topk(d_xt, self.k, largest=False)
        k_dist_nn = self.k_distances[idx_knn]
        reach_x = torch.maximum(k_dist_nn, d_knn)
        lrd_x = self.k / (reach_x.sum(dim=1) + 1e-10)
        lrd_neighbors = self.lrd_train[idx_knn]
        lof = (lrd_neighbors.mean(dim=1) / (lrd_x + 1e-10))
        return 1.0 / (1.0 + lof)
