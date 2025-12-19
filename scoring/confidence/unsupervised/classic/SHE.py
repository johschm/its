from typing import Optional, TypeVar
import torch
from torch import nn, Tensor

from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase

Self = TypeVar("Self")

class SHETorchConfidence(ClassicConfidenceBase):
    """
    Simplified Hopfield Energy (feature-only). Adapted from pytorch_ood.detector.SHE.
    Fit per‐class mean patterns, then score via -⟨z, pattern_y⟩.
    """

    def __init__(self, input_transform: Optional[nn.Module] = None,map_function: Optional[callable] = None):

        super().__init__(input_transform=input_transform)
        self.patterns_: Optional[Tensor] = None
        self.fitted = False
        self.map_function = map_function if map_function is not None else lambda x: -x
        #print("For correct functionality one must prefilter embeddings to only keep correctly classified ones.")
        self.penalize_missmatches = True  # penalize misclassifications by default
        self.cosine_debug = False  # debug flag for cosine similarity

    def _fit(self, z: Tensor, y: Tensor) -> Self:
        # classes must be 0..C-1
        classes = torch.unique(y)
        assert len(classes) == classes.max().item() + 1, "labels must cover 0..C-1"
        # compute per-class mean feature
        patterns = [z[y == c].mean(dim=0) for c in classes]
        self.patterns_ = torch.stack(patterns, dim=0)
        self.fitted = True
        return self

    def _forward(self, z: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if not self.fitted:
            raise RuntimeError("Call fit() before forward()")
        if y is None:
            raise ValueError("Class labels 'y' required at inference")

        # compute only true‐class scores
        patterns_y = self.patterns_[y]  # (batch_size, feature_dim)
        if not self.cosine_debug:
            true_scores = (z * patterns_y).sum(dim=1)  # (batch_size,)
        else:
            z_norm = z / z.norm(dim=1, keepdim=True)
            patterns_y_norm = patterns_y / patterns_y.norm(dim=1, keepdim=True)
            cosine_distance = 1-(z_norm * patterns_y_norm).sum(dim=1)
            true_scores = -cosine_distance  # (batch_size,)

        return self.map_function(-true_scores)


