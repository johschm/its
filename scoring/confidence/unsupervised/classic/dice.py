import torch
import numpy as np
import logging
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_ood.utils import extract_features, is_known
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from confidence.input_transform import InputTransform
from pytorch_ood.detector import DICE
from confidence.base_confidence import ConfidenceModule
from confidence.direct.logit_based import EnergyConfidence
from experiment_thesis.dataset_preperation.basic_networks import find_last_linear_layer

log = logging.getLogger(__name__)

class DICEConfidence(ClassicConfidenceBase):
    """
    Feature-based confidence for DICE with passable score-to-confidence mapping.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        percentile: float,
        input_transform: InputTransform = None,
        map_function=None,
        confidence: ConfidenceModule = EnergyConfidence()
    ):
        super().__init__(input_transform)
        linear = find_last_linear_layer(model)
        w = linear.weight.detach().cpu()
        b = linear.bias.detach().cpu()
        self.dice = DICE(model=None, w=w, b=b, p=percentile,detector=None)
        self.dice.detector = confidence  # Set detector to the confidence module
        self.map_fn = map_function or (lambda score: score)
        self.confidence = confidence
        self.linear = linear  # Store the linear layer for feature extraction

    def _fit(self, X: Tensor, y: Tensor) -> "DICEConfidence":
        feats = X.detach().cpu()
        labs = y.detach().cpu()
        keep = is_known(labs)
        if not keep.any():
            raise ValueError("No in-distribution data for DICEConfidence")
        self.dice.fit_features(feats[keep], labs[keep])
        return self

    def _forward(self, X: Tensor, y: Tensor = None) -> Tensor:
        feats = X
        scores = self.dice.predict_features(feats)
        conf = self.map_fn(scores)
        return conf.to(X.device, X.dtype)