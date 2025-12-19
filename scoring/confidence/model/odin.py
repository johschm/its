import warnings
import torch
import torch.nn.functional as F
from typing import Optional, Callable, Union
from confidence.base_confidence import ConfidenceModule
from confidence.scaler.calibration import TemperatureCalibrationModule
from confidence.scaler.torch_uncertainty_scaler import (
    TemperatureScalingWrapper,
    MatrixScalingWrapper,
    VectorScalingWrapper,
)

_SCALED_MODULES = (
    TemperatureCalibrationModule,
    TemperatureScalingWrapper,
    MatrixScalingWrapper,
    VectorScalingWrapper,
)
#TODO add the possibility to add another confidenec module to multiply the confidence with.

class OdinConfidence(ConfidenceModule):
    def __init__(
            self,
            model: torch.nn.Module,
            epsilon: float = 1e-3,
            map_function: Optional[Callable] = None,
    ):
        """
        ODIN detector with separate confidence scoring and decision thresholding.

        Args:
            model: Trained classifier (should be temperature-scaled for best results)
            epsilon: Perturbation magnitude (ε in paper)
            temperature: Temperature scaling parameter (T in paper)
            map_function: Optional post-processing for confidence scores
        """
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.map_fn = map_function or (lambda x: x)

        if not isinstance(self.model, _SCALED_MODULES):
            warnings.warn(
                "Model is not a temperature-scaled wrapper. "
            )

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        self.model.eval()
        x_adv = x.detach().clone().requires_grad_(True)

        # First forward pass with temperature scaling
        with torch.enable_grad():
            logits = self.model(x_adv)

            # Generate perturbation using max softmax (original ODIN formulation)
            softmax_scores = F.softmax(logits, dim=1)
            max_softmax, _ = softmax_scores.max(dim=1)
            loss = -torch.log(max_softmax).mean()  # Negative log of max softmax

            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.detach_()
                x_adv.grad.zero_()
            loss.backward()

        # Rest remains the same...
        x_pert = x_adv - self.epsilon * x_adv.grad.sign()

        logits_pert = self.model(x_pert)
        probs = F.softmax(logits_pert, dim=1)
        max_probs, _ = probs.max(dim=1)

        return self.map_fn(max_probs), logits
