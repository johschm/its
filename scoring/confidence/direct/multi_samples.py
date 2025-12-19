#TODO adjust to make similar to other confidence modules

import torch
import torch.nn.functional as F

from confidence.base_confidence import ConfidenceModule

class MutualInformationCriterion(ConfidenceModule):
    def __init__(self, input_logits=False):

        super().__init__()
        # This criterion computes the mutual information between the mean and individual predictions
        # It is based on the assumption that higher mutual information indicates more confidence in the predictions.
        # The output is normalized to be in the range [0, 1].
        self.input_logits = input_logits  # This criterion expects softmax probabilities, not logits

    def forward(self, outputs, y=None):
        # outputs: [batch, samples, classes]
        if self.input_logits:
            # If the input is logits, apply softmax to convert to probabilities
            outputs = F.softmax(outputs, dim=-1)
        p = outputs
        p_mean = p.mean(dim=-2)
        h_mean = -torch.sum(p_mean * torch.log(p_mean + 1e-12), dim=-1)
        h_per = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
        h_ind = h_per.mean(dim=-1)
        mi = h_mean - h_ind
        max_mi = torch.log(torch.tensor(outputs.size(-1), device=mi.device))
        mi_norm = mi / (max_mi + 1e-12)
        return 1.0 - mi_norm



class VariationRatioCriterion(ConfidenceModule):
    """Variation Ratio Criterion- Non Differentiable Criterion
    """
    def forward(self, outputs, y=None):
        # outputs: [batch, samples, classes]
        preds = outputs.argmax(dim=-1)            # [batch, samples]
        mode_vals, _ = preds.mode(dim=-2)
        counts = (preds == mode_vals.unsqueeze(-2)).sum(dim=-2)
        vr = 1.0 - counts.float() / outputs.size(-2)
        return 1.0 - vr


class EnergyMultiSampleConfidence(ConfidenceModule):
    """
    Energy-based confidence for multiple samples.
    Computes energy for each sample, then averages.
    Input: logits [batch, samples, classes] (not averaged)
    """

    def __init__(self, t: float = 1.0):
        super().__init__()
        self.t = t

    def forward(self, outputs, y=None):
        """
        Args:
            outputs: Logits from multiple samples. Shape: [batch, samples, classes]
        Returns:
            confidence: Averaged energy scores. Shape: [batch]
        """
        # outputs: [batch, samples, classes]
        # Compute energy for each sample: -t * log(sum(exp(logits/t)))
        energy_per_sample = -self.t * torch.logsumexp(outputs / self.t, dim=-1)  # [batch, samples]

        # Average energy across samples
        energy_avg = energy_per_sample.mean(dim=-1)  # [batch]

        return energy_avg


class VarianceCriterion(ConfidenceModule):
    def forward(self, outputs, y=None):
        # outputs: [batch, samples, classes]
        var = outputs.var(dim=-2)                        # [batch, classes]
        u = var.sum(dim=-1)
        return 1.0 / (1.0 + u)


class AgreementScoreCriterion(ConfidenceModule):
    def forward(self, outputs, y=None):
        distances = torch.cdist(outputs, outputs, p=2)  # [batch, samples, samples]

        # Set diagonal (self-distances) to zero explicitly
        batch_size, num_samples, _ = distances.shape
        eye = torch.eye(num_samples, device=outputs.device).unsqueeze(0)  # [1, samples, samples]
        distances = distances * (1 - eye)  # Zero out the diagonal


        avg_dist = distances.sum(dim=(1, 2)) / (num_samples * (num_samples - 1))

        return 1.0/ (1.0 + avg_dist)