import torch

from confidence.model.base_model import ModelBasedConfidence
from optree import tree_map


class BayesianModelSamplingConfidence(ModelBasedConfidence):
    """
    Multi-sample confidence for torch_uncertainty Bayesian models:
    - Verifies presence of Bayesian layers via `kl_loss`
    - Runs `samples` forward passes
    - Uses EntropyConfidence to compute negative predictive entropy
    """
    def __init__(self, model: torch.nn.Module,confidence, samples: int = 10,index: int = None,average: bool = True):
        super().__init__(model, confidence=confidence,index=index)
        self.samples = samples
        # ensure model has Bayesian layers
        self.model.eval()
        self.average = average


    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        # [batch, samples, num_classes]
        all_values =[self.model(x) for _ in range(self.samples)]
        # compute confidence via existing EntropyConfidence
        if self.average:
            mc_logits = tree_map(lambda *ts: torch.stack(ts, dim=1).mean(dim=1), *all_values)
        else:
            mc_logits = tree_map(lambda *ts: torch.stack(ts, dim=1), *all_values)

        confidence = self.confidence(mc_logits, y)
        if self.index is None:
            return confidence, mc_logits
        else:
            return confidence, mc_logits[self.index]