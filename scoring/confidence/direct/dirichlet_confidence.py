# python
import torch
from confidence.base_confidence import ConfidenceModule





class DirichletExpectedProbabilityConfidence(ConfidenceModule):
    """
    Compute class probabilities from Dirichlet parameters and then apply
    any other ConfidenceModule on those probabilities. The model has to be trained with torch uncertainty's DEC loss and
    should have positive outputs in the last layer(otherwise the dce loss will do the relu activation for you).
    """
    def __init__(self, base_confidence: ConfidenceModule):
        super().__init__()
        self.base_conf = base_confidence

    def forward(self, logits: torch.Tensor, y=None) -> torch.Tensor:
        # derive evidence
        e = torch.relu(logits)


        alpha = e + 1.0
        S = alpha.sum(dim=-1, keepdim=True)       # (..., 1)
        p_hat = alpha / S                          # (..., K)
        # pass expected probabilities into the wrapped confidence module
        return self.base_conf(p_hat)


import torch
from confidence.base_confidence import ConfidenceModule

class DirichletConfidence(ConfidenceModule):
    """
    Compute confidence = 1 - (K / S) where
      e_k = activation(z_k) (relu, softplus or exp)
      alpha_k = e_k + 1
      S = sum_k alpha_k
      K = number of classes

    The model has to be trained with torch uncertainty's DEC loss and should have positive outputs
    in the last layer(the dce loss will do the relu activation for you).
    """
    def __init__(self,base_confidence: ConfidenceModule = None):
        super().__init__()
        self.base_conf = base_confidence if base_confidence is not None else ConfidenceModule()

    def forward(self, logits: torch.Tensor, y=None) -> torch.Tensor:
        # logits: (..., K)

        e = torch.relu(logits)

        alpha = e + 1.0
        S = alpha.sum(dim=-1, keepdim=True)  # (..., 1)
        K = alpha.size(-1)
        u = K / S                              # (..., 1)
        confidence = 1.0 - u                   # (..., 1)

        return confidence.squeeze(-1)