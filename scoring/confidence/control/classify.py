import torch

from confidence.base_confidence import ConfidenceModule


class ClassifyingConfidence(ConfidenceModule):
    """
    Computes confidence given a confidence module that requires a class assignment.
    The class is assumed to be the one assigned the highest logit by the model.
    """

    def __init__(self,confidence: torch.nn.Module = None, index=None,index_confidence=None):
        """
        Initializes the ClassifyingConfidence class.

        Args:
            model: A classification model that outputs confidence scores.
        """
        super(ClassifyingConfidence, self).__init__()
        self.index = index
        self.confidence = confidence
        self.index_confidence = index_confidence

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Computes confidence scores by calling a confidence that requires a class assignment.

        Args:
            x: Input tensor for the model. Shape: (*batch_dims, *data_dims)
            y: Optional labels for modules that use them.

        Returns:
            confidence_scores: Confidence scores computed by the classification model. Shape: (*batch_dims)
        """
        log = x[self.index] if self.index is not None else x
        clas = torch.argmax(log, dim=-1)
        inp2 = x[self.index_confidence] if self.index_confidence is not None else x
        confidence_scores = self.confidence(inp2, clas)
        return confidence_scores