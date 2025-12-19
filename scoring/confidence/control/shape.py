import torch

from confidence.base_confidence import ConfidenceModule


class ReshapingConfidence(ConfidenceModule):
    """
    This class reshapes the input tensor to a specified shape and then applies a confidence module.
    It is useful when the input tensor needs to be reshaped before computing confidence scores.
    """

    def __init__(self, confidence: torch.nn.Module, new_shape: tuple):
        """
        Initializes the ReshapingConfidence class.

        Args:
            confidence: A torch.nn.Module that computes the confidence scores.
            new_shape: The new shape to which the input tensor will be reshaped.
        """
        super(ReshapingConfidence, self).__init__()
        self.confidence = confidence
        self.new_shape = new_shape

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Reshapes the input tensor and computes confidence scores.

        Args:
            x: Input tensor for the model. Shape: (*batch_dims, *data_dims)
            y: Optional labels for modules that use them.

        Returns:
            confidence_scores: Confidence scores computed by the confidence module. Shape: (*batch_dims)
        """
        batch_dims = x.shape[:-len(self.new_shape)]
        reshaped_x = x.view(*batch_dims, *self.new_shape)
        return self.confidence(reshaped_x, y)
