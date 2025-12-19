import torch

from confidence.base_confidence import ConfidenceModule


class CurvatureRefiner(ConfidenceModule):
    """
    Refines the energy scores returned by EnergyConfince. To use use something like
    torch.nn.Sequential(EnergyConfidence(), CurvatureRefiner()).
    """
    def __init__(self):
        super(CurvatureRefiner, self).__init__()

    def forward(self, neg_energy,y: torch.Tensor = None):
        """
        Refine the energy scores using curvature. This is done by computing the second derivative of the energy scores.
        Args:
            neg_energy: The energy scores to refine. This should have the shape (*batch_dims,).

        Returns:

        """
        # Compute the curvature of the logits
        g1 = torch.gradient(neg_energy, dim=-1)[0]
        g2 = torch.gradient(g1, dim=-1)[0]
        return -g2

class GaussianSmoothingRefiner(ConfidenceModule):
    """
    Refines the energy scores using Gaussian smoothing. This regures the input to have the form
    (batch_size,channels, samples_next_to_each_other). Gaussian smoothing can be applied channel-wise or not.
    Use as something like torch.nn.Sequential(EnergyConfidence(), GaussianSmoothingRefiner()).
    """
    def __init__(self, sigma=2.0,radius=3,channel_wise=False):
        """
        Gaussian smoothing refiner. This requires the input to have the form (batch_size,channels, samples_next_to_each_other).
        Args:
            sigma: The standard deviation of the Gaussian kernel.
            radius: The radius of the Gaussian kernel.(The kernel size is 2*radius+1)
            channel_wise: If True, apply the Gaussian smoothing channel-wise. If False, apply it across all channels.
        """
        super(GaussianSmoothingRefiner, self).__init__()
        self.sigma = sigma
        self.radius = radius
        self.channel_wise = channel_wise

    def forward(self, confidence_scores,y: torch.Tensor = None):
        """
        Apply Gaussian smoothing to the confidence scores. This requires the input to have the form
        Args:
            confidence_scores: The confidence scores to smooth. This should have the shape (batch_size,channels, samples_next_to_each_other).

        Returns:
            The smoothed confidence scores. With the same shape as the input.

        """
        #for this we assume our input has shape (batch_size, samples_next_to_each_other)
        # Apply Gaussian smoothing
        gaussian_kernel = torch.exp(-torch.arange(-self.radius, self.radius + 1,dtype=confidence_scores.dtype,device=confidence_scores.device)**2 / (2 * self.sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()
        #pad negative_energy_scores
        confidence_scores = torch.nn.functional.pad(confidence_scores, (self.radius, self.radius), mode='replicate')
        expanded = False
        if confidence_scores.dim() <=2:
            confidence_scores = confidence_scores.unsqueeze(-2)
            expanded = True
        in_channels = confidence_scores.shape[1]
        if self.channel_wise:
            # Apply Gaussian smoothing
            gaussian_kernel = gaussian_kernel.view(1, 1, -1).tile(in_channels, 1, 1)
            confidence_scores = torch.nn.functional.conv1d(confidence_scores, gaussian_kernel, groups=in_channels)
        else:
            # Apply Gaussian smoothing
            gaussian_kernel = gaussian_kernel.view(1, 1, -1).tile(in_channels, in_channels, 1)
            confidence_scores = torch.nn.functional.conv1d(confidence_scores, gaussian_kernel, groups=1)
        if expanded:
            confidence_scores = confidence_scores.squeeze(-2)
        return confidence_scores







