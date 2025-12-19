"""
This model contains TemperatureCalibrationModule which can be used to temperate scale the logits of a model given a calibration dataset.
Did not really change the results as confidence was still a rather bad estimate.
"""


import torch.nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader


class TemperatureCalibrationModule(torch.nn.Module):
    """
    A module for temperature scaling to make probabilities more similar to the true distribution.
    """
    def __init__(self, model: torch.nn.Module, per_logit: bool = False) -> None:
        """
        Initialize the TemperatureCalibrationModule.

        Args:
            model (torch.nn.Module): The model to be calibrated. It should return logits.
            per_logit (bool): If True, use a separate temperature for each logit. If False, use a single temperature for all logits.
        """
        super(TemperatureCalibrationModule, self).__init__()
        self.model = model
        self.per_logit = per_logit
        self.temperature = None

    def fit(self, dataloader: DataLoader, device: str = 'cuda') -> None:
        """
        Fit the temperature scaling to the model's logits.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the calibration dataset.
                Each batch should return:
                  - data: Tensor of shape (batch_size, ...) the input data.
                  - labels: Tensor of shape (batch_size,) the ground truth labels.
            device (str): The device to use for computation.
        """
        self.model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                logits = self.model(data.to(device))
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())

            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)



        # Compute temperature

        if self.per_logit:
            self.temperature = torch.nn.Parameter(torch.ones(logits.size(-1), device=logits.device))
        else:
            self.temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
        self.register_parameter('temperature', self.temperature)
        self._compute_temperature(logits, labels)

    def _compute_temperature(self, logits: Tensor, labels: Tensor) -> None:
        """
        Compute the temperature for scaling the logits using LBFGS optimization.

        Args:
            logits (Tensor): The logits from the model. Expected shape is (num_samples, num_classes).
            labels (Tensor): The true labels. Expected shape is (num_samples,).

        Returns:
            None
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)
        def closure() -> Tensor:
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = torch.nn.functional.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        optimizer.step(closure)

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns the temperature-scaled logits of the model.

        Args:
            x (Tensor): Batched input data for the model. Expected shape: (batch_size, ...).

        Returns:
            Tensor: Scaled logits with shape corresponding to (batch_size, num_classes).
        """
        logits = self.model(x)
        if self.per_logit:
            return logits / self.temperature.unsqueeze(0)
        else:
            return logits / self.temperature
