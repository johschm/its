from typing import Any, Dict
import torch
from torch.optim.optimizer import Optimizer
from torch import nn
from pytorch_ood.loss import (
    DeepSVDDLoss,
    CACLoss,
    IILoss,
    CenterLoss,
    MCHADLoss,
)
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
#heads to test. Result all failed for pretrained case. #TODO maybe remove

class OODEncoderConfidence(MLConfidenceBase):
    def __init__(
        self,
        encoder: nn.Module,
        loss_module: nn.Module,
        input_transform: InputTransform | None = None,
        trainer_kwargs: Dict[str, Any] | None = None,
        dataloader_kwargs: Dict[str, Any] | None = None,
        optimizer_type: Optimizer | None = None,
        optimizer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.save_hyperparameters(
            ignore=['encoder', 'loss_module', 'input_transform', 'trainer_kwargs', 'dataloader_kwargs']
        )
        self.encoder = encoder
        self.loss_module = loss_module

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        z = self.encoder(x)
        dist = self.loss_module.distance(z)
        loss = self.loss_module(dist, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x)
        dist = self.loss_module.distance(z)
        loss_vals = self.loss_module(dist, y)
        return loss_vals

    def configure_optimizers(self) -> Optimizer:
        params = list(self.encoder.parameters()) + list(self.loss_module.parameters())
        return self.optimizer_type(params, **self.optimizer_kwargs)


class DeepSVDDEncoderConfidence(OODEncoderConfidence):
    def __init__(
        self,
        encoder: nn.Module,
        n_dim: int,
        radius: float = 0.0,
        center: torch.Tensor | None = None,
        input_transform: InputTransform | None = None,
        trainer_kwargs: Dict[str, Any] | None = None,
        dataloader_kwargs: Dict[str, Any] | None = None,
        optimizer_type: Optimizer | None = None,
        optimizer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        loss_module = DeepSVDDLoss(n_dim=n_dim, radius=radius, center=center)
        super().__init__(
            encoder=encoder,
            loss_module=loss_module,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        z = self.encoder(x)
        loss = self.loss_module(z)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x)
        # add back radius^2 to remove boundary subtraction
        raw = self.loss_module.distance(z) + self.loss_module.radius.pow(2)
        return (1.0 / (1.0 + raw)).clamp(0.0, 1.0)


class CACEncoderConfidence(OODEncoderConfidence):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        magnitude: float = 1.0,
        alpha: float = 1.0,
        input_transform: InputTransform | None = None,
        trainer_kwargs: Dict[str, Any] | None = None,
        dataloader_kwargs: Dict[str, Any] | None = None,
        optimizer_type: Optimizer | None = None,
        optimizer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        loss_module = CACLoss(n_classes=n_classes, magnitude=magnitude, alpha=alpha)
        super().__init__(
            encoder=encoder,
            loss_module=loss_module,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x)
        distances = self.loss_module.distance(z)
        score = self.loss_module.score(z)
        return -score


import torch.nn.functional as F


class MCHADEncoderConfidence(OODEncoderConfidence):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        n_dim: int,
        input_transform: InputTransform | None = None,
        trainer_kwargs: Dict[str, Any] | None = None,
        dataloader_kwargs: Dict[str, Any] | None = None,
        optimizer_type: Optimizer | None = None,
        optimizer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        loss_module = MCHADLoss(n_classes=n_classes, n_dim=n_dim)
        super().__init__(
            encoder=encoder,
            loss_module=loss_module,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
            z = self.encoder(x)
            # add back radius^2 to remove boundary subtraction
            distances = self.loss_module.distance(z)
            #take the score
            scores = distances * (1 - F.softmin(distances, dim=1))
            return scores.max(dim=1).values
