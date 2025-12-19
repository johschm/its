import torch
import pytorch_lightning as pl
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from torch_uncertainty.losses import DECLoss

from confidence.direct.dirichlet_confidence import DirichletConfidence
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import MLConfidenceBase


class DecFineTuningConfidence(MLConfidenceBase):
    """
    Fine-tune any head model on top of a frozen encoder using a DEC-style loss,
    then apply a pluggable confidence module (defaults to DirichletConfidence).
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
        *,
        dec_loss: torch.nn.Module = DECLoss(),
        confidence: Optional[pl.LightningModule] = None,
        input_transform: Optional[InputTransform] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )

        # freeze encoder
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # learnable head + loss + confidence
        self.head = head
        self.dec_loss = dec_loss
        self.confidence_module = confidence or DirichletConfidence()

        # datasets



    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        feats = self.encoder(x)
        logits = self.head(feats)
        loss = self.dec_loss(logits, y, self.current_epoch)
        self.log("train_loss", loss)
        return loss

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.head(feats)
        return self.confidence_module(logits)