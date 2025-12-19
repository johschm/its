from typing import Optional, Callable, Dict, Any
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.direct.logit_based import EnergyConfidence
from pytorch_ood.loss import (
    ConfidenceLoss,
    OutlierExposureLoss,
    EntropicOpenSetLoss,
    ObjectosphereLoss,
    EnergyRegularizedLoss,
    VOSRegLoss,
    VirtualOutlierSynthesizingRegLoss,
    BackgroundClassLoss
)

#heads to test. Result all failed for pretrained case. #TODO maybe remove

class PytorchOODConfidenceBase(MLConfidenceBase):
    loss_class: Any = None

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
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
        # print all kwargs
        print("encoder =", encoder)
        print("loss_args =", loss_args)
        print("input_transform =", input_transform)
        print("trainer_kwargs =", trainer_kwargs)
        print("dataloader_kwargs =", dataloader_kwargs)
        print("optimizer_type =", optimizer_type)
        print("optimizer_kwargs =", optimizer_kwargs)

        self.encoder = encoder
        self.loss_args = loss_args
        self.ood_loss: torch.nn.Module = self.loss_class(**loss_args)
        self.energy_conf = EnergyConfidence()

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch[0], (batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None)
        out = self.encoder(x)
        if isinstance(self.ood_loss, ObjectosphereLoss):
            logits, feats = out, x
        else:
            logits, feats = out, None
        # ConfidenceLoss and BackgroundClassLoss: use extra class probability
        if isinstance(self.ood_loss, ConfidenceLoss):
            loss = self.ood_loss(logits[:, :-1], logits[:, [-1]], y)
        else:
            if feats is not None:
                loss = self.ood_loss(logits, feats, y)
            else:
                loss = self.ood_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        out = self.encoder(x)
        if isinstance(out, dict):
            logits, feats = out["logits"], out.get("features")
        else:
            logits, feats = out, None

        # ConfidenceLoss and BackgroundClassLoss: use extra class probability
        if isinstance(self.ood_loss, (ConfidenceLoss, BackgroundClassLoss)):
            probs = torch.softmax(logits, dim=-1)
            p_ood = probs[..., -1]
            return 1.0 - p_ood

        # ObjectosphereLoss: use its score method on embeddings
        if isinstance(self.ood_loss, ObjectosphereLoss):
            raw = self.ood_loss.score(x)
            return torch.sigmoid(raw)

        # All others: energy-based confidence
        return self.energy_conf(logits)

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)


class ConfidenceLossConfidence(PytorchOODConfidenceBase):
    loss_class = ConfidenceLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any]={},
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )


class OutlierExposureLossConfidence(PytorchOODConfidenceBase):
    loss_class = OutlierExposureLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )


class EntropicOpenSetLossConfidence(PytorchOODConfidenceBase):
    loss_class = EntropicOpenSetLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )


class ObjectosphereLossConfidence(PytorchOODConfidenceBase):
    loss_class = ObjectosphereLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )


class EnergyRegularizedLossConfidence(PytorchOODConfidenceBase):
    loss_class = EnergyRegularizedLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )


class VOSRegLossConfidence(PytorchOODConfidenceBase):
    loss_class = VOSRegLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )


class VirtualOutlierSynthesizingRegLossConfidence(PytorchOODConfidenceBase):
    loss_class = VirtualOutlierSynthesizingRegLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )


class BackgroundClassLossConfidence(PytorchOODConfidenceBase):
    loss_class = BackgroundClassLoss

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_args: Dict[str, Any],
        input_transform: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            encoder=encoder,
            loss_args=loss_args,
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs
        )