import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch
import pytorch_lightning as pl
from confidence.input_transform import InputTransform
from typing import Optional, Callable, Union, Any, Dict

from confidence.unsupervised.unsupervised_base import MLConfidenceBase


class BasicAutoencoderConfidence(MLConfidenceBase):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform=None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
            negative_sampling_module=None,
    ):
        super().__init__(input_transform=input_transform,
                         trainer_kwargs=trainer_kwargs,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_type=optimizer_type,
                         optimizer_kwargs=optimizer_kwargs,negative_sampling_module=negative_sampling_module)
        self.save_hyperparameters(
            ignore=['encoder','decoder','map_fn',
                    'input_transform','trainer_kwargs','dataloader_kwargs']
        )
        self.encoder = encoder
        self.decoder = decoder
        self.map_fn = map_fn or (lambda err: 1.0/(1.0+err))

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        #mean over all except first dim this depends on the input shape
        err = F.mse_loss(recon, x, reduction='none').mean(dim=tuple(range(1, x.ndim)))
         # Filter out OOD samples (y < 0) if y is available
        return self.map_fn(err)

    def _training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
        z = self.encoder(x)
        recon = self.decoder(z)
        err = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        
        # Filter out OOD samples (y < 0) if y is available
        if y is not None:
            mask = y >= 0.0
            if mask.sum() > 0:  # Check if any samples remain
                err = err[mask]
            
        loss = err.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)


class DenoisingAutoencoderConfidence(MLConfidenceBase):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        noise_scale: float = 0.1,
        map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform=None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(input_transform=input_transform,
                         trainer_kwargs=trainer_kwargs,
                         dataloader_kwargs=dataloader_kwargs,
                         optimizer_type=optimizer_type,
                         optimizer_kwargs=optimizer_kwargs)
        self.save_hyperparameters(
            ignore=['encoder','decoder','map_fn',
                    'input_transform','trainer_kwargs','dataloader_kwargs']
        )
        self.encoder = encoder
        self.decoder = decoder
        self.noise_scale = noise_scale
        self.map_fn = map_fn or (lambda err: 1.0/(1.0+err))

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
        x_noisy = x + torch.randn_like(x)*self.noise_scale
        z = self.encoder(x_noisy)
        recon = self.decoder(z)
        err = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        
        # Filter out OOD samples (y < 0) if y is available
        if y is not None:
            mask = y >= 0.0
            if mask.sum() > 0:  # Check if any samples remain
                err = err[mask]
            
        loss = err.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x_noisy = x + torch.randn_like(x)*self.noise_scale
        z = self.encoder(x_noisy)
        recon = self.decoder(z)
        err = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        return self.map_fn(err)

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)


from torch.nn import functional as F
import torch
from typing import Optional, Callable, Any, Dict
from confidence.unsupervised.unsupervised_base import MLConfidenceBase

class ConditionalAutoencoderConfidence(MLConfidenceBase):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        num_classes: int,
        map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform=None,
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
        self.save_hyperparameters(
            ignore=['encoder', 'decoder', 'map_fn',
                    'input_transform', 'trainer_kwargs', 'dataloader_kwargs']
        )
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.map_fn = map_fn or (lambda err: 1.0 / (1.0 + err))

    def _encode_cond(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y_onehot = F.one_hot(y.long(), num_classes=self.num_classes).float()
        return torch.cat([z, y_onehot], dim=1)

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch[0], batch[1]
        z_cond = self._encode_cond(x, y)
        recon = self.decoder(z_cond)
        err = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        loss = err.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_cond = self._encode_cond(x, y)
        recon = self.decoder(z_cond)
        err = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        return self.map_fn(err)

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)

