import math
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import MLConfidenceBase


class VAEConfidence(MLConfidenceBase):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim=None,
        beta: float = 1.0,
        map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mode: str = "elbo",  # options: 'reconstruction', 'elbo', 'prior', 'kld'
        use_sampling: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.save_hyperparameters(
            ignore=[
                "encoder",
                "decoder",
                "input_transform",
                "map_fn",
                "trainer_kwargs",
                "dataloader_kwargs",
            ]
        )
        self.encoder = encoder
        self.decoder = decoder
        self.mode = mode
        self.use_sampling = use_sampling
        self.beta = beta
        self.map_fn = map_fn if map_fn is not None else (lambda x: x)
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs or {}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_prior_log_density(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.view(z.size(0), -1)
        dim_z = z_flat.size(1)
        const = dim_z * math.log(2.0 * math.pi)
        return -0.5 * (z_flat.pow(2).sum(dim=1) + const)

    def compute_elbo(
        self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        # Gaussian log-likelihood reconstruction term
        recon = self.decoder(z)
        x_flat = x.view(x.size(0), -1)
        recon_flat = recon.view(recon.size(0), -1)
        recon_err = (x_flat - recon_flat).pow(2).sum(dim=1)
        const = x_flat.size(1) * math.log(2.0 * math.pi)
        log_px_z = -0.5 * (recon_err + const)

        # KL divergence
        kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)

        # include beta factor (consistent with training)
        return log_px_z - self.beta * kld

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is not None:
            mask = y >= 0
            if mask.sum() == 0:
                return torch.empty((0,), device=x.device)
            x = x[mask]

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if self.use_sampling else mu

        if self.mode == "reconstruction":
            recon = self.decoder(z)
            err = F.mse_loss(recon, x, reduction="none").view(x.size(0), -1).mean(dim=1)
            score = -err
        elif self.mode == "elbo":
            score = self.compute_elbo(x, mu, logvar, z)
        elif self.mode == "prior":
            score = self.compute_prior_log_density(z)
        elif self.mode == "kld":
            kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)
            score = -kld
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. Choose 'reconstruction', 'elbo', 'prior', or 'kld'."
            )

        return self.map_fn(score)

    def _training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            x = batch[0]
            y = batch[1]
            if y is not None:
                mask = y >= 0.0
                if mask.sum() > 0:
                    x = x[mask]
                else:
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            x = batch

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        # Gaussian log-likelihood reconstruction term
        x_flat = x.view(x.size(0), -1)
        recon_flat = recon.view(recon.size(0), -1)
        recon_err = (x_flat - recon_flat).pow(2).sum(dim=1)
        log_px_z = -0.5 * (recon_err + x_flat.size(1) * math.log(2 * math.pi))

        # KL divergence
        kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)

        # negative ELBO (to minimize)
        loss = -(log_px_z - self.beta * kld).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            x = batch[0]
            y = batch[1]
            if y is not None:
                mask = y >= 0.0
                if mask.sum() > 0:
                    x = x[mask]
                else:
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            x = batch

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        x_flat = x.view(x.size(0), -1)
        recon_flat = recon.view(recon.size(0), -1)
        recon_err = (x_flat - recon_flat).pow(2).sum(dim=1)
        log_px_z = -0.5 * (recon_err + x_flat.size(1) * math.log(2 * math.pi))

        kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)
        loss = -(log_px_z - self.beta * kld).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type is None:
            return torch.optim.Adam(self.parameters(), lr=1e-3)
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)
