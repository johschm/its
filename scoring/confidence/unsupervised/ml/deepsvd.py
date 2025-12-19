import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from typing import Optional, Callable, Dict, Any, Literal


def get_radius(distances: torch.Tensor, nu: float) -> float:
    """Compute R as the (1-nu)-quantile of sqrt(distances)."""
    sqrt_distances = torch.sqrt(distances)
    return float(torch.quantile(sqrt_distances, 1 - nu).item())


class DeepSVDDConfidence(MLConfidenceBase):
    def __init__(
        self,
        encoder: torch.nn.Module,
        nu: float = 0.1,
        objective: Literal["one-class", "soft-boundary"] = "one-class",
        weight_decay: float = 1e-6,
        map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform=None,
        trainer_kwargs=None,
        dataloader_kwargs=None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        warm_up_n_epochs: int = 10,
        learnable_radius: bool = False
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type if optimizer_type is not None else torch.optim.AdamW,
            optimizer_kwargs=optimizer_kwargs
        )
        self.save_hyperparameters(
            ignore=[
                "encoder",
                "input_transform",
                "map_fn",
                "trainer_kwargs",
                "dataloader_kwargs",
            ]
        )

        print("bias terms are not recommended for DeepSVDD, this is not checked here. Similary unbounded activations are prefered and weight decay/reg to avoid trivial solutions.")

        self.nu = nu
        self.objective = objective
        self.weight_decay = weight_decay
        self.encoder = encoder
        self.map_fn = map_fn or (lambda score: 1.0 / (1.0 + score))
        self.center_: Optional[torch.Tensor] = None
        self.warm_up_n_epochs = warm_up_n_epochs

        if self.objective == "soft-boundary":
            # non-learnable radius buffer, initialized in fit()
            self.R = torch.nn.Parameter(
                torch.tensor(0.0, device=self.device),
                requires_grad= learnable_radius
            )

    def fit(self, data, y=None):
        # build loader exactly as MLConfidenceBase does
        if isinstance(data, torch.Tensor):
            X = data.to(self.device)
            dataset = TensorDataset(X)
            loader = DataLoader(
                dataset,
                **self.dataloader_kwargs,
            )
        elif isinstance(data, np.ndarray):
            X = torch.from_numpy(data).to(self.device)
            dataset = TensorDataset(X)
            loader = DataLoader(
                dataset,
                **self.dataloader_kwargs,
            )
        elif isinstance(data, DataLoader):
            loader = data
        else:
            raise TypeError(f"Expected Tensor or DataLoader, got {type(data)}")

        # initialize center
        self.encoder.eval()
        zs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                x = self.input_transform.transform(x) if self.input_transform else x
                z = self.encoder(x)
                zs.append(z)
        zs_cat = torch.cat(zs, dim=0)
        self.center_ = zs_cat.mean(dim=0,keepdim=True).detach()
        self.encoder.train()

        # initialize radius if soft-boundary
        if self.objective == "soft-boundary":
            dist2 = ((zs_cat - self.center_) ** 2).sum(dim=1)
            R_init = get_radius(dist2, self.nu)
            # set buffer to initial radius
            self.R.data = torch.tensor(R_init, device=self.device)

        return super().fit(data, y)

    def _training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
        z = self.encoder(x)
        dist2 = ((z - self.center_) ** 2).sum(dim=1)

        # Filter out OOD samples (y < 0) if y is available
        if y is not None:
            mask = y >= 0.0
            if mask.sum() > 0:  # Check if any samples remain
                dist2 = dist2[mask]

        if self.objective == "one-class":
            loss = dist2.mean()
        else:
            if y is not None and mask.sum() > 0:
                slack = F.relu(dist2 - self.R ** 2)
            else:
                slack = F.relu(dist2 - self.R ** 2)
            loss = self.R ** 2 + (1.0 / self.nu) * slack.mean()

        # update R after warm-up epochs (batch-wise computation)
        if (self.objective == "soft-boundary") and (self.current_epoch >= self.warm_up_n_epochs) and not torch.requires_grad(self.R):
            # Use filtered dist2 if y is available
            dist2_for_radius = dist2 if y is None or mask.sum() == 0 else dist2
            new_R = get_radius(dist2_for_radius.detach(), self.nu)
            self.R.data = torch.tensor(new_R, device=self.device)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _forward(self, x, y=None):
        z = self.encoder(x)
        dist2 = ((z - self.center_) ** 2).sum(dim=1)
        score = dist2 if self.objective == "one-class" else F.relu(dist2 - self.R ** 2)
        return self.map_fn(score)

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if name == "R":
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = self.optimizer_type(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            **self.optimizer_kwargs
        )
        return optimizer
