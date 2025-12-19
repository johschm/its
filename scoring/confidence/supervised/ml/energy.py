import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable, Dict, Any
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable, Dict, Any
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable, Dict, Any
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform
from sklearn.metrics import roc_auc_score

class EnergyPredictionConfidence(MLConfidenceBase):
    def __init__(
        self,
        energy_model: torch.nn.Module,
        loss_type: str = 'bce',  # options: 'mse', 'bce', 'margin', 'ranking', 'contrastive', 'triplet', 'discriminator'
        margin: float = 1.0,
        input_transform: Optional[InputTransform] = None,
        negative_sampling_module: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        gp_weight: float = 5.0,  # WGAN-GP lambda
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            negative_sampling_module=negative_sampling_module,
        )
        self.energy_model = energy_model
        self.loss_type = loss_type
        self.margin = margin
        self.gp_weight = gp_weight
        self.map_func = lambda x: x  # default identity function, can be overridden
        # Flag for convex ICNN style models requiring non-negative weights
        self._has_nonneg_constraint = hasattr(self.energy_model, "zero_clip")
        print(f"Model has non-negative constraint: {self._has_nonneg_constraint}")
        self._val_e_id_list = []
        self._val_e_ood_list = []


    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch  # y: ID if >= 0, OOD if < 0
        e = self.energy_model(x).squeeze(-1)

        id_mask = y >= 0
        ood_mask = y < 0
        e_id = e[id_mask]
        e_ood = e[ood_mask]

        if e_id.numel() == 0 or e_ood.numel() == 0:
            pass

        if self.loss_type == 'mse':
            preds = torch.cat([e_id, e_ood], dim=0)
            labels = torch.cat([torch.ones_like(e_id), torch.zeros_like(e_ood)], dim=0)
            loss = F.mse_loss(preds, labels)

        elif self.loss_type == 'bce':
            logits = torch.cat([e_id, e_ood], dim=0)
            labels = torch.cat([torch.ones_like(e_id), torch.zeros_like(e_ood)], dim=0)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        elif self.loss_type == 'margin':
            criterion = torch.nn.MarginRankingLoss(margin=self.margin)
            e_id_rep = e_id.unsqueeze(1).expand(-1, e_ood.size(0)).reshape(-1)
            e_ood_rep = e_ood.repeat(e_id.size(0))
            target_rep = torch.ones_like(e_id_rep)
            loss = criterion(e_id_rep, e_ood_rep, target_rep)

        elif self.loss_type == 'triplet':
            if (y >= 0).sum() < 2 or (y < 0).sum() < 1:
                raise ValueError("Need at least two ID samples and one OOD sample for triplet loss.")
            anchor = e_id
            positive = e_id.roll(1)
            negative = e_ood.repeat(e_id.size(0))
            if positive.size(0) != anchor.size(0):
                positive = positive[:anchor.size(0)]
            if negative.size(0) != anchor.size(0):
                negative = negative[:anchor.size(0)]
            loss = F.triplet_margin_loss(
                anchor.unsqueeze(1), positive.unsqueeze(1), negative.unsqueeze(1), margin=self.margin
            )

        elif self.loss_type == 'discriminator':
            # WGAN critic (minimization form)
            if e_id.numel() == 0 or e_ood.numel() == 0:
                #self.log('train_loss', 0.0, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                return None
            loss = e_ood.mean() - e_id.mean()

            #small reg loss to penalize large energies l2
            reg_loss = 0.01 * (e_id.pow(2).mean() + e_ood.pow(2).mean())
            center_reg = 0.001 * (e_id.mean().pow(2) + e_ood.mean().pow(2))
            loss = loss + reg_loss + center_reg

            # WGAN-GP penalty
            x_id = x[id_mask]
            x_ood = x[ood_mask]
            n = min(x_id.size(0), x_ood.size(0))
            if n > 0 and self.gp_weight > 0.0:
                idx_id = torch.randperm(x_id.size(0), device=x.device)[:n]
                idx_ood = torch.randperm(x_ood.size(0), device=x.device)[:n]
                x_id_s = x_id[idx_id]
                x_ood_s = x_ood[idx_ood]

                eps = torch.rand(n, *([1] * (x.dim() - 1)), device=x.device, dtype=x.dtype)
                z = eps * x_id_s + (1.0 - eps) * x_ood_s
                z.requires_grad_(True)

                e_z = self.energy_model(z).squeeze(-1)
                grad = torch.autograd.grad(
                    outputs=e_z.sum(), inputs=z, create_graph=True, only_inputs=True
                )[0]
                gp = ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()
                loss = loss + self.gp_weight * gp
                self.log('gp', gp, on_step=True, on_epoch=True, logger=True,prog_bar=True)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'energy_id': e_id.mean(), 'energy_ood': e_ood.mean()}, on_step=True, on_epoch=True)
        return loss

    def _validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        e = self.energy_model(x).squeeze(-1)
        id_mask = y >= 0
        ood_mask = y < 0
        e_id = e[id_mask]
        e_ood = e[ood_mask]

        if self.loss_type == 'discriminator':
            loss = e_ood.mean() - e_id.mean()
        elif self.loss_type == 'mse':
            preds = torch.cat([e_id, e_ood], dim=0)
            labels = torch.cat([torch.ones_like(e_id), torch.zeros_like(e_ood)], dim=0)
            loss = F.mse_loss(preds, labels)
            # Accuracy: threshold at 0.5
            acc = ((preds >= 0.5) == (labels == 1)).float().mean()
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        elif self.loss_type == 'bce':
            logits = torch.cat([e_id, e_ood], dim=0)
            labels = torch.cat([torch.ones_like(e_id), torch.zeros_like(e_ood)], dim=0)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            # Accuracy: threshold at 0
            acc = ((logits >= 0) == (labels == 1)).float().mean()
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        elif self.loss_type == 'margin':
            criterion = torch.nn.MarginRankingLoss(margin=self.margin)
            e_id_rep = e_id.unsqueeze(1).expand(-1, e_ood.size(0)).reshape(-1)
            e_ood_rep = e_ood.repeat(e_id.size(0))
            target_rep = torch.ones_like(e_id_rep)
            loss = criterion(e_id_rep, e_ood_rep, target_rep)
            # Accuracy: e_id should be greater than e_ood
            acc = (e_id_rep > e_ood_rep).float().mean()
            self.log('val_smaller', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)



        elif self.loss_type == 'triplet':
            anchor = e_id
            positive = e_id.roll(1)
            negative = e_ood.repeat(e_id.size(0))
            if positive.size(0) != anchor.size(0):
                positive = positive[:anchor.size(0)]
            if negative.size(0) != anchor.size(0):
                negative = negative[:anchor.size(0)]
            loss = F.triplet_margin_loss(
                anchor.unsqueeze(1), positive.unsqueeze(1), negative.unsqueeze(1), margin=self.margin
            )
            # Accuracy: anchor closer to positive than negative
            acc = ((anchor - positive).abs() < (anchor - negative).abs()).float().mean()
            self.log('val_smaller', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # store energies in memory for epoch-end computation
        if self.loss_type == 'discriminator' or self.loss_type == "margin":
            self._val_e_id_list.append(e_id.detach())
            self._val_e_ood_list.append(e_ood.detach())


        return {'loss': loss, 'e_id': e_id, 'e_ood': e_ood}

    def on_validation_epoch_end(self):
        if self.loss_type not in ['discriminator', 'margin']:
            return

        # --- FIX: Handle case where validation set has no ID or no OOD ---
        if not self._val_e_id_list or not self._val_e_ood_list:
            self.log('val_acc', 0.0, prog_bar=True)  # Log 0 acc if no data
            self._val_e_id_list.clear()
            self._val_e_ood_list.clear()
            return

        # concatenate all batch energies
        all_e_id = torch.cat(self._val_e_id_list)
        all_e_ood = torch.cat(self._val_e_ood_list)

        # compute threshold-based accuracy
        threshold = (all_e_id.mean() + all_e_ood.mean()) / 2

        # --- FIX: Logic must match training goal ---
        # Training goal: e_id (LOW) and e_ood (HIGH)
        correct_id = (all_e_id > threshold).float().sum()
        correct_ood = (all_e_ood <= threshold).float().sum()  # Use >= for consistency

        acc = (correct_id + correct_ood) / (all_e_id.numel() + all_e_ood.numel())
        self.log('val_acc', acc, prog_bar=True)
        # compute AUROC
        try :
            from sklearn.metrics import roc_auc_score
            y_true = torch.cat([torch.ones_like(all_e_id), torch.zeros_like(all_e_ood)]).cpu().numpy()
            y_scores = torch.cat([all_e_id, all_e_ood]).cpu().numpy()
            auroc = roc_auc_score(y_true, y_scores)
            self.log('val_auroc', auroc, prog_bar=True)
        except Exception as e:
            print(f"Could not compute AUROC: {e}")


        # clear memory for next epoch
        self._val_e_id_list.clear()
        self._val_e_ood_list.clear()


    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        scores = self.energy_model(x).squeeze(-1)
        if self.loss_type in ['triplet', 'contrastive']:
            return self.map_func(1 - scores), None
        return self.map_func(scores), None

    def configure_optimizers(self):
        opt = self.optimizer_type(self.energy_model.parameters(), **self.optimizer_kwargs)

        if not self._has_nonneg_constraint:
            return opt

        original_step = opt.step

        def step_with_nonneg(closure=None):
            loss = original_step(closure=closure) if closure is not None else original_step()
            self.energy_model.zero_clip()
            return loss

        opt.step = step_with_nonneg
        return opt


import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Any, Dict
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform

class EnergyDisConfidence(MLConfidenceBase):
    def __init__(
        self,
        energy_model: torch.nn.Module,
        input_transform: Optional[InputTransform] = None,
        negative_sampling_module: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        gp_weight: float = 5.0,  # WGAN-GP lambda
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            negative_sampling_module=negative_sampling_module,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.energy_model = energy_model
        self.loss_type = 'discriminator'
        self.gp_weight = gp_weight
        self.map_func = lambda x: -x  # invert for scoring: higher = ID
        self._has_nonneg_constraint = hasattr(self.energy_model, "zero_clip")
        print(f"Model has non-negative constraint: {self._has_nonneg_constraint}")

        self._val_e_id_list = []
        self._val_e_ood_list = []

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        e = self.energy_model(x).squeeze(-1)

        id_mask = y >= 0
        ood_mask = y < 0
        e_id = e[id_mask]
        e_ood = e[ood_mask]

        if e_id.numel() == 0 or e_ood.numel() == 0:
            return None

        # Correct discriminator loss: want ID < OOD
        loss = e_id.mean() - e_ood.mean()

        # L2 / center regularization
        reg_loss = 0.01 * (e_id.pow(2).mean() + e_ood.pow(2).mean())
        center_reg = 0.001 * (e_id.mean().pow(2) + e_ood.mean().pow(2))
        loss = loss + reg_loss + center_reg

        # WGAN-GP
        n = min(e_id.size(0), e_ood.size(0))
        if n > 0 and self.gp_weight > 0.0:
            eps = torch.rand(n, *([1] * (x.dim() - 1)), device=x.device, dtype=x.dtype)
            x_id_s = x[id_mask][torch.randperm(e_id.size(0))[:n]]
            x_ood_s = x[ood_mask][torch.randperm(e_ood.size(0))[:n]]
            z = eps * x_id_s + (1 - eps) * x_ood_s
            z.requires_grad_(True)

            e_z = self.energy_model(z).squeeze(-1)
            grad = torch.autograd.grad(outputs=e_z.sum(), inputs=z, create_graph=True)[0]
            gp = ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()
            loss = loss + self.gp_weight * gp
            self.log('gp', gp, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'energy_id': e_id.mean(), 'energy_ood': e_ood.mean()}, on_step=True, on_epoch=True)
        return loss

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        # Return inverted energy for scoring: higher = ID
        e = self.energy_model(x).squeeze(-1)
        return self.map_func(e), None

    def configure_optimizers(self):
        opt = self.optimizer_type(self.energy_model.parameters(), **self.optimizer_kwargs)
        if not self._has_nonneg_constraint:
            return opt

        original_step = opt.step

        def step_with_nonneg(closure=None):
            loss = original_step(closure=closure) if closure is not None else original_step()
            self.energy_model.zero_clip()
            return loss

        opt.step = step_with_nonneg
        return opt

    def _validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        e = self.energy_model(x).squeeze(-1)
        id_mask = y >= 0
        ood_mask = y < 0
        e_id = e[id_mask]
        e_ood = e[ood_mask]

        # Correct discriminator loss: want ID < OOD
        loss = e_id.mean() - e_ood.mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # store energies in memory for epoch-end computation
        self._val_e_id_list.append(e_id.detach())
        self._val_e_ood_list.append(e_ood.detach())

        return {'loss': loss, 'e_id': e_id, 'e_ood': e_ood}

    def on_validation_epoch_end(self):
        # concatenate all batch energies
        all_e_id = torch.cat(self._val_e_id_list)
        all_e_ood = torch.cat(self._val_e_ood_list)
        #apply map function
        all_e_id = self.map_func(all_e_id)
        all_e_ood = self.map_func(all_e_ood)

        # compute threshold-based accuracy
        threshold = (all_e_id.mean() + all_e_ood.mean()) / 2

        correct_id = (all_e_id > threshold).float().sum()
        correct_ood = (all_e_ood <= threshold).float().sum()

        acc = (correct_id + correct_ood) / (all_e_id.numel() + all_e_ood.numel())
        self.log('val_acc', acc, prog_bar=True)

        # compute AUROC
        try :
            from sklearn.metrics import roc_auc_score
            y_true = torch.cat([torch.ones_like(all_e_id), torch.zeros_like(all_e_ood)]).cpu().numpy()
            y_scores = torch.cat([all_e_id, all_e_ood]).cpu().numpy()
            auroc = roc_auc_score(y_true, y_scores)
            self.log('val_auroc', auroc, prog_bar=True)
        except Exception as e:
            print(f"Could not compute AUROC: {e}")

        # clear memory for next epoch
        self._val_e_id_list.clear()
        self._val_e_ood_list.clear()



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform

class ValuePredictionConfidence(MLConfidenceBase):
    """
    Direct scalar prediction.
    loss_type:
      mse -> mean squared error
      mae -> mean absolute error
      bce -> binary cross entropy (expects y in {0,1}; model outputs logits)
    """
    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "mse",
        input_transform: Optional[InputTransform] = None,
        negative_sampling_module: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            negative_sampling_module=negative_sampling_module,
        )
        self.model = model
        self.loss_type = loss_type.lower()
        if self.loss_type not in {"mse", "mae", "bce"}:
            raise ValueError(f"Unsupported loss_type {loss_type}")

    def _compute_loss(self, pred, y):
        if self.loss_type == "mse":
            return F.mse_loss(pred.view_as(y), y)
        if self.loss_type == "mae":
            return F.l1_loss(pred.view_as(y), y)
        # bce
        return F.binary_cross_entropy_with_logits(pred.view_as(y), y)

    def _training_step(self, batch, batch_idx):
        x, y = batch
        if self.input_transform:
            x = self.input_transform(x)
        pred = self.model(x).squeeze(-1)
        loss = self._compute_loss(pred, y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _validation_step(self, batch, batch_idx):
        x, y = batch
        if self.input_transform:
            x = self.input_transform(x)
        pred = self.model(x).squeeze(-1)
        loss = self._compute_loss(pred, y.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.loss_type == "bce":
            with torch.no_grad():
                probs = torch.sigmoid(pred)
                acc = ((probs >= 0.5).long() == y.long()).float().mean()
                self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _forward(self, x, y=None):
        out = self.model(x).squeeze(-1)
        if self.loss_type == "bce":
            return torch.sigmoid(out), None
        return out, None

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform

class ClassWithOODConfidence(MLConfidenceBase):
    """
    Multi-class classification with an extra OOD class.
    Any target < 0 is remapped to ood_index = num_id_classes.
    """
    def __init__(
        self,
        classifier_model: nn.Module,
        num_id_classes: int,
        input_transform: Optional[InputTransform] = None,
        negative_sampling_module: Optional[Any] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            negative_sampling_module=negative_sampling_module,
        )
        self.classifier_model = classifier_model
        self.num_id = num_id_classes
        self.ood_index = num_id_classes
        self.total_classes = num_id_classes + 1
        self.label_smoothing = label_smoothing

    def _remap_targets(self, y: torch.Tensor) -> torch.Tensor:
        y_new = y.clone()
        y_new[y_new < 0] = self.ood_index
        return y_new.long()

    def _ce_loss(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0
        )

    def _training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier_model(x)
        if logits.size(-1) != self.total_classes:
            raise ValueError(f"Classifier must output {self.total_classes} logits (got {logits.size(-1)})")
        targets = self._remap_targets(y)
        loss = self._ce_loss(logits, targets)
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier_model(x)
        targets = self._remap_targets(y)
        loss = self._ce_loss(logits, targets)
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # ID vs OOD accuracy (optional)
        with torch.no_grad():
            id_mask = targets != self.ood_index
            ood_mask = ~id_mask
            if id_mask.any():
                self.log("val_id_acc", (preds[id_mask] == targets[id_mask]).float().mean(),
                         on_step=False, on_epoch=True)
            if ood_mask.any():
                self.log("val_ood_recall", (preds[ood_mask] == self.ood_index).float().mean(),
                         on_step=False, on_epoch=True)
        return loss

    def _forward(self, x, y=None):
        logits = self.classifier_model(x)
        #return in distribution score higner for in distribution samples
        ood_logit = logits[:, self.ood_index]
        return  - ood_logit

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)