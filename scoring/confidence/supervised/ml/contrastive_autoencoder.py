from typing import Optional, Callable, Dict, Any, Any, Union, Literal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from confidence.input_transform import InputTransform
from confidence.unsupervised.ml.autoencoder import BasicAutoencoderConfidence
from confidence.unsupervised.unsupervised_base import MLConfidenceBase

class ContrastiveAutoencoder(BasicAutoencoderConfidence):
    """
    ContrastiveAutoencoder with stable center estimation, EMA updates,
    and tunable tradeoff between contrastive and reconstruction losses.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        input_transform: Optional[InputTransform] = None,
        mode: str = 'reconstruction',
        margin: float = 1.0,
        rec_weight: float = 1.0,
        ema_momentum: float = 0.0,
        map_fn: Callable[[torch.Tensor], torch.Tensor] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        distance_type: Literal['l2', 'cosine'] = 'l2',
            negative_sampling_module=None,

    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder, 
            input_transform=input_transform,
            map_fn=map_fn,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            negative_sampling_module=negative_sampling_module

        )
        self.mode = mode
        self.margin = margin
        self.rec_weight = rec_weight
        self.ema_momentum = ema_momentum
        self.center_: Optional[torch.Tensor] = None
        self.distance_type = distance_type

    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between tensors x and y based on the chosen distance metric.
        
        Args:
            x: First tensor
            y: Second tensor (or center)
            
        Returns:
            Tensor of distances (higher values = more distant)
        """
        if self.distance_type == 'l2':
            # L2 squared distance
            return ((x - y) ** 2).sum(dim=1)
        elif self.distance_type == 'cosine':
            # Cosine distance (1 - cosine similarity)
            # Normalize vectors for cosine similarity
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            # 1 - cos_sim ranges from 0 (identical) to 2 (opposite)
            return 1 - torch.sum(x_norm * y_norm, dim=1)
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

    def compute_reconstruction_distance(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction distance based on the chosen distance metric.

        Args:
            x: Original input tensor
            recon: Reconstructed tensor

        Returns:
            Tensor of reconstruction distances (higher values = worse reconstruction)
        """
        if self.distance_type == 'l2':
            # MSE per sample 1 till end
            mean_dim = tuple(range(1, x.dim())) if x.dim() > 2 else 1
            return F.mse_loss(recon, x, reduction='none').mean(dim=mean_dim)
        elif self.distance_type == 'cosine':
            # Flatten tensors if they have more than 2 dimensions
            if x.dim() > 2:
                x_flat = x.reshape(x.size(0), -1)
                recon_flat = recon.reshape(recon.size(0), -1)
            else:
                x_flat = x
                recon_flat = recon

            # Normalize vectors for cosine similarity
            x_norm = F.normalize(x_flat, p=2, dim=1)
            recon_norm = F.normalize(recon_flat, p=2, dim=1)

            # 1 - cos_sim ranges from 0 (identical) to 2 (opposite)
            return 1 - torch.sum(x_norm * recon_norm, dim=1)
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

    def contrastive_losses(self, scores: torch.Tensor, y: torch.Tensor):
        """Compute positive and negative contrastive terms."""
        #-1 for OOD, y>=0 for in-distribution
        yf = (y >= 0).float()  # Convert to float tensor for calculations
        pos_loss = (scores * yf).sum() / (yf.sum() + 1e-6)
        neg_term = F.relu(self.margin - scores)
        neg_loss = (neg_term * (1 - yf)).sum() / ((1 - yf).sum() + 1e-6)
        return pos_loss, neg_loss

    def reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor):
        """Per‐batch reconstruction loss using the specified distance metric."""
        rec_err = self.compute_reconstruction_distance(x, recon)
        return rec_err.mean()

    def _training_step(self, batch, batch_idx):
        x, y = batch
        mask = y >= 0  # in-distribution mask
        z = self.encoder(x)

        # Optionally EMA‐update center using positives
        if mask.any() and self.ema_momentum > 0:
            batch_center = z[mask].mean(dim=0).detach()
            self.center_ = (self.ema_momentum * self.center_ +
                             (1 - self.ema_momentum) * batch_center)

        logs = {}
        if self.mode == 'reconstruction':
            recon = self.decoder(z)
            err = self.compute_reconstruction_distance(x, recon)
            pos_loss, neg_loss = self.contrastive_losses(err, y)
            loss = pos_loss + neg_loss

            logs.update(pos_loss=pos_loss, neg_loss=neg_loss)

        else:  # latent mode
            dist = self.compute_distance(z, self.center_)
            pos_loss, neg_loss = self.contrastive_losses(dist, y)

            recon = self.decoder(z[mask])
            rec_loss = self.reconstruction_loss(x[mask], recon)

            loss = pos_loss + neg_loss + rec_loss
            logs.update(pos_loss=pos_loss, neg_loss=neg_loss, rec_loss=rec_loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _forward(self, x: torch.Tensor,y=None) -> torch.Tensor:
        """
        Compute confidence for inputs x.
        In 'reconstruction' mode, uses reconstruction error with selected distance metric.
        In 'latent' mode, uses distance from the learned center.
        Finally, map the score to a confidence value via map_fn.
        """


        # encode
        z = self.encoder(x)

        if self.mode == 'reconstruction':
            # decode and compute per-sample distance using selected metric
            recon = self.decoder(z)
            score = self.compute_reconstruction_distance(x, recon)
        else:
            # latent distance to center using the specified distance metric
            score = self.compute_distance(z, self.center_)

        # higher score → lower confidence
        return self.map_fn(score)


    def fit(self, data, y=None):
        # build loader of (x, label) with in‐dist=1, out‐dist=0
        if isinstance(data, torch.Tensor):
            X = data
            loader = DataLoader(TensorDataset(X,y),
                                **self.dataloader_kwargs)
        else:
            loader = data

        self.encoder.eval()
        zs = []
        with torch.no_grad():
            for x, label in loader:
                x.to(self.device)
                mask = (label >= 0).to(self.device)
                if mask.any():
                    x = x.to(self.device)[mask]
                    if self.feature_extractor:
                        images = x
                        x= self.feature_extractor(images)
                    if self.input_transform:
                        x = self.input_transform.transform(x)
                    zs.append(self.encoder(x))
        self.center_ = torch.cat(zs, 0).mean(dim=0,keepdim=True).detach()
        self.encoder.train()
        return super().fit(data, y=y)
    
    
    def configure_optimizers(self):
        """
        Configure optimizer for the encoder and decoder.
        Uses Adam with the specified learning rate.
        """
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)
    
    
class OODContrastiveAutoencoderConfidence(MLConfidenceBase):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        margin: float = 1.0,
        rec_weight: float = 1.0,
        cont_weight: float = 1.0,
        map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_transform=None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        distance_type: Literal['l2', 'cosine'] = 'l2'
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
        self.margin = margin
        self.rec_weight = rec_weight
        self.cont_weight = cont_weight
        self.map_fn = map_fn or (lambda d: 1.0/(1.0+d))
        self.center_: Optional[torch.Tensor] = None
        self.distance_type = distance_type

    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between tensors x and y based on the chosen distance metric.
        
        Args:
            x: First tensor
            y: Second tensor (or center)
            
        Returns:
            Tensor of distances (higher values = more distant)
        """
        if self.distance_type == 'l2':
            # L2 squared distance
            return ((x - y) ** 2).sum(dim=1)
        elif self.distance_type == 'cosine':
            # Cosine distance (1 - cosine similarity)
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            return 1 - torch.sum(x_norm * y_norm, dim=1)
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

    def fit(self, data, y=None):
        # build loader of (x, label) with in‐dist=1, out‐dist=0
        if isinstance(data, torch.Tensor):
            X = data
            if self.input_transform:
                self.input_transform.fit(X, y)
                X = self.input_transform(X)
            loader = DataLoader(TensorDataset(X, torch.ones(len(X))),
                                **self.dataloader_kwargs)
        else:
            loader = data

        self.encoder.eval()
        zs = []
        with torch.no_grad():
            for x, label in loader:
                # Create mask where label >= 0 indicates in-distribution samples
                mask = (label >= 0).to(self.device)
                if mask.any():
                    x = x.to(self.device)
                    if self.input_transform:
                        x = self.input_transform.transform(x)
                    zs.append(self.encoder(x)[mask])
        self.center_ = torch.cat(zs, 0).mean(dim=0,keepdim=True).detach()
        self.encoder.train()
        return super().fit(data, y)

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, label = batch
        # Create in-distribution mask where label >= 0
        in_dist_mask = (label >= 0)
        z = self.encoder(x)
        dist = self.compute_distance(z, self.center_)
        # Use in-distribution mask for positive samples
        pos = dist[in_dist_mask].mean()
        # Use negation of in-distribution mask for OOD samples
        neg = F.relu(self.margin - dist)[~in_dist_mask].mean()
        # Only reconstruct in-distribution samples
        recon = self.decoder(z[in_dist_mask])
        rec = F.mse_loss(recon, x[in_dist_mask], reduction='mean')
        loss = self.cont_weight*(pos+neg) + self.rec_weight*rec
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        z = self.encoder(x)
        dist = self.compute_distance(z, self.center_)
        return self.map_fn(dist)

    def configure_optimizers(self):
        """
        Configure optimizer for the encoder and decoder.
        Uses Adam with the specified learning rate.
        """
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)


import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform

class ValuePredictionConfidence(MLConfidenceBase):
    """
    Predicts a scalar target y from x using MSE or MAE.
    """
    def __init__(
        self,
        value_model: torch.nn.Module,
        loss_type: str = 'mse',  # 'mse' or 'mae'
        input_transform: Optional[InputTransform] = None,
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
            negative_sampling_module=None,
        )
        if loss_type not in ('mse', 'mae'):
            raise ValueError("loss_type must be 'mse' or 'mae'")
        self.value_model = value_model
        self.loss_type = loss_type

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)

    def _training_step(self, batch, batch_idx):
        x, y = batch  # y: scalar value
        if self.input_transform:
            x = self.input_transform(x)
        pred = self.value_model(x).squeeze(-1)
        y = y.view_as(pred).to(pred.dtype)
        loss = self._compute_loss(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _validation_step(self, batch, batch_idx):
        x, y = batch
        if self.input_transform:
            x = self.input_transform(x)
        pred = self.value_model(x).squeeze(-1)
        y = y.view_as(pred).to(pred.dtype)
        loss = self._compute_loss(pred, y)
        mae = F.l1_loss(pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        if self.input_transform:
            x = self.input_transform(x)
        return self.value_model(x).squeeze(-1)

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)


import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from confidence.input_transform import InputTransform

class ClassWithOODConfidence(MLConfidenceBase):
    """
    Treats y >= 0 as in-distribution class indices in [0, num_classes-1],
    and any y < 0 as an extra OOD class (index = num_classes).
    Uses standard cross-entropy classification.
    """
    def __init__(
        self,
        classifier_model: torch.nn.Module,
        num_classes: int,
        input_transform: Optional[InputTransform] = None,
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
            negative_sampling_module=None,
        )
        self.classifier_model = classifier_model
        self.num_id_classes = num_classes  # without OOD
        self.total_classes = num_classes + 1  # include OOD
        self.ood_index = num_classes

    def _prepare_targets(self, y: torch.Tensor) -> torch.Tensor:
        y = y.clone()
        # map any negative to ood_index
        y[y < 0] = self.ood_index
        # sanity: any positive must be < num_id_classes
        if (y >= self.total_classes).any():
            raise ValueError("Label out of range for provided num_classes.")
        return y.long()

    def _training_step(self, batch, batch_idx):
        x, y = batch
        if self.input_transform:
            x = self.input_transform(x)
        logits = self.classifier_model(x)
        if logits.size(-1) != self.total_classes:
            raise ValueError(f"Model must output {self.total_classes} logits, got {logits.size(-1)}.")
        targets = self._prepare_targets(y)
        loss = F.cross_entropy(logits, targets)
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _validation_step(self, batch, batch_idx):
        x, y = batch
        if self.input_transform:
            x = self.input_transform(x)
        logits = self.classifier_model(x)
        targets = self._prepare_targets(y)
        loss = F.cross_entropy(logits, targets)
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        # Track separation: mean max prob ID vs OOD
        probs = logits.softmax(dim=-1)
        id_mask = targets != self.ood_index
        ood_mask = targets == self.ood_index
        id_conf = probs[id_mask].max(dim=-1).mean() if id_mask.any() else torch.tensor(0.0, device=logits.device)
        ood_conf = probs[ood_mask].max(dim=-1).mean() if ood_mask.any() else torch.tensor(0.0, device=logits.device)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_id_conf', id_conf, on_step=False, on_epoch=True)
        self.log('val_ood_conf', ood_conf, on_step=False, on_epoch=True)
        return loss

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        if self.input_transform:
            x = self.input_transform(x)
        return self.classifier_model(x)  # return raw logits

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)