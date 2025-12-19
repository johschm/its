import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import pytorch_lightning as pl
from typing import Optional, List, Tuple, Literal
import random
from sklearn.decomposition import PCA


class SamplingStrategyLatent:
    """Abstract base class for negative sampling strategies."""
    def __init__(self, neg_margin: float = 0.1):
        self.neg_margin = neg_margin

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        """Generate negative samples based on X_pos (batch).
        
        Args:
            X_pos: Tensor of shape [batch_size, dim]
            
        Returns:
            Tensor of shape [batch_size, dim] with negative samples
        """
        raise NotImplementedError


class GaussianSamplingStrategyLatent(SamplingStrategyLatent):
    def __init__(self, neg_margin: float = 0.1):
        super().__init__(neg_margin)

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n, d = X_pos.shape
        mu = X_pos.mean(0)
        sigma = X_pos.std(0) + 1e-6
        sigma = sigma * (1 + self.neg_margin)
        return torch.randn(n, d, device=X_pos.device) * sigma + mu


class UniformSamplingStrategyLatent(SamplingStrategyLatent):
    def __init__(self, neg_margin: float = 0.1):
        super().__init__(neg_margin)

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n, d = X_pos.shape
        mins, maxs = X_pos.min(0).values, X_pos.max(0).values
        span = maxs - mins
        low = mins - self.neg_margin * span
        high = maxs + self.neg_margin * span
        return torch.rand(n, d, device=X_pos.device) * (high - low) + low


class AlphaSamplingStrategyLatent(SamplingStrategyLatent):
    """Generate negatives by extrapolation between random positive pairs using alpha outside [0 - neg_margin, 1 + neg_margin]."""
    def __init__(self, neg_margin: float = 0.1):
        super().__init__(neg_margin)

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n_pos, d = X_pos.shape
        n = n_pos  # Same number of negatives as positives
        m = self.neg_margin
        low1, high1 = -(1 + m), -m
        low2, high2 = 1 + m, 1 + 2 * m
        X_neg = torch.zeros(n, d, device=X_pos.device)
        for i in range(n):
            i1, i2 = random.sample(range(n_pos), 2)
            x1, x2 = X_pos[i1], X_pos[i2]
            if random.random() < 0.5:
                alpha = random.uniform(low1, high1)
            else:
                alpha = random.uniform(low2, high2)
            X_neg[i] = x1 + alpha * (x1 - x2)
        return X_neg


class PCABasedStrategyLatent(SamplingStrategyLatent):
    """Generate negatives by sampling in PCA-reduced space and projecting back."""
    def __init__(self, X_pos: torch.Tensor, neg_margin: float = 0.1,
                 n_components: int = 10):
        super().__init__(neg_margin)
        # PCA needs to be fit on full dataset, not batches
        xp = X_pos.cpu().numpy()
        self.pca = PCA(n_components=min(n_components, xp.shape[1]))
        self.pca.fit(xp)
        transformed = torch.from_numpy(self.pca.transform(xp)).to(X_pos.device)
        mins, maxs = transformed.min(0).values, transformed.max(0).values
        span = maxs - mins
        self.low = mins - neg_margin * span
        self.high = maxs + neg_margin * span

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n = X_pos.size(0)
        z = torch.rand(n, self.low.size(0), device=X_pos.device) * (self.high - self.low) + self.low
        z_cpu = z.cpu().numpy()
        x_neg = self.pca.inverse_transform(z_cpu)
        return torch.from_numpy(x_neg).to(X_pos.device)


class AutoencoderSamplingStrategyLatent(SamplingStrategyLatent):
    """Generate negatives by sampling in autoencoder latent space and projecting back."""
    def __init__(self, X_pos: torch.Tensor, autoencoder, neg_margin: float = 0.1):
        super().__init__(neg_margin)
        self.autoencoder = autoencoder
        # Compute statistics on full dataset
        with torch.no_grad():
            z_pos = self.autoencoder.encode(X_pos)
        self.mu_z = z_pos.mean(0)
        sigma_z = z_pos.std(0) + 1e-6
        self.sigma_z = sigma_z * (1 + neg_margin)

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n = X_pos.size(0)
        z = torch.randn(n, self.mu_z.size(0), device=X_pos.device) * self.sigma_z + self.mu_z
        return self.autoencoder.decode(z)


class RemoveClosePointsStrategyLatent(SamplingStrategyLatent):
    """Generate random uniform points and remove those within a threshold distance from positives."""
    def __init__(self, X_pos: torch.Tensor, neg_margin: float = 0.1,
                 threshold: float = 0.5):
        super().__init__(neg_margin)
        # Compute bounds on full dataset
        mins, maxs = X_pos.min(0).values, X_pos.max(0).values
        span = maxs - mins
        self.low = mins - neg_margin * span
        self.high = maxs + neg_margin * span
        self.threshold = threshold
        # Store for distance checks
        self.ref_points = X_pos

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n, d = X_pos.shape
        X_neg = []
        while len(X_neg) < n:
            candidate = torch.rand(1, d, device=X_pos.device) * (self.high - self.low) + self.low
            # Check distance against all reference points
            dist = torch.cdist(candidate, self.ref_points)
            if dist.min() > self.threshold:
                X_neg.append(candidate.squeeze(0))
        return torch.stack(X_neg, 0)


class VAESamplingStrategyLatent(SamplingStrategyLatent):
    """Generate negatives using a trained VAE over embeddings."""
    def __init__(self, vae_model: 'VAE', neg_margin: float = 0.1,
                 use_margin_in_latent: bool = True):
        super().__init__(neg_margin)
        self.vae = vae_model
        self.use_margin_in_latent = use_margin_in_latent

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n = X_pos.size(0)
        z_dim = self.vae.latent_dim
        if not self.use_margin_in_latent:
            sigma = 1 + self.neg_margin
            z = torch.randn(n, z_dim, device=X_pos.device) * sigma
        else:
            z = torch.randn(n, z_dim, device=X_pos.device)
            #move z away by moving self.neg_margin in direction of z
            z_norm = torch.normalize(z, dim=1)
            z = z + self.neg_margin * z_norm
        return self.vae.decode(z)


class FlowSamplingStrategyLatent(SamplingStrategyLatent):
    """Generate negatives using a trained Normalizing Flow model."""
    def __init__(self, flow_model, neg_margin: float = 0.1,
                 use_margin_in_latent: bool = True):
        super().__init__(neg_margin)
        self.flow = flow_model
        self.use_margin_in_latent = use_margin_in_latent

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n = X_pos.size(0)
        z_dim = self.flow.latent_dim
        if not self.use_margin_in_latent:
            sigma = 1 + self.neg_margin
            z = torch.randn(n, z_dim, device=X_pos.device) * sigma
        else:
            z = torch.randn(n, z_dim, device=X_pos.device)
            z_norm = torch.normalize(z, dim=1)
            z = z + self.neg_margin * z_norm
        x_neg, _ = self.flow.reverse(z)
        return x_neg


class NoisySamplingStrategyLatent(SamplingStrategyLatent):
    """Generate negatives by adding Gaussian noise to sampled positive embeddings."""
    def __init__(self, noise_std: float = 0.1):
        super().__init__(neg_margin=0.0)  # neg_margin unused here
        self.noise_std = noise_std

    def sample(self, X_pos: torch.Tensor) -> torch.Tensor:
        n, d = X_pos.shape
        # Pick random positives from the batch (with replacement)
        idx = torch.randint(0, n, (n,), device=X_pos.device)
        X_sel = X_pos[idx]
        noise = torch.randn(n, d, device=X_pos.device) * self.noise_std
        return X_sel + noise


class VAE(pl.LightningModule):
    """Variational Autoencoder for embeddings."""
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: Optional[List[int]] = None, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        hidden_dims = hidden_dims or [128, 64]
        self.latent_dim = latent_dim
        modules = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(prev_dim, h_dim))
            modules.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder_net = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        hidden_dims.reverse()
        modules = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(prev_dim, h_dim))
            modules.append(nn.ReLU())
            prev_dim = h_dim
        modules.append(nn.Linear(hidden_dims[-1], input_dim))
        self.decoder_net = nn.Sequential(*modules)
        self.lr = lr

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss_function(self, x, x_recon, mu, log_var):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mu, log_var = self(x)
        loss = self.loss_function(x, x_recon, mu, log_var)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


import torch
from torch import nn

class TransformLatentSamplingStrategy:
    """
    Samples negatives by applying random transforms_old in image space and
    extracting latent features with a model (no gradients).
    """
    def __init__(
        self,
        transform_sequence,
        feature_extractor: nn.Module=nn.Identity(),
        neg_margin: float = 0.0,
        index=None,
        clip_data: bool = False,
        mode: Literal["default", "double", "double_resampled"] = "default",

    ):
        self.transform_sequence = transform_sequence
        self.feature_extractor = feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.index = index
        self.clip_data = clip_data
        self.mode = mode

    def sample(self, X_pos: torch.Tensor,seed=None) -> torch.Tensor:
        return self.sample_and_params(X_pos,seed=seed)[0]

    def sample_and_params(self, X_pos: torch.Tensor,seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        batch_size = X_pos.size(0)

        params1 = self.transform_sequence.initial_param(batch_size)
        T1 = self.transform_sequence(params1)


        if self.mode == "double":
            # second transform
            params2 = self.transform_sequence.initial_param(batch_size)
            T2 = self.transform_sequence(params2)

            # combined transform
            T_combined = torch.bmm(T2, T1)

            # apply combined transform ONCE (equivalent for affine)
            X_out = self.transform_sequence.application_method(X_pos, T_combined)

            if self.clip_data and X_out.dim() == 4:
                X_out = torch.clamp(X_out, 0.0, 1.0)

            if seed is not None:
                torch.set_rng_state(cpu_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_state)


            return X_out, T_combined

        elif self.mode == "double_resampled":
            # --- First transform ---


            # Apply first transform
            X1 = self.transform_sequence.application_method(X_pos, T1)


            # --- Second transform, resampled ---
            params2 = self.transform_sequence.initial_param(batch_size)
            T2 = self.transform_sequence(params2)

            # Apply second transform on already-transformed image
            X2 = self.transform_sequence.application_method(X1, T2)

            # Combined transform T2 ∘ T1
            T_combined = torch.bmm(T2, T1)

            if self.clip_data and X2.dim() == 4:
                X2 = torch.clamp(X2, 0.0, 1.0)

            if seed is not None:
                torch.set_rng_state(cpu_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_state)

            return X2, T_combined


        # Apply first transform
        X1 = self.transform_sequence.application_method(X_pos, T1)


        # --- default mode (single transform) ---
        X_trans = X1
        T = T1

        if self.clip_data and X_trans.dim() == 4:
            X_trans = torch.clamp(X_trans, 0.0, 1.0)

        if seed is not None:
            torch.set_rng_state(cpu_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
        return X_trans, T

    def get_identity_transform(self,batch_size) -> torch.Tensor:
        params = self.transform_sequence.get_identity_parameters(batch_size)
        #old_version_params = torch.zeros_like(params)
        #T_old = self.transform_sequence(old_version_params)


        T = self.transform_sequence(params)
        #assert torch.allclose(T,T_old), "Transform implementations do not match!"
        #assert this to be identity matrix
        assert torch.allclose(T, torch.eye(T.size(-1), device=T.device,dtype=T.dtype).unsqueeze(0).expand(batch_size, -1, -1)), "Identity transform is not identity matrix!"

        return T

    def identity_params(self, batch_size) -> torch.Tensor:
        return self.transform_sequence.get_identity_parameters(batch_size)

    # --- NEW: alias used by BatchNegativeSampler ---
    def identity_transform(self, batch_size) -> torch.Tensor:
        return self.get_identity_transform(batch_size)
    # --- END NEW ---