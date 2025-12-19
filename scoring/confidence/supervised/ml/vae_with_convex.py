import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Callable, Dict, Optional

from confidence.unsupervised.unsupervised_base import MLConfidenceBase
from experiment_thesis.dataset_preperation.basic_networks import DownsampleICNN


# -------------------------
# VAE wrapper
# -------------------------
class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def elbo(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decoder(z)

        # reconstruction
        recon_logprob = F.mse_loss(x_logits, x, reduction='none').sum(dim=(1, 2, 3))
        recon_logprob = -recon_logprob  # convert to log probability

        # KL divergence
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        return recon_logprob - kl  # per-sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#https://github.com/Subhadip-1/data_driven_convex_regularization/blob/main/convex_models.py
class ICNN(nn.Module):
    def __init__(self, n_in_channels=1, n_filters=48, kernel_size=5, n_layers=10):
        super(ICNN, self).__init__()
        self.n_layers = n_layers
        # these layers should have non-negative weights
        self.wz = nn.ModuleList(
            [nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding="same", bias=False) \
             for i in range(self.n_layers)])

        # these layers can have arbitrary weights
        self.wx = nn.ModuleList(
            [nn.Conv2d(n_in_channels, n_filters, kernel_size=kernel_size, stride=1, padding="same", bias=True) \
             for i in range(self.n_layers + 1)])

        # one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(n_filters, 1, kernel_size=kernel_size, stride=1, padding="same", bias=False)

        # slope of leaky-relu
        self.negative_slope = 0.2
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.initialize_weights(min_val=0.0, max_val=0.001)


    def forward(self, x):
        z = torch.nn.functional.leaky_relu(self.wx[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx[layer + 1](x),
                                               negative_slope=self.negative_slope)
        z = self.final_conv2d(z)
        z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)

        return z_avg

    # a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001, device=device):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val) \
                                         * torch.rand(self.n_filters, self.n_filters, self.kernel_size, self.kernel_size).to(device)

        self.final_conv2d.weight.data = min_val + (max_val - min_val) \
                                        * torch.rand(1,self.n_filters, self.kernel_size, self.kernel_size).to(device)
        return self

    # a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip(self):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)

        self.final_conv2d.weight.data.clamp_(0)
        return self
#test convexity
import numpy as np
def tst_convexity(net, x, device=device):
        # check convexity of the net numerically
        print('running a numerical convexity test...')
        n_trials = 100
        convexity = 0
        for trial in np.arange(n_trials):
            x1 = torch.rand(x.size()).to(device)
            x2 = torch.rand(x.size()).to(device)
            alpha = torch.rand(1).to(device)

            cvx_combo_of_input = net(alpha * x1 + (1 - alpha) * x2)
            cvx_combo_of_output = alpha * net(x1) + (1 - alpha) * net(x2)

            convexity += (cvx_combo_of_input.mean() <= cvx_combo_of_output.mean())
        if (convexity == n_trials):
            flag = True
            print('Passed convexity test!')
        else:
            flag = False
            print('Failed convexity test!')
        return flag


    # -------------------------
# Lightning module
# -------------------------
class VAEDiscriminatorConfidence(MLConfidenceBase):
    def __init__(
        self,
        vae: VAE,
        regularizer: torch.nn.Module,
        negative_sampling_module: Optional[Callable] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        gp_weight: float = 1.0,
        beta=100.0,
        reg_loss_weight=100.0,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,

    ):
        super().__init__(trainer_kwargs= trainer_kwargs,
            dataloader_kwargs= dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            negative_sampling_module=negative_sampling_module
        )
        self.vae = vae
        self.regularizer = regularizer
        self.gp_weight = gp_weight
        self.map_func = lambda x: x
        self.beta = beta  # scaling factor for regularizer
        self.reg_loss_weight = reg_loss_weight  # scaling factor for regularizer loss


    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x_both, y_both = batch

        #filter true ones with label 0 or higher and ood ones with negative label
        x = x_both[y_both >= 0].to(self.device)



        # --- Step 1: VAE ELBO loss ---
        elbo = self.vae.elbo(x)  # shape [B]
        vae_loss = -elbo.mean()



        v= x_both[y_both < 0].to(self.device)
        #assert that v is not empty
        if v.size(0) == 0:
            raise ValueError("No negative samples found in batch. Ensure batch contains both in-distribution and OOD samples.")

        # Interpolate
        eps = torch.rand(x.size(0), device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        z = eps * x + (1 - eps) * v
        z.requires_grad_(True)

        R_x = self.regularizer(x)
        R_v = self.regularizer(v)
        R_z = self.regularizer(z)

        # gradient penalty
        grad = torch.autograd.grad(R_z.sum(), z, create_graph=True)[0]
        gp = ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()

        reg_loss = (R_x.mean() - R_v.mean()) + self.gp_weight * gp

        # --- Combined loss ---
        loss = vae_loss + reg_loss *self.reg_loss_weight

        self.log_dict({
            "train_loss": loss,
            "vae_loss": vae_loss,
            "reg_loss": reg_loss,
            "R_x": R_x.mean(),
            "R_v": R_v.mean(),
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def _validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x_both, y_both = batch

        #filter true ones with label 0 or higher and ood ones with negative label
        x = x_both[y_both >= 0].to(self.device)



        # --- Step 1: VAE ELBO loss ---
        elbo = self.vae.elbo(x)  # shape [B]
        vae_loss = -elbo.mean()



        v= x_both[y_both < 0].to(self.device)
        #assert that v is not empty
        if v.size(0) == 0:
            raise ValueError("No negative samples found in batch. Ensure batch contains both in-distribution and OOD samples.")

        # Interpolate
        eps = torch.rand(x.size(0), device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        z = eps * x + (1 - eps) * v
        z.requires_grad_(True)

        R_x = self.regularizer(x)
        R_v = self.regularizer(v)
        R_z = self.regularizer(z)


        # --- Combined loss ---
        loss = vae_loss

        self.log_dict({
            "val_loss": loss,
            "val_vae_loss": vae_loss,
            "val_R_x": R_x.mean(),
            "val_R_v": R_v.mean(),
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:

        elbo = self.vae.elbo(x)
        reg = self.regularizer(x).squeeze(-1)  # shape [B]
        energy = elbo - self.beta * reg  # scale regularizer
        return self.map_func(energy)

    def configure_optimizers(self):
        opt = self.optimizer_type(self.parameters(), **self.optimizer_kwargs)

        # Wrap step to include zero-clipping
        #ceck wether zero clipping is needed
        if not hasattr(self.regularizer, 'zero_clip'):
            print("Regularizer does not support zero clipping, using original optimizer step.")
            return opt

        orig_step = opt.step
        def step_with_clipping(closure=None):
            loss = orig_step(closure)
            self.regularizer.zero_clip()
            return loss

        opt.step = step_with_clipping
        return opt

if __name__ == "__main__":

    regularizer = ICNN().to(device)


    x = torch.rand(4,1,28,28).to(device)
    tst_convexity(regularizer, x, device=device)

    downsample_icnn = DownsampleICNN(n_in_channels=1,
                                     n_filters=[32, 64, 128, 256],
                                     n_layers=3,
                                     downsample_layers=[0, 1],
                                     pool_kernel=2,
                                     pool_stride=2,initial_stride=2).to(device)
    tst_convexity(downsample_icnn, x, device=device)

