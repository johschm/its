import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbSpatialTransformerClassifier(pl.LightningModule):
    """
    Probabilistic spatial transformer classifier:
     - regression_net outputs [mu, log_var]
     - samples via reparameterization
     - KL regularization + L2 on sampled params
     - Supports confidence module and boundary regularization
    """
    def __init__(self,
                 main_model,
                 localization_net,
                 localization_dim,
                 transformation_problem,
                 freeze_main=False,
                 optimizer_class=torch.optim.Adam,
                 optimizer_params={"lr": 1e-3},
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 lr_config=None,
                 pretransform=None,
                 kl_weight=1e-4,
                 l2_weight=0.001,
                 use_l2_instead_of_per_transform=False,
                 conf_module=None):
        super().__init__()
        self.main_model = main_model
        self.localization_net = localization_net
        self.transformation_problem = transformation_problem
        self.pretransform = pretransform
        self.train_main = not freeze_main
        regression_dim = transformation_problem.calc_complete_size()

        # regression head outputs mean (zero→identity) and log-variance
        self.regression_net = nn.Linear(localization_dim, regression_dim * 2)
        nn.init.zeros_(self.regression_net.weight)             # output = bias only
        bias = torch.zeros(regression_dim * 2)
        bias[regression_dim:] = -4.6                            # log_var ≈ ln(0.01)
        self.regression_net.bias.data.copy_(bias)

        # optimizer & scheduler settings
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_config = lr_config

        # regularization weights
        self.kl_weight = kl_weight
        self.l2_weight = l2_weight
        self.conf_module = conf_module
        self.use_l2_instead_of_per_transform = use_l2_instead_of_per_transform

    def reset_head(self):
        """
        Reset the regression head (reinitialize weights and bias).
        """
        nn.init.zeros_(self.regression_net.weight)
        regression_dim = self.transformation_problem.calc_complete_size()
        bias = torch.zeros(regression_dim * 2)
        bias[regression_dim:] = -4.6
        self.regression_net.bias.data.copy_(bias)

    def forward(self, x, return_stats: bool = False):
        # apply fixed pre-transform if any
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)

        # predict distribution over transform params
        feat = self.localization_net(x).flatten(1)
        stats = self.regression_net(feat)               # [B, 2*D]
        mu, log_var = stats.chunk(2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        params = mu + eps * std

        # Correct parameters to be within bounds
        #params = self.transformation_problem.correct_param(params)

        # Apply transformation to input using the TransformationProblem
        x_t = self.transformation_problem.transform(x, params)

        if self.conf_module is None:
            logits = self.main_model(x_t)
        else:
            conf, logits = self.conf_module(x_t)

        if return_stats:
            return logits, mu, log_var, params


        return logits

    def forward_with_loss(self, x, y=None):
        # apply fixed pre-transform if any
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)

        # predict distribution over transform params
        feat = self.localization_net(x).flatten(1)
        stats = self.regression_net(feat)               # [B, 2*D]
        mu, log_var = stats.chunk(2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        params = mu + eps * std

        boundary_reg = self.transformation_problem.boundary_violation(params).mean() * 1.0


        # Correct parameters to be within bounds
        #params = self.transformation_problem.correct_param(params)

        # Apply transformation to input using the TransformationProblem
        x_t = self.transformation_problem.transform(x, params)


        if self.conf_module is None:
            # Apply confidence module to transformed input
            logits = self.main_model(x_t)
            loss = F.cross_entropy(logits, y)
        else:
            # Apply confidence module to transformed input
            conf, logits = self.conf_module(x_t, y=y)
            loss = (-conf).mean(0)

        return loss, logits, mu, log_var, params, boundary_reg

    def transform_input(self, x, return_stats: bool = False):
        """
        Apply the deterministic transform given by the predicted mean (mu) to x.
        If return_stats=True, also return (mu, log_var, params=mu).
        """
        # 1. optional pretransform
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)

        # 2. predict mean & alog-variance
        feat = self.localization_net(x).flatten(1)
        stats = self.regression_net(feat)               # [B,2*D]
        mu, log_var = stats.chunk(2, dim=1)
        params = mu                                    # use mean deterministically

        # 3. Correct parameters to be within bounds
        #params = self.transformation_problem.correct_param(params)

        # 4. Apply transformation using the TransformationProblem
        x_t = self.transformation_problem.transform(x, params)

        if return_stats:
            return x_t, mu, log_var, params
        return x_t

    def training_step(self, batch, batch_idx):
        x, y = batch
        ce, logits, mu, log_var, params, boundary_reg = self.forward_with_loss(x, y)
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        if self.use_l2_instead_of_per_transform:
            boundary_reg = torch.norm(params, dim=1).mean() * 1.0

        loss = ce + self.kl_weight * kl +  + boundary_reg *self.l2_weight
        self.log("train_loss", loss)
        self.log("train_kl", kl)

        self.log("train_reg", boundary_reg)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ce, logits, mu, log_var, params, boundary_reg = self.forward_with_loss(x, y)
        ce = F.cross_entropy(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        if self.use_l2_instead_of_per_transform:
            boundary_reg = torch.norm(params, dim=1).mean() * 1.0

        loss = ce + self.kl_weight * kl + boundary_reg *self.l2_weight
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_reg", boundary_reg, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.train_main:
            opt = self.optimizer_class(self.localization_net.parameters(), **self.optimizer_params)
            opt.add_param_group({"params": self.regression_net.parameters()})
            opt.add_param_group({"params": self.main_model.parameters()})
        else:
            for p in self.main_model.parameters():
                p.requires_grad = False
            opt = self.optimizer_class(self.localization_net.parameters(), **self.optimizer_params)
            opt.add_param_group({"params": self.regression_net.parameters()})

        if self.lr_scheduler is not None:
            sched = self.lr_scheduler(opt, **(self.lr_scheduler_params or {}))
            cfg = self.lr_config or {"interval": "epoch", "monitor": "val_loss", "frequency": 1, "strict": True}
            cfg["scheduler"] = sched
            return {"optimizer": opt, "lr_scheduler": cfg}
        return opt
