import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#TODO Frankenstein build from ideas copilot, and corrections. Remove all parts that make no difference later.
from typing import Optional, Literal
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, Literal, Sequence
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMProbSpatialTransformerClassifier(pl.LightningModule):
    """
    Simplified probabilistic spatial transformer classifier with a GMM head.

    Changes from previous version:
      - `reg_type` is now a list/sequence of strings (e.g. ["selfnll", "l2"])
      - removed `min_log_var` and `max_log_var` constructor args; internal clamping constants are used instead
      - fewer hyperparameters (single kl_weight)
      - identity-initialized component means (mus)
      - supports target_param_form: 'params', 'matrix', 'matrix_inverted'
    """

    # internal numeric clamping defaults (kept inside class, not constructor args)
    _DEFAULT_MIN_LOG_VAR = -10.0
    _DEFAULT_MAX_LOG_VAR = 5.0

    def __init__(self,
                 main_model,
                 localization_net,
                 localization_dim,
                 transformation_problem,
                 num_components: int = 3,
                 freeze_main: bool = True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_params={"lr": 1e-3},
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 lr_config=None,
                 pretransform=None,
                 kl_weight: float = 1e-4,
                 l2_weight: float = 1.0,
                 use_l2_instead_of_per_transform: bool = True,
                 conf_module=None,
                 reg_type: Optional[Sequence[str]] = None,  # now accepts list/sequence
                 train_mode: float = 0.0,
                 negative_sampling_module=None,
                 clamp: bool = False,
                 init_log_var: float = -4.6,
                 target_param_form: Literal['params', 'matrix', 'matrix_inverted'] = 'params',
                 ):
        super().__init__()
        # core modules
        self.main_model = main_model
        self.localization_net = localization_net
        self.transformation_problem = transformation_problem
        self.pretransform = pretransform

        # training flags
        assert 0.0 <= train_mode <= 1.0
        self.train_mode = float(train_mode)
        self.train_main = not freeze_main

        # GMM head shape
        self.num_components = int(num_components)
        self.regression_dim = int(transformation_problem.calc_complete_size())
        gmm_output_dim = self.num_components * (1 + self.regression_dim + self.regression_dim)
        self.gmm_params_head = nn.Linear(localization_dim, gmm_output_dim)

        # optimizer / scheduler
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_config = lr_config

        # regularization settings (simplified)
        self.kl_weight = float(kl_weight)
        self.l2_weight = float(l2_weight)
        self.use_l2_instead_of_per_transform = bool(use_l2_instead_of_per_transform)
        self.conf_module = conf_module

        # reg options: accept list/sequence or None -> sensible default
        if reg_type is None:
            self.reg_options = {"selfnll", "l2"}
        else:
            # allow strings in different cases
            self.reg_options = {opt.lower() for opt in reg_type}

        # numeric stability & init
        self.init_log_var = float(init_log_var)
        # internal clamping constants (no constructor args)
        self._min_log_var = float(self._DEFAULT_MIN_LOG_VAR)
        self._max_log_var = float(self._DEFAULT_MAX_LOG_VAR)

        # other hooks
        self.negative_sampling_module = negative_sampling_module
        self.clamp = bool(clamp)

        # target param form implementation
        assert target_param_form in {"params", "matrix", "matrix_inverted"}
        self.target_param_form = target_param_form

        # initialize head near identity
        self.reset_head()

    # -------------------------
    # Initialization / controls
    # -------------------------
    def _initialize_gmm_head_defaults(self):
        """Zero weights and set biases to identity-like mus and small log_vars."""
        nn.init.zeros_(self.gmm_params_head.weight)
        device = self.gmm_params_head.weight.device
        dtype = self.gmm_params_head.weight.dtype
        K = self.num_components
        D = self.regression_dim

        bias = torch.zeros(self.gmm_params_head.out_features, device=device, dtype=dtype)

        # layout in bias: [K log_coeffs, K*D mus, K*D log_vars]
        mus_start = K
        mus_end = mus_start + K * D

        # get identity params and repeat for each component
        idp = self.transformation_problem.get_identity_parameters()  # expected shape [D] or [1,D]
        if idp.dim() > 1:
            idp = idp.view(-1)
        idp = idp[:D].detach().to(device=device, dtype=dtype)
        bias[mus_start:mus_end] = idp.repeat(K)

        # small initial log-variance
        bias[mus_end:] = float(self.init_log_var)

        with torch.no_grad():
            self.gmm_params_head.bias.copy_(bias)

    def reset_head(self):
        self._initialize_gmm_head_defaults()

    def freeze_backbone(self):
        for p in self.localization_net.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.localization_net.parameters(): p.requires_grad = True

    def freeze_main_model(self):
        for p in self.main_model.parameters(): p.requires_grad = False

    def unfreeze_main_model(self):
        for p in self.main_model.parameters(): p.requires_grad = True

    def freeze_head(self):
        for p in self.gmm_params_head.parameters(): p.requires_grad = False

    def unfreeze_head(self):
        for p in self.gmm_params_head.parameters(): p.requires_grad = True

    # -------------------------
    # GMM helpers
    # -------------------------
    def _get_gmm_params(self, x):
        feat = self.localization_net(x).flatten(1)
        gmm_stats = self.gmm_params_head(feat)
        B = x.size(0)
        K = self.num_components
        D = self.regression_dim

        log_coeffs_unnorm = gmm_stats[:, :K]                                 # [B, K]
        mus_flat = gmm_stats[:, K: K + K * D]                                # [B, K*D]
        mus = mus_flat.reshape(B, K, D)                                      # [B, K, D]
        log_vars_flat = gmm_stats[:, K + K * D:]                             # [B, K*D]
        log_vars = log_vars_flat.reshape(B, K, D)                            # [B, K, D]

        # clamp log_vars using internal defaults for numeric stability
        log_vars = torch.clamp(log_vars, min=self._min_log_var, max=self._max_log_var)
        return log_coeffs_unnorm, mus, log_vars

    def _sample_from_gmm(self, log_coeffs_unnorm, mus, log_vars):
        coeffs = F.softmax(log_coeffs_unnorm, dim=-1)
        coeffs = torch.clamp(coeffs, min=1e-9)
        comp_idx = torch.multinomial(coeffs, 1).squeeze(-1)
        batch_idx = torch.arange(mus.size(0), device=mus.device)
        mu_sel = mus[batch_idx, comp_idx, :]
        logvar_sel = log_vars[batch_idx, comp_idx, :]
        std_sel = torch.exp(0.5 * logvar_sel)
        eps = torch.randn_like(std_sel)
        params = mu_sel + eps * std_sel
        return params, coeffs

    def _gmm_nll(self, theta, log_coeffs_unnorm, mus, log_vars):
        """Negative log-likelihood (mean over batch) of theta under predicted mixture."""
        pi = F.softmax(log_coeffs_unnorm, dim=-1) + 1e-12  # [B,K]
        theta_e = theta.unsqueeze(1)                       # [B,1,D]
        var = torch.exp(log_vars)
        log_prob_dims = -0.5 * (((theta_e - mus).pow(2) / (var + 1e-12)) + log_vars + math.log(2 * math.pi))
        log_prob_comp = log_prob_dims.sum(dim=-1)          # [B, K]
        weighted = log_prob_comp + torch.log(pi)
        log_likelihood = torch.logsumexp(weighted, dim=-1)  # [B]
        return -log_likelihood.mean()

    # -------------------------
    # Regularization (sensible & simple)
    # -------------------------
    def _compute_regularization_terms(self, log_coeffs_unnorm, mus, log_vars, coeffs, sampled_params):
        device = mus.device
        D = mus.size(-1)
        K = mus.size(1)

        # boundary / l2 regularization
        if self.use_l2_instead_of_per_transform:
            # MSE to identity parameters (averaged)
            identity_batch = self.transformation_problem.get_identity_parameters(batch_size=sampled_params.shape[0])
            boundary_reg = F.mse_loss(sampled_params, identity_batch, reduction="mean")
        else:
            boundary_reg = self.transformation_problem.boundary_violation(sampled_params).mean()

        total_gmm_reg = torch.tensor(0.0, device=device, dtype=mus.dtype)

        # core MDN loss: self NLL (useful even when only classifying to keep mixture sensible)
        if "selfnll" in self.reg_options:
            total_gmm_reg = total_gmm_reg + self._gmm_nll(sampled_params, log_coeffs_unnorm, mus, log_vars)

        # optional KL to N(0,I) on component params
        if "kl" in self.reg_options:
            kl_comp = -0.5 * (1 + log_vars - mus.pow(2) - log_vars.exp()).sum(dim=2)  # [B,K]
            expected_kl = (coeffs * kl_comp).sum(dim=-1).mean()  # scalar
            total_gmm_reg = total_gmm_reg + expected_kl

        # normalize gmm regularizer by D*K to reduce sensitivity to dimensionality/components
        normalization = float(max(1.0, D * K))
        total_gmm_reg = total_gmm_reg / normalization

        return boundary_reg, total_gmm_reg

    # -------------------------
    # Loss helpers and forward
    # -------------------------
    def _compute_classification_loss(self, x_transformed, y):
        if self.conf_module is None:
            logits = self.main_model(x_transformed)
            ce_loss = F.cross_entropy(logits, y)
        else:
            conf, logits = self.conf_module(x_transformed, y=y)
            ce_loss = (-conf).mean(0)
        return ce_loss, logits

    def _compute_regression_loss(self, predicted_params, true_params):
        """
        Compute regression loss according to target_param_form.
        For 'matrix' and 'matrix_inverted' we compute MSE between transformation matrices.
        """
        if self.target_param_form == 'params':
            raise RuntimeError("Use direct_loss(x, true_params) to compute param-form regression NLL.")
        else:
            T_p = self.transformation_problem(predicted_params)  # predicted matrices
            if self.target_param_form == 'matrix':
                T_t = true_params.detach()
            else:  # matrix_inverted
                orig_dtype = true_params.dtype
                maybe = true_params
                if maybe.dtype in (torch.float16, torch.bfloat16):
                    maybe = maybe.float()
                T_t = torch.inverse(maybe).detach()
                if orig_dtype in (torch.float16, torch.bfloat16):
                    T_t = T_t.to(orig_dtype)
            return F.mse_loss(T_p, T_t)

    # --- Backward-compatible wrappers (classification/regression helpers) ---
    def classifiction_losses(self, x, y):
        log_coeffs_unnorm, mus, log_vars = self._get_gmm_params(x)
        sampled_params, coeffs = self._sample_from_gmm(log_coeffs_unnorm, mus, log_vars)
        x_t = self.transformation_problem.transform(x, sampled_params)
        ce_loss, logits = self._compute_classification_loss(x_t, y)
        boundary_reg, gmm_reg = self._compute_regularization_terms(log_coeffs_unnorm, mus, log_vars, coeffs, sampled_params)

        kl_term = self.kl_weight * gmm_reg if "kl" in self.reg_options else 0.0
        total_loss = ce_loss + self.l2_weight * boundary_reg + kl_term
        return total_loss, ce_loss, boundary_reg, gmm_reg, logits

    def direct_loss(self, x, true_params):
        log_coeffs_unnorm, mus, log_vars = self._get_gmm_params(x)
        sampled_params, coeffs = self._sample_from_gmm(log_coeffs_unnorm, mus, log_vars)

        if self.target_param_form == 'params':
            # Regression NLL (MDN-style)
            nll_loss = self._gmm_nll(true_params, log_coeffs_unnorm, mus, log_vars)
        else:
            # Matrix targets -> MSE between predicted params->matrices and true matrices
            nll_loss = self.regress_target(sampled_params, true_params, use_mse=True)

        boundary_reg, gmm_reg = self._compute_regularization_terms(
            log_coeffs_unnorm, mus, log_vars, coeffs, sampled_params
        )
        total_loss = nll_loss + self.l2_weight * boundary_reg + self.kl_weight * gmm_reg
        return total_loss, nll_loss, boundary_reg, gmm_reg


    # -------------------------
    # forward / deterministic transform
    # -------------------------
    def forward(self, x, return_stats: bool = False):
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)
        log_coeffs_unnorm, mus, log_vars = self._get_gmm_params(x)
        params, _ = self._sample_from_gmm(log_coeffs_unnorm, mus, log_vars)
        if self.clamp:
            params = self.transformation_problem.correct_param(params, reflect=False)
        x_t = self.transformation_problem.transform(x, params)
        logits = self.main_model(x_t)
        if return_stats:
            return logits, mus, log_vars, log_coeffs_unnorm, params
        return logits

    def transform_input(self, x, return_stats: bool = False):
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)
        log_coeffs_unnorm, mus, log_vars = self._get_gmm_params(x)
        coeffs = F.softmax(log_coeffs_unnorm, dim=-1)
        best_k = torch.argmax(coeffs, dim=-1)
        batch_idx = torch.arange(mus.size(0), device=mus.device)
        params_det = mus[batch_idx, best_k, :]
        if self.clamp:
            params_det = self.transformation_problem.correct_param(params_det, reflect=False)
        x_t = self.transformation_problem.transform(x, params_det)
        if return_stats:
            return x_t, mus, log_vars, log_coeffs_unnorm, params_det
        return x_t


    def regress_target(self, predicted_params, true_params, use_mse: bool = True):
        """
        Compute regression loss when targets are matrices (or when you want matrix MSE).
        - predicted_params: [B, D] (param-vector predicted/sampled)
        - true_params: either transformation matrices [B, M, M...] depending on transformation_problem,
                       or param-vector if you specifically want to compare params (only used for 'params' case).
        - use_mse: use MSE on transformation matrices (True) otherwise fallback to transformation_problem.distance.
        """
        # If target form is params, we shouldn't be here in normal flow — handled elsewhere.
        # Convert predicted params -> transformation matrix/tensor
        T_p = self.transformation_problem(predicted_params)  # expected shape [B, ... matrix shape ...]
        if self.target_param_form == 'matrix':
            # assume true_params are already matrices
            T_t = true_params.detach()
        else:  # 'matrix_inverted'
            orig_dtype = true_params.dtype
            maybe = true_params
            if maybe.dtype in (torch.float16, torch.bfloat16):
                maybe = maybe.float()
            T_t = torch.inverse(maybe).detach()
            if orig_dtype in (torch.float16, torch.bfloat16):
                T_t = T_t.to(orig_dtype)

        # Use MSE between matrices
        return F.mse_loss(T_p, T_t)


    # -------------------------
    # Lightning steps
    # -------------------------
    def training_step(self, batch, batch_idx):
        # support (x, y) or (x, true_params, y)
        if len(batch) == 3:
            x, true_params, y = batch
        elif len(batch) == 2:
            x, y = batch
            true_params = None
        else:
            raise ValueError("Training batch must contain either 2 or 3 elements (x, y, [true_params]).")

        # negative sampling hook
        with torch.no_grad():
            if self.negative_sampling_module is not None:
                zw = self.negative_sampling_module((x, y))
                if isinstance(zw[0], tuple):
                    x, y = zw[0]
                    true_params = zw[1]
                else:
                    x, y = zw
            if true_params is None and self.train_mode > 0.0:
                raise ValueError("true_params required for regression mode when train_mode > 0.0.")

        # pretransform if available
        if self.pretransform is not None:
            with torch.no_grad():
                x_proc = self.pretransform(x)
        else:
            x_proc = x

        log_coeffs_unnorm, mus, log_vars = self._get_gmm_params(x_proc)
        sampled_params, coeffs = self._sample_from_gmm(log_coeffs_unnorm, mus, log_vars)
        if self.clamp:
            sampled_params = self.transformation_problem.correct_param(sampled_params, reflect=False)

        boundary_reg, gmm_reg = self._compute_regularization_terms(log_coeffs_unnorm, mus, log_vars, coeffs, sampled_params)
        x_transformed = self.transformation_problem.transform(x_proc, sampled_params)

        cls_loss = torch.tensor(0.0, device=self.device)
        reg_loss = torch.tensor(0.0, device=self.device)

        # classification
        if self.train_mode < 1.0:
            if y is None:
                raise ValueError("y (labels) required for classification mode.")
            cls_loss, logits = self._compute_classification_loss(x_transformed, y)
            self.log("train_acc", (logits.argmax(dim=1) == y).float().mean(), on_step=False, on_epoch=True)
        else:
            logits = None

        # regression (mixture NLL)
        if self.train_mode > 0.0:
            if true_params is None:
                raise ValueError("true_params required for regression mode when train_mode > 0.0.")

            if self.target_param_form == 'params':
                # true_params are expected to be param-vectors [B, D]
                reg_loss = self._gmm_nll(true_params, log_coeffs_unnorm, mus, log_vars)
            else:
                # true_params are matrices -> compute MSE between predicted sampled params -> matrices and true matrices
                # Use sampled_params (stochastic) or consider using component mean (mus) if you prefer deterministic regression
                reg_loss = self.regress_target(sampled_params, true_params, use_mse=True)

        main_loss = (1.0 - self.train_mode) * cls_loss + self.train_mode * reg_loss

        # apply KL only if requested in reg_options
        kl_term = self.kl_weight * gmm_reg if "kl" in self.reg_options else 0.0
        total_loss = main_loss + self.l2_weight * boundary_reg + kl_term

        # logging
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_main_loss", main_loss, on_step=False, on_epoch=True)
        self.log("train_boundary_reg", boundary_reg, on_step=False, on_epoch=True)
        self.log("train_gmm_reg", gmm_reg, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, _, y = batch
        elif len(batch) == 2:
            x, y = batch
        else:
            raise ValueError("Validation batch must contain either 2 or 3 elements (x, y, [true_params]).")

        if self.pretransform is not None:
            with torch.no_grad():
                x_proc = self.pretransform(x)
        else:
            x_proc = x

        log_coeffs_unnorm, mus, log_vars = self._get_gmm_params(x_proc)
        sampled_params, coeffs = self._sample_from_gmm(log_coeffs_unnorm, mus, log_vars)
        if self.clamp:
            sampled_params = self.transformation_problem.correct_param(sampled_params, reflect=False)
        x_transformed_sampled = self.transformation_problem.transform(x_proc, sampled_params)
        ce_loss, logits_sampled = self._compute_classification_loss(x_transformed_sampled, y)
        boundary_reg, gmm_reg = self._compute_regularization_terms(log_coeffs_unnorm, mus, log_vars, coeffs, sampled_params)
        kl_term = self.kl_weight * gmm_reg if "kl" in self.reg_options else 0.0
        total_loss_sampled = ce_loss + self.l2_weight * boundary_reg + kl_term
        acc_sampled = (logits_sampled.argmax(dim=1) == y).float().mean()

        x_transformed_det = self.transform_input(x)
        _, logits_det = self._compute_classification_loss(x_transformed_det, y)
        acc_max_prop = (logits_det.argmax(dim=1) == y).float().mean()

        self.log("val_loss", total_loss_sampled, prog_bar=True)
        self.log("val_acc_sampled", acc_sampled, prog_bar=True)
        self.log("val_acc_max_prop", acc_max_prop, prog_bar=True)
        return total_loss_sampled

    def configure_optimizers(self):
        # mirror your SpatialTransformerClassifier grouping
        if self.train_main:
            opt = self.optimizer_class(self.localization_net.parameters(), **self.optimizer_params)
            opt.add_param_group({"params": self.gmm_params_head.parameters()})
            opt.add_param_group({"params": self.main_model.parameters()})
        else:
            opt = self.optimizer_class(self.localization_net.parameters(), **self.optimizer_params)
            opt.add_param_group({"params": self.gmm_params_head.parameters()})

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(opt, **(self.lr_scheduler_params or {}))
            sched_conf = self.lr_config or {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}
            sched_conf["scheduler"] = scheduler
            return {"optimizer": opt, "lr_scheduler": sched_conf}
        return opt




import pytorch_lightning as pl


class TrainModeSchedulerCallback(pl.Callback):
    """
    A PyTorch Lightning callback to schedule the train_mode of a module.

    This callback cycles through a schedule:
    1.  For `regression_epochs`, it sets `train_mode` to 1.0 (pure regression).
        It also sets the module's `l2_weight` to 0.0 for this phase.
    2.  For the next `transition_epochs`, it linearly interpolates `train_mode`
        from 1.0 down to 0.0, restoring the original `l2_weight`.
    This cycle repeats throughout training.
    """

    def __init__(self, regression_epochs: int, transition_epochs: int):
        super().__init__()
        if regression_epochs < 0:
            raise ValueError("regression_epochs must be non-negative.")
        if transition_epochs <= 0:
            raise ValueError("transition_epochs must be positive.")
        self.regression_epochs = regression_epochs
        self.transition_epochs = transition_epochs
        self.cycle_length = self.regression_epochs + self.transition_epochs
        self.original_l2_weight = None  # To store the initial l2_weight

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Called at the beginning of each training epoch to update train_mode and l2_weight."""
        if not hasattr(pl_module, 'train_mode'):
            return

        # Store the original l2_weight on the first run
        if self.original_l2_weight is None and hasattr(pl_module, 'l2_weight'):
            self.original_l2_weight = pl_module.l2_weight

        if self.cycle_length == 0:
            return

        epoch_in_cycle = trainer.current_epoch % self.cycle_length
        new_train_mode = 1.0

        if epoch_in_cycle < self.regression_epochs:
            # Case 1: Pure regression phase
            new_train_mode = 1.0
            if self.original_l2_weight is not None:
                pl_module.l2_weight = 0.0
        else:
            # Case 2: Transition phase
            t = epoch_in_cycle - self.regression_epochs
            new_train_mode = 1.0 - (t + 1.0) / self.transition_epochs
            new_train_mode = max(0.0, new_train_mode)
            # Restore original l2_weight during transition and classification phases
            if self.original_l2_weight is not None:
                pl_module.l2_weight = self.original_l2_weight

        pl_module.train_mode = new_train_mode
        pl_module.log("train_mode_scheduled", new_train_mode, on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log("l2_weight_scheduled", pl_module.l2_weight, on_step=False, on_epoch=True)
