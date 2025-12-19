
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Literal
from confidence.unsupervised.unsupervised_base import MLConfidenceBase

class EquiCanonicalizationConfidence(MLConfidenceBase):
    def __init__(
        self,
        canonicalization_network: nn.Module,
        main_model: nn.Module,
        loss_main_model: nn.Module = torch.nn.CrossEntropyLoss(),
        vector_dim: Optional[int] = None,
        beta: float = 1.0,
        task_loss_multiplier: float = 1.0,
        prior_loss_multiplier: float = 1.0,
        diversity_loss_multiplier: float = 1.0,
        reg_loss_multiplier: float = 0.0,
        similarity_metric: Literal["cosine", "dot", "mse", "logit"] = "cosine",
        learn_main_model: bool = False,
        learn_ref_vec: bool = False,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        negative_sampling_module: Optional[Any] = None,
        input_transform: Optional[Any] = None,
        gradient_passthrough: Literal["gumbel", "ste"] = "gumbel",
        diversity_loss_type: Literal["gram_abs", "variance", "uniformity"] = "variance",
        use_parameterized_transforms: bool = True,  # NEW: selection method
        gumbel_temperature: float = 1.0,  # Temperature for Gumbel-Softmax
    ):
        super().__init__(
            input_transform=input_transform,
            trainer_kwargs=trainer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            negative_sampling_module=negative_sampling_module,
        )

        self.canon_net = canonicalization_network
        self.main_model = main_model
        self.beta = float(beta)
        self.loss_main_model = loss_main_model
        self.task_loss_multiplier = float(task_loss_multiplier)
        self.prior_loss_multiplier = float(prior_loss_multiplier)
        self.diversity_loss_multiplier = float(diversity_loss_multiplier)
        self.reg_loss_multiplier = float(reg_loss_multiplier)
        self.similarity_metric = similarity_metric
        self.out_vector_size = vector_dim
        self.use_parameterized_transforms = use_parameterized_transforms
        self.gumbel_temperature = gumbel_temperature

        if self.similarity_metric in ("cosine", "dot", "mse"):
            if self.out_vector_size is None:
                raise ValueError("vector_dim must be provided for similarity_metric != 'logit'.")
            self.reference_vector = nn.Parameter(
                torch.randn(1, self.out_vector_size),
                requires_grad=learn_ref_vec,
            )
        else:
            self.reference_vector = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

        self.gradient_passthrough = gradient_passthrough
        self.learn_main_model = learn_main_model
        self.diversity_loss_type = diversity_loss_type

        # Check if negative sampling module returns params
        if negative_sampling_module is not None:
            if not hasattr(negative_sampling_module, 'return_params'):
                raise ValueError("negative_sampling_module must have 'return_params' attribute")
            if use_parameterized_transforms and not negative_sampling_module.return_params:
                raise ValueError("use_parameterized_transforms=True requires negative_sampling_module.return_params=True")

            # Store transform_sequence for applying transforms
            if hasattr(negative_sampling_module.strategy, 'transform_sequence'):
                self.transform_sequence = negative_sampling_module.strategy.transform_sequence
            else:
                raise ValueError("negative_sampling_module.strategy must have 'transform_sequence' attribute")

    def _split_orbit(self, x: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, int]:
        B_total = x.shape[0]
        assert B_total % batch_size == 0
        K = B_total // batch_size
        new_shape = (K, batch_size) + tuple(x.shape[1:])
        x_reshaped = x.view(*new_shape)
        return x_reshaped.permute(1, 0, *range(2, x_reshaped.dim())), K

    def _compute_scores(self, z: torch.Tensor) -> torch.Tensor:
        if self.similarity_metric == "cosine":
            return F.cosine_similarity(z, self.reference_vector, dim=-1, eps=1e-8)
        elif self.similarity_metric == "dot":
            return (z * self.reference_vector).sum(dim=-1)
        elif self.similarity_metric == "mse":
            return -((z - self.reference_vector) ** 2).mean(dim=-1)
        elif self.similarity_metric == "logit":
            return z if z.dim() == 1 else z[..., 0]
        raise ValueError(f"Unknown similarity_metric {self.similarity_metric}")

    def _canonicalizer_pass(self, x: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.canon_net(x)
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        scores = self._compute_scores(z)
        scores_reshaped, _ = self._split_orbit(scores, batch_size)
        return z, scores_reshaped

    def _diversity_loss(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B, K, D] already grouped
        B, K, D = Z.shape
        if K < 2 or self.diversity_loss_multiplier == 0.0:
            return Z.new_zeros(())

        if self.diversity_loss_type == "gram_abs":
            G = Z @ Z.transpose(-1, -2)  # B,K,K
            mask = 1.0 - torch.eye(K, device=Z.device, dtype=Z.dtype).unsqueeze(0)
            return (G.abs() * mask).mean()

        Zc = Z - Z.mean(dim=1, keepdim=True)

        if self.diversity_loss_type == "variance":
            var = Zc.var(dim=1, unbiased=False)  # B,D
            std = torch.sqrt(var + 1e-6)
            return F.relu(1.0 - std).mean()

        if self.diversity_loss_type == "uniformity":
            Zn = F.normalize(Zc, dim=-1)
            alpha = 2.0
            d2 = torch.cdist(Zn, Zn, p=2).pow(2)  # B,K,K
            mask = ~torch.eye(K, dtype=torch.bool, device=Z.device)
            exp_term = torch.exp(-alpha * d2[:, mask].view(B, -1))
            return torch.log(exp_term.mean(dim=1) + 1e-6).mean()

        raise ValueError(f"Unknown diversity_loss_type {self.diversity_loss_type}")

    def select_group_act(self, Q: torch.Tensor) -> torch.Tensor:
        """Discrete selection for image-based approach (not used in parameterized)."""
        if self.gradient_passthrough == "gumbel":
            return F.gumbel_softmax(Q, tau=self.gumbel_temperature, hard=True, dim=-1)
        # STE
        one_hot = torch.nn.functional.one_hot(Q.argmax(dim=-1), num_classes=Q.shape[1]).to(Q.dtype)
        soft = F.softmax(self.beta * Q, dim=-1)
        return one_hot - soft.detach() + soft

    def _select_transform_params(
        self,
        S: torch.Tensor,
        T_shaped: torch.Tensor
    ) -> torch.Tensor:
        """
        Select transform parameters using gradient estimators.

        Args:
            S: Scores [batch_size, K]
            T_shaped: Transform parameters [batch_size, K, ...] (can be multi-dimensional)

        Returns:
            Selected transform parameters [batch_size, ...]
        """
        batch_size, K = S.shape
        # T_shaped can have shape [batch_size, K, transform_dim] or [batch_size, K, ...]
        # We need to handle arbitrary trailing dimensions

        if self.gradient_passthrough == "gumbel":
            # Gumbel-Softmax: samples from categorical distribution with reparameterization
            # Returns soft weights that are differentiable
            selection_weights = F.gumbel_softmax(
                self.beta * S,
                tau=self.gumbel_temperature,
                hard=False,  # Keep soft for continuous gradients
                dim=-1
            )  # [batch_size, K]

            # Weighted sum of transform parameters
            # Need to broadcast selection_weights to match T_shaped dimensions
            # T_shaped: [batch_size, K, ...], selection_weights: [batch_size, K]
            # Reshape to [batch_size, K, 1, 1, ...] to broadcast correctly
            num_extra_dims = len(T_shaped.shape) - 2
            for _ in range(num_extra_dims):
                selection_weights = selection_weights.unsqueeze(-1)

            T_selected = (selection_weights * T_shaped).sum(dim=1)  # [batch_size, ...]

        elif self.gradient_passthrough == "ste":
            # Straight-Through Estimator:
            # Forward pass uses hard selection, backward pass uses soft gradients
            soft_weights = F.softmax(self.beta * S, dim=-1)  # [batch_size, K]

            # Hard selection (one-hot)
            hard_selection = torch.zeros_like(soft_weights)
            hard_indices = S.argmax(dim=-1)
            hard_selection.scatter_(1, hard_indices.unsqueeze(1), 1.0)

            # STE: forward uses hard, backward uses soft
            selection_weights = hard_selection - soft_weights.detach() + soft_weights

            # Select transform parameters
            # Broadcast to match T_shaped dimensions
            num_extra_dims = len(T_shaped.shape) - 2
            for _ in range(num_extra_dims):
                selection_weights = selection_weights.unsqueeze(-1)

            T_selected = (selection_weights * T_shaped).sum(dim=1)  # [batch_size, ...]

        else:
            raise ValueError(f"Unknown gradient_passthrough: {self.gradient_passthrough}")

        return T_selected

    def _compute_losses_discrete(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Original discrete selection method."""
        in_dist = y >= 0.0
        batch_size = in_dist.sum().item()
        assert batch_size > 0 and x.shape[0] % batch_size == 0

        z, S = self._canonicalizer_pass(x, batch_size)
        one_hot_selected = self.select_group_act(S)

        x_shaped = self._split_orbit(x, batch_size)[0]
        x_selected = (one_hot_selected.reshape(batch_size, -1, *([1] * (x.dim() - 1))) * x_shaped).sum(dim=1)

        main_res = self.main_model(x_selected)
        y_in = y[in_dist]
        main_loss = self.loss_main_model(main_res, y_in)

        dataset_prior = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        prior_loss = F.cross_entropy(S, dataset_prior)

        Z = self._split_orbit(z, batch_size)[0]
        diversity_loss = self._diversity_loss(Z)

        reg_loss = (z ** 2).mean() if self.reg_loss_multiplier > 0.0 else z.new_zeros(())

        return {
            "task_loss": main_loss,
            "prior_loss": prior_loss,
            "diversity_loss": diversity_loss,
            "reg_loss": reg_loss,
            "S_mean": S.mean(),
        }

    def _compute_losses_parameterized(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        T_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """New parameterized transform method using transform_sequence with gradient estimators."""
        in_dist = y >= 0.0
        batch_size = in_dist.sum().item()
        assert batch_size > 0 and x.shape[0] % batch_size == 0

        # Get positive samples
        x_pos = x[in_dist]
        y_pos = y[in_dist]

        # Get all orbit samples for computing scores
        z, S = self._canonicalizer_pass(x, batch_size)

        # Split transform params into orbits [batch_size, K, transform_dim]
        T_shaped, K_orbits = self._split_orbit(T_params, batch_size)

        # Select transform parameters using Gumbel-Softmax or STE
        # This is the key difference: we select PARAMETERS, not images
        T_selected = self._select_transform_params(S, T_shaped)  # [batch_size, transform_dim]

        # Apply the selected transform to positive images
        # Gradients flow through T_selected back to the canonicalization network
        x_canonicalized = self.transform_sequence.application_method(x_pos, T_selected)

        # Task loss
        main_res = self.main_model(x_canonicalized)
        main_loss = self.loss_main_model(main_res, y_pos)

        # Prior loss: encourage selecting the identity (first element)
        dataset_prior = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        prior_loss = F.cross_entropy(S, dataset_prior)

        # Diversity loss
        Z = self._split_orbit(z, batch_size)[0]
        diversity_loss = self._diversity_loss(Z)

        # Regularization
        reg_loss = (z ** 2).mean() if self.reg_loss_multiplier > 0.0 else z.new_zeros(())

        return {
            "task_loss": main_loss,
            "prior_loss": prior_loss,
            "diversity_loss": diversity_loss,
            "reg_loss": reg_loss,
            "S_mean": S.mean(),
        }

    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.use_parameterized_transforms:
            # Expect batch to be tuple (modified_batch, T_params) from BatchNegativeSampler
            if isinstance(batch, tuple) and len(batch) == 2:
                actual_batch, T_params = batch
                x, y = actual_batch
            else:
                raise ValueError(
                    "use_parameterized_transforms=True requires BatchNegativeSampler "
                    "with return_params=True, which should return (batch, T_params)"
                )
            losses = self._compute_losses_parameterized(x, y, T_params)
        else:
            x, y = batch
            losses = self._compute_losses_discrete(x, y)

        loss = (
            self.task_loss_multiplier * losses["task_loss"]
            + self.prior_loss_multiplier * losses["prior_loss"]
            + self.diversity_loss_multiplier * losses["diversity_loss"]
            + self.reg_loss_multiplier * losses["reg_loss"]
        )

        self.log_dict(
            {
                "train_task": losses["task_loss"],
                "train_prior": losses["prior_loss"],
                "train_diversity": losses["diversity_loss"],
                "train_reg": losses["reg_loss"],
                "train_S_mean": losses["S_mean"],
            },
            on_step=True,
            on_epoch=True,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if self.use_parameterized_transforms:
            if isinstance(batch, tuple) and len(batch) == 2:
                actual_batch, T_params = batch
                x, y = actual_batch
            else:
                raise ValueError(
                    "use_parameterized_transforms=True requires BatchNegativeSampler "
                    "with return_params=True"
                )
            with torch.no_grad():
                losses = self._compute_losses_parameterized(x, y, T_params)
        else:
            x, y = batch
            with torch.no_grad():
                losses = self._compute_losses_discrete(x, y)

        loss = (
            self.task_loss_multiplier * losses["task_loss"]
            + self.prior_loss_multiplier * losses["prior_loss"]
            + self.diversity_loss_multiplier * losses["diversity_loss"]
            + self.reg_loss_multiplier * losses["reg_loss"]
        )

        self.log_dict(
            {
                "val_task": losses["task_loss"],
                "val_prior": losses["prior_loss"],
                "val_diversity": losses["diversity_loss"],
                "val_reg": losses["reg_loss"],
                "val_S_mean": losses["S_mean"],
            },
            on_step=False,
            on_epoch=True,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        in_dist = y >= 0.0 if y is not None else torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        batch_size = in_dist.sum().item() if in_dist.any() else x.shape[0]
        _, sim = self._canonicalizer_pass(x, batch_size=batch_size)
        return sim.squeeze(-1)

    def configure_optimizers(self):
        if self.learn_main_model:
            return self.optimizer_type(self.parameters(), **self.optimizer_kwargs)
        params = list(self.canon_net.parameters())
        if self.reference_vector is not None:
            params.append(self.reference_vector)
        return self.optimizer_type(params, **self.optimizer_kwargs)