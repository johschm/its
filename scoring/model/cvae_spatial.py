import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder for transformation parameters θ given input features x.

    Allows either a learned prior p(z|x) (a small MLP) or a fixed standard normal prior.
    Encodes (θ, x) → q(z | θ, x), and decodes z, x → p(θ | z, x) as a diagonal Gaussian.
    """

    def __init__(
        self,
        param_dim: int,
        context_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        use_learned_prior: bool = True,
    ):
        """
        Arguments:
          • param_dim        : dimensionality D of θ
          • context_dim      : dimensionality of input features x
          • latent_dim       : dimensionality of latent z
          • hidden_dim       : hidden layer size in encoder/decoder/prior nets
          • use_learned_prior: if True, learn p(z|x) via self.prior_net;
                               if False, use standard normal prior N(0,I)
        """
        super().__init__()
        self.param_dim = param_dim
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.use_learned_prior = use_learned_prior

        # ---------------- Encoder: q(z | θ, x) ----------------
        # Input: [θ, features] → MLP → output 2*latent_dim (μ_enc, logσ²_enc)
        self.encoder = nn.Sequential(
            nn.Linear(param_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # μ_enc, logσ²_enc
        )

        # ---------------- Decoder: p(θ | z, x) ----------------
        # Input: [z, features] → MLP → output 2*param_dim (μ_dec, logσ²_dec)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim * 2),  # μ_dec, logσ²_dec
        )

        # ---------------- Learned Prior: p(z | x) ----------------
        # Input: features → MLP → output 2*latent_dim (μ_prior, logσ²_prior)
        if use_learned_prior:
            self.prior_net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, latent_dim * 2),  # μ_prior, logσ²_prior
            )
        else:
            self.prior_net = None  # will use standard normal prior

    def encode(self, params: torch.Tensor, context: torch.Tensor):
        """
        Encode (θ, x) → μ_enc, logσ²_enc
        params: [B, param_dim], context: [B, context_dim]
        Returns: (μ_enc [B, latent_dim], logσ²_enc [B, latent_dim])
        """
        h = torch.cat([params, context], dim=-1)  # [B, param_dim + context_dim]
        h = self.encoder(h)                       # [B, latent_dim * 2]
        μ_enc, logσ2_enc = h.chunk(2, dim=-1)      # each [B, latent_dim]
        return μ_enc, logσ2_enc

    def prior(self, context: torch.Tensor):
        """
        Compute prior p(z | x) → μ_prior, logσ²_prior.
        If use_learned_prior=False, returns (0, 0) meaning standard normal.
        context: [B, context_dim]
        Returns: (μ_prior [B, latent_dim], logσ²_prior [B, latent_dim])
        """
        B = context.shape[0]
        if self.use_learned_prior:
            h = self.prior_net(context)               # [B, latent_dim * 2]
            μ_prior, logσ2_prior = h.chunk(2, dim=-1)  # each [B, latent_dim]
        else:
            μ_prior = torch.zeros(B, self.latent_dim, device=context.device)
            logσ2_prior = torch.zeros(B, self.latent_dim, device=context.device)
        return μ_prior, logσ2_prior

    def reparameterize(self, μ: torch.Tensor, logσ2: torch.Tensor):
        """
        Reparameterization: z = μ + σ * ε, where ε ~ N(0, I).
        μ: [B, D], logσ2: [B, D]
        Returns: z [B, D]
        """
        σ = torch.exp(0.5 * logσ2)
        ε = torch.randn_like(σ)
        return μ + σ * ε

    def decode(self, z: torch.Tensor, context: torch.Tensor):
        """
        Decode z, x → μ_dec, logσ²_dec for p(θ | z, x).
        z: [B, latent_dim], context: [B, context_dim]
        Returns: (μ_dec [B, param_dim], logσ²_dec [B, param_dim])
        """
        h = torch.cat([z, context], dim=-1)  # [B, latent_dim + context_dim]
        h = self.decoder(h)                  # [B, param_dim * 2]
        μ_dec, logσ2_dec = h.chunk(2, dim=-1) # each [B, param_dim]
        return μ_dec, logσ2_dec

    def sample_prior(self, context: torch.Tensor):
        """
        Sample θ ~ p(θ | x):
          1) (μ_prior, logσ²_prior) = p(z|x)  (learned or zeros)
          2) z ~ N(μ_prior, σ²_prior)
          3) (μ_dec, logσ²_dec) = decode(z, x)
          4) θ ~ N(μ_dec, σ²_dec)
        Returns: θ [B, param_dim],
                 μ_dec [B, param_dim],
                 logσ²_dec [B, param_dim],
                 μ_prior [B, latent_dim],
                 logσ²_prior [B, latent_dim]
        """
        μ_prior, logσ2_prior = self.prior(context)            # [B, latent_dim] each
        z = self.reparameterize(μ_prior, logσ2_prior)         # [B, latent_dim]
        μ_dec, logσ2_dec = self.decode(z, context)             # [B, param_dim] each
        θ = self.reparameterize(μ_dec, logσ2_dec)              # [B, param_dim]
        return θ, μ_dec, logσ2_dec, μ_prior, logσ2_prior

    def log_prob(self, params: torch.Tensor, context: torch.Tensor, beta: float = 1.0):
        """
        Approximate log p(θ | x) via evidence lower bound (ELBO):
          ELBO = E_{q(z|θ,x)}[log p(θ|z,x)] – β * KL(q(z|θ,x) || p(z|x)).
        We return per-sample ELBO [B].

        params: [B, param_dim], context: [B, context_dim]
        beta: weight on the KL term inside ELBO.
        Returns: elbo [B]
        """
        # 1) q(z | θ, x)
        μ_enc, logσ2_enc = self.encode(params, context)         # [B, latent_dim] each
        z = self.reparameterize(μ_enc, logσ2_enc)               # [B, latent_dim]

        # 2) p(θ | z, x)
        μ_dec, logσ2_dec = self.decode(z, context)               # [B, param_dim] each

        # 3) Reconstruction term:
        #    rec_term = E_q [ log p(θ | z, x) ]
        #    = -0.5 * [ Σ_d ( (θ_d - μ_dec_d)^2 / σ2_dec_d  + log σ2_dec_d + log(2π) ) ]
        rec_term = -0.5 * (
            torch.sum((params - μ_dec).pow(2) / logσ2_dec.exp(), dim=-1)    # Σ ( (θ-μ)^2 / σ2 )
            + torch.sum(logσ2_dec, dim=-1)                                  # Σ log σ2
            + params.shape[-1] * math.log(2 * math.pi)
        )  # [B]

        # 4) Prior p(z | x)
        μ_prior, logσ2_prior = self.prior(context)                   # [B, latent_dim] each

        # 5) KL[q || p]:
        #    0.5 * Σ_d [ logσ2_prior_d - logσ2_enc_d
        #                + (σ2_enc_d + (μ_enc_d - μ_prior_d)^2) / σ2_prior_d  - 1 ]
        kl_term = 0.5 * torch.sum(
            logσ2_prior - logσ2_enc
            + (logσ2_enc.exp() + (μ_enc - μ_prior).pow(2)) / logσ2_prior.exp()
            - 1,
            dim=-1
        )  # [B]

        # 6) ELBO:
        elbo = rec_term - beta * kl_term  # [B]
        return elbo  # [B]


def _compute_cvae_reg_on_decoder(
    mu_dec: torch.Tensor,
    log_sigma2_dec: torch.Tensor,
    transform_problem,
    reg_options: set,
    beta_uniform: float = 1.0,
    gamma_normal: float = 1.0
) -> torch.Tensor:
    """
    Compute CVAE decoder‐based regularizers *only on* p(θ|z,x)=N(mu_dec, σ2_dec), matching a flow‐based philosophy.

    reg_options may include:
      • "kl_uniform" : KL[ N(mu_dec, σ²) || Uniform(bounds) ]  (weighted by beta_uniform)
      • "entropy"    : -H( N(mu_dec, σ²) )  (encourage high entropy)
      • "sharpness"  : Σ_d log(σ^2_dec[d]) averaged  (encourage low variance)
      • "kl_normal"  : KL[ N(mu_dec, σ²) || N(0, I) ]  (weighted by gamma_normal)

    Arguments:
      – mu_dec         : [B, D] = μ_dec(z, x)
      – log_sigma2_dec : [B, D] = log σ²_dec(z, x)
      – transform_problem : has transform_sequence.lower_bounds, .upper_bounds each length D
      – reg_options    : set of strings from {"kl_uniform","entropy","sharpness","kl_normal"}
      – beta_uniform   : weight on the "kl_uniform" term
      – gamma_normal   : weight on the "kl_normal" term

    Returns:
      – total_cvae_reg : scalar Tensor (averaged over batch)
    """
    B, D = mu_dec.shape
    device = mu_dec.device

    # 1) Clamp log_sigma2_dec for numerical stability
    log_sigma2_dec = torch.clamp(log_sigma2_dec, min=-10.0, max=10.0)

    # ========== Compute entropy H[N] for each sample ==========
    # H = 0.5 * Σ_d [ log(2πe σ²_d) ] = 0.5 * Σ_d [ logσ²_d + log(2πe) ]
    H_dec = 0.5 * torch.sum(
        log_sigma2_dec + math.log(2 * math.pi * math.e),
        dim=1
    )  # [B]

    total = torch.zeros((), device=device)

    # ========== 2) KL[ N(μ_dec, σ²) || Uniform(bounds) ] ==========
    if "kl_uniform" in reg_options:
        lower = torch.tensor(transform_problem.transform_sequence.lower_bounds,
                             device=device).view(1, D)  # [1, D]
        upper = torch.tensor(transform_problem.transform_sequence.upper_bounds,
                             device=device).view(1, D)  # [1, D]
        ranges = upper - lower                         # [1, D]
        log_vol = torch.sum(torch.log(ranges), dim=1)  # [1]

        # KL_uniform_i = (-H_dec[i] + log_vol)
        KL_uniform = (-H_dec + log_vol).mean()  # scalar
        total = total + beta_uniform * KL_uniform

    # ========== 3) Entropy Bonus (encourage high entropy) ==========
    if "entropy" in reg_options:
        # We want to maximize H_dec, so add -H_dec to the loss
        L_entropy = (-H_dec).mean()  # scalar
        total = total + L_entropy

    # ========== 4) Sharpness Penalty (encourage low variance) ==========
    if "sharpness" in reg_options:
        # L_sharp = Σ_d log σ²_dec[d], averaged
        L_sharp = torch.sum(log_sigma2_dec, dim=1).mean()  # scalar
        total = total + L_sharp

    # ========== 5) KL[ N(μ_dec, σ²) || N(0, I) ] ==========
    if "kl_normal" in reg_options:
        sigma2_dec = torch.exp(log_sigma2_dec)                  # [B, D]
        kl_per_dim = mu_dec.pow(2) + sigma2_dec - log_sigma2_dec - 1  # [B, D]
        KL_normal = 0.5 * torch.sum(kl_per_dim, dim=1).mean()        # scalar
        total = total + gamma_normal * KL_normal

    return total  # scalar


class CVAESpatialTransformerClassifier(pl.LightningModule):
    """
    CVAE‐based Spatial Transformer Classifier.

    - Models p(θ | x) with a CVAE:
        • q(z|θ,x), p(z|x) (learned or fixed N(0,I)), p(θ|z,x)=N(μ_dec,σ²_dec).
    - Classification: sample θ ~ p(θ|x), warp x, compute CE loss.
    - Regularizes the decoder‐Gaussian via:
        - "kl_uniform", "entropy", "sharpness", "kl_normal" (all on decoder).
        - Boundary (or L₂) penalty on θ.
        - Optionally, negative ELBO on sampled θ if include_elbo_in_classification=True.
    - Regression (direct) mode: supervised negative ELBO on true θ + boundary on true θ,
      plus optionally boundary/decoder regs on sampled θ only if extra_samples > 1.
    """

    def __init__(
        self,
        main_model: nn.Module,
        localization_net: nn.Module,
        transformation_problem,
        localization_dim: int,
        latent_dim: int = 64,
        cvae_hidden_dim: int = 128,
        use_learned_prior: bool = True,
        freeze_main: bool = True,
        optimizer_class=torch.optim.Adam,
        optimizer_params: dict = {"lr": 1e-3},
        lr_scheduler=None,
        lr_scheduler_params=None,
        lr_config=None,
        pretransform=None,
        l2_weight: float = 0.0,
        use_l2_instead_of_per_transform: bool = False,
        conf_module=None,
        kl_weight: float = 1e-4,           # global weight for CVAE‐based reg terms
        beta_cvae: float = 1.0,            # inside ELBO: weight on KL (q‖p)
        beta_uniform: float = 1.0,         # weight on kl_uniform
        gamma_normal: float = 1.0,         # weight on kl_normal
        reg_options: str = "kl_uniform,entropy,sharpness,kl_normal",
        include_elbo_in_classification: bool = False,
        extra_samples: int = 1,            # number of samples for classification/regression
    ):
        """
        Arguments:
          • main_model       : classifier that takes warped x
          • localization_net : network producing features from x
          • transformation_problem: has:
                - calc_complete_size() → D
                - transform(x, θ) → warped x
                - transform_sequence.lower_bounds, upper_bounds each length D
                - boundary_violation(θ) → penalty per sample
          • localization_dim : dimension of features from localization_net
          • latent_dim       : size of CVAE latent z
          • cvae_hidden_dim  : hidden size in encoder/decoder/prior
          • use_learned_prior: if False, use standard Normal prior for z
          • freeze_main      : if True, do not train main_model
          • l2_weight        : weight on L₂/boundary penalty on θ
          • use_l2_instead_of_per_transform: if True, use ‖θ‖₂ instead of boundary_violation(θ)
          • conf_module      : optional confidence model wrapper
          • kl_weight        : global multiplier for CVAE‐based reg terms
          • beta_cvae        : inside ELBO: weight on posterior KL
          • beta_uniform     : weight on "kl_uniform" inside decoder regs
          • gamma_normal     : weight on "kl_normal" inside decoder regs
          • reg_options      : comma‐sep string from {"kl_uniform","entropy","sharpness","kl_normal"}
          • include_elbo_in_classification: if True, in classification mode compute negative ELBO
                                             on sampled θ (only if extra_samples > 1).
          • extra_samples    : number of MC samples to draw for classification and regression.
                                If ≤ 1, skip sampling‐based ELBO and decoding regs.
        """
        super().__init__()
        self.main_model = main_model
        self.localization_net = localization_net
        self.transformation_problem = transformation_problem
        self.pretransform = pretransform
        self.train_main = not freeze_main

        # Dimensions
        self.param_dim = transformation_problem.calc_complete_size()  # D
        self.localization_dim = localization_dim
        self.l2_weight = l2_weight
        self.use_l2_instead_of_per_transform = use_l2_instead_of_per_transform
        self.conf_module = conf_module

        # CVAE hyperparameters
        self.kl_weight = kl_weight              # scales all CVAE‐based reg terms
        self.beta_cvae = beta_cvae              # inside ELBO: weight on KL (q‖p)
        self.beta_uniform = beta_uniform        # inside decoder: weight on "kl_uniform"
        self.gamma_normal = gamma_normal        # inside decoder: weight on "kl_normal"
        self.include_elbo_in_classification = include_elbo_in_classification

        # How many MC samples to draw for classification/regression
        self.extra_samples = extra_samples

        # Parse reg_options string
        self.reg_options = set(r.strip().lower() for r in reg_options.split(",") if r.strip())
        valid_opts = {"kl_uniform", "entropy", "sharpness", "kl_normal"}
        unknown = self.reg_options - valid_opts
        if unknown:
            raise ValueError(f"Unknown reg_options: {unknown}. Valid: {valid_opts}.")

        # Instantiate CVAE
        self.cvae = CVAE(
            param_dim=self.param_dim,
            context_dim=self.localization_dim,
            latent_dim=latent_dim,
            hidden_dim=cvae_hidden_dim,
            use_learned_prior=use_learned_prior,
        )

        # Optimizer & scheduler
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_config = lr_config

    def _get_cvae_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute features for CVAE conditioning:
        Optionally apply pretransform, then pass x through localization_net and flatten.
        Returns: [B, localization_dim]
        """
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)
        feat = self.localization_net(x).flatten(1)
        return feat

    def _compute_classification_step(self, x: torch.Tensor, y: torch.Tensor):
        """
        Core of classification: sample θ ~ p(θ|x), warp, compute CE, boundary, decoder regs,
        and optionally negative ELBO on sampled θ—but only if extra_samples > 1.

        Returns:
          total_loss, ce_loss, boundary_reg, decoder_reg, elbo_reg, logits

        Where:
          – boundary_reg = boundary penalty on sampled θ (single sample if extra_samples ≤ 1)
          – decoder_reg  = sum of decoder‐based penalties ("kl_uniform","entropy","sharpness","kl_normal")
                           averaged over extra_samples samples if > 1, else from one sample
          – elbo_reg     = negative ELBO on sampled θ, only if include_elbo_in_classification and extra_samples > 1
        """
        B = x.shape[0]
        features = self._get_cvae_features(x)                           # [B, localization_dim]

        # ---- If extra_samples ≤ 1, do exactly one sample; else, average over extra_samples ----
        num_samples = max(1, self.extra_samples)

        total_ce = 0.0
        total_boundary = 0.0
        total_decoder_reg = 0.0
        total_elbo_reg = 0.0

        for _ in range(num_samples):
            # 1) Sample from prior: (μ_prior, logσ²_prior) → z → (μ_dec, logσ²_dec) → θ
            μ_prior, logσ2_prior = self.cvae.prior(features)  # [B, latent_dim]
            z = self.cvae.reparameterize(μ_prior, logσ2_prior)            # [B, latent_dim]
            μ_dec, logσ2_dec = self.cvae.decode(z, features)               # [B, param_dim]
            θ = self.cvae.reparameterize(μ_dec, logσ2_dec)                 # [B, param_dim]

            # 2) Warp input and compute logits + CE loss
            x_t = self.transformation_problem.transform(x, θ)
            if self.conf_module is None:
                logits = self.main_model(x_t)                              # [B, num_classes]
                ce_loss = F.cross_entropy(logits, y)
            else:
                conf, logits = self.conf_module(x_t, y=y)
                ce_loss = (-conf).mean(0)

            # 3) Boundary (or L₂) penalty on sampled θ
            if self.use_l2_instead_of_per_transform:
                boundary_reg = torch.norm(θ, dim=1).mean()
            else:
                boundary_reg = self.transformation_problem.boundary_violation(θ).mean()

            # 4) Decoder‐based CVAE regularizers (on μ_dec, logσ²_dec)
            decoder_reg = _compute_cvae_reg_on_decoder(
                mu_dec=μ_dec,
                log_sigma2_dec=logσ2_dec,
                transform_problem=self.transformation_problem,
                reg_options=self.reg_options,
                beta_uniform=self.beta_uniform,
                gamma_normal=self.gamma_normal
            )

            # 5) Optional: negative ELBO on sampled θ for reconstruction learning
            if self.include_elbo_in_classification and (self.extra_samples > 1):
                # a) q(z|θ,x)
                μ_enc, logσ2_enc = self.cvae.encode(θ, features)              # [B, latent_dim]
                z_enc = self.cvae.reparameterize(μ_enc, logσ2_enc)           # [B, latent_dim]
                # b) p(θ|z_enc, x)
                μ_dec_enc, logσ2_dec_enc = self.cvae.decode(z_enc, features)  # [B, param_dim]
                # c) rec_term_enc = E_q[ log p(θ|z_enc, x) ]
                rec_term_enc = -0.5 * (
                    torch.sum((θ - μ_dec_enc).pow(2) / logσ2_dec_enc.exp(), dim=-1)
                    + torch.sum(logσ2_dec_enc, dim=-1)
                    + self.param_dim * math.log(2 * math.pi)
                )  # [B]
                # d) prior p(z|x)
                μ_prior_enc, logσ2_prior_enc = self.cvae.prior(features)      # [B, latent_dim]
                # e) KL_enc = KL[q(z|θ,x) || p(z|x)]
                kl_enc = 0.5 * torch.sum(
                    logσ2_prior_enc - logσ2_enc
                    + (logσ2_enc.exp() + (μ_enc - μ_prior_enc).pow(2)) / logσ2_prior_enc.exp()
                    - 1,
                    dim=-1
                )  # [B]
                elbo_sample = rec_term_enc - self.beta_cvae * kl_enc          # [B]
                elbo_reg = -elbo_sample.mean()                                # scalar
            else:
                elbo_reg = torch.tensor(0.0, device=x.device)

            total_ce           += ce_loss
            total_boundary     += boundary_reg
            total_decoder_reg  += decoder_reg
            total_elbo_reg     += elbo_reg

        # Average over samples
        total_ce          /= num_samples
        total_boundary    /= num_samples
        total_decoder_reg /= num_samples
        total_elbo_reg    /= num_samples

        # Total loss: CE + l2_weight * boundary + kl_weight * (decoder_reg + elbo_reg)
        total_loss = (
            total_ce
            + self.l2_weight * total_boundary
            + self.kl_weight * (total_decoder_reg + total_elbo_reg)
        )

        # Return only one set of logits (from the last iteration)
        return total_loss, total_ce, total_boundary, total_decoder_reg, total_elbo_reg, logits

    def classification_losses(self, x: torch.Tensor, y: torch.Tensor):
        """
        Wrapper to match earlier API: returns
            total_loss, ce_loss, boundary_reg, decoder_reg, elbo_reg, logits
        """
        return self._compute_classification_step(x, y)

    def direct_loss(self, x: torch.Tensor, true_params: torch.Tensor):
        """
        Supervised regression on ground‐truth θ:
          L_reg = -ELBO(true_params; x)
                + l2_weight * boundary(true_params)
                + [boundary + decoder regs on sampled θ if extra_samples > 1]

        Returns:
          total_loss, nll_loss, boundary_reg, decoder_reg, elbo_true

        Where:
          - elbo_true = mean ELBO(true_params; x)  (before negation)
          - nll_loss  = - elbo_true
          - boundary_reg = boundary(true_params) + boundary(sampled θ) if extra_samples>1, else only boundary(true_params)
          - decoder_reg = decoder penalties averaged over extra_samples, or 0 if extra_samples ≤ 1
        """
        B = x.shape[0]
        features = self._get_cvae_features(x)                          # [B, localization_dim]

        # ---- Compute supervised ELBO on true_params ----
        elbo_true = self.cvae.log_prob(true_params, features, beta=self.beta_cvae)  # [B]
        nll_loss = -elbo_true.mean()                                        # scalar

        # ---- Boundary on true_params ----
        if self.use_l2_instead_of_per_transform:
            boundary_true = torch.norm(true_params, dim=1).mean()
        else:
            boundary_true = self.transformation_problem.boundary_violation(true_params).mean()

        # ---- If extra_samples > 1, sample additional θ and compute boundary + decoder regs ----
        if self.extra_samples > 1:
            total_boundary_sample = 0.0
            total_decoder_reg = 0.0
            for _ in range(self.extra_samples):
                μ_prior, logσ2_prior = self.cvae.prior(features)              # [B, latent_dim]
                z = self.cvae.reparameterize(μ_prior, logσ2_prior)            # [B, latent_dim]
                μ_dec, logσ2_dec = self.cvae.decode(z, features)               # [B, param_dim]
                θ = self.cvae.reparameterize(μ_dec, logσ2_dec)                 # [B, param_dim]

                # Boundary on sampled θ
                if self.use_l2_instead_of_per_transform:
                    boundary_sample = torch.norm(θ, dim=1).mean()
                else:
                    boundary_sample = self.transformation_problem.boundary_violation(θ).mean()

                # Decoder‐based penalties
                decoder_reg = _compute_cvae_reg_on_decoder(
                    mu_dec=μ_dec,
                    log_sigma2_dec=logσ2_dec,
                    transform_problem=self.transformation_problem,
                    reg_options=self.reg_options,
                    beta_uniform=self.beta_uniform,
                    gamma_normal=self.gamma_normal
                )

                total_boundary_sample += boundary_sample
                total_decoder_reg    += decoder_reg

            total_boundary_sample /= self.extra_samples
            total_decoder_reg    /= self.extra_samples
        else:
            total_boundary_sample = torch.tensor(0.0, device=x.device)
            total_decoder_reg    = torch.tensor(0.0, device=x.device)

        # ---- Combine boundaries and regs ----
        boundary_reg = boundary_true + total_boundary_sample
        decoder_reg  = total_decoder_reg

        # ---- Total loss: nll + l2_weight * boundary_reg + kl_weight * decoder_reg ----
        total_loss = nll_loss + self.l2_weight * boundary_reg + self.kl_weight * decoder_reg
        return total_loss, nll_loss, boundary_reg, decoder_reg, elbo_true.mean()

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """
        Inference: sample θ ~ p(θ|x) or use deterministic mean if desired externally.
        Returns logits (and optionally θ, μ_dec, logσ²_dec, μ_prior, logσ²_prior).
        """
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)
        features = self.localization_net(x).flatten(1)  # [B, localization_dim]

        # Sample from prior
        μ_prior, logσ2_prior = self.cvae.prior(features)      # [B, latent_dim]
        z = self.cvae.reparameterize(μ_prior, logσ2_prior)    # [B, latent_dim]
        μ_dec, logσ2_dec = self.cvae.decode(z, features)      # [B, param_dim]
        θ = self.cvae.reparameterize(μ_dec, logσ2_dec)        # [B, param_dim]

        x_t = self.transformation_problem.transform(x, θ)
        if self.conf_module is None:
            logits = self.main_model(x_t)
        else:
            conf, logits = self.conf_module(x_t)

        if return_stats:
            return logits, θ, μ_dec, logσ2_dec, μ_prior, logσ2_prior
        return logits

    def transform_input(self, x: torch.Tensor, deterministic: bool = False):
        """
        Apply deterministic (mean) transformation or sample:
          - deterministic=True: θ = μ_dec(z=μ_prior, x)
          - else: sample θ ~ p(θ|x)
        Returns warped x.
        """
        if self.pretransform is not None:
            with torch.no_grad():
                x = self.pretransform(x)
        features = self.localization_net(x).flatten(1)  # [B, localization_dim]

        if deterministic:
            μ_prior, logσ2_prior = self.cvae.prior(features)       # [B, latent_dim]
            z = μ_prior                                            # mean of p(z|x)
            μ_dec, _ = self.cvae.decode(z, features)               # [B, param_dim]
            θ = μ_dec
        else:
            μ_prior, logσ2_prior = self.cvae.prior(features)
            z = self.cvae.reparameterize(μ_prior, logσ2_prior)
            μ_dec, logσ2_dec = self.cvae.decode(z, features)
            θ = self.cvae.reparameterize(μ_dec, logσ2_dec)

        return self.transformation_problem.transform(x, θ)

    def training_step(self, batch, batch_idx):
        """
        Standard classification training step: assumes batch = (x, y).
        """
        x, y = batch
        total_loss, ce_loss, boundary_reg, decoder_reg, elbo_reg, logits = \
            self.classification_losses(x, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({
            "train_total_loss": total_loss,
            "train_ce_loss": ce_loss,
            "train_boundary_reg": boundary_reg,
            "train_decoder_reg": decoder_reg,
            "train_elbo_reg": elbo_reg,
            "train_acc": acc,
        }, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Standard validation step for classification: batch = (x, y).
        """
        x, y = batch
        total_loss, ce_loss, boundary_reg, decoder_reg, elbo_reg, logits = \
            self.classification_losses(x, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({
            "val_total_loss": total_loss,
            "val_ce_loss": ce_loss,
            "val_boundary_reg": boundary_reg,
            "val_decoder_reg": decoder_reg,
            "val_elbo_reg": elbo_reg,
            "val_acc": acc,
        }, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        """
        Optimize:
          • localization_net parameters
          • CVAE parameters
          • optionally main_model parameters if not frozen
        """
        params = [
            {"params": self.localization_net.parameters()},
            {"params": self.cvae.parameters()},
        ]
        if self.train_main:
            params.append({"params": self.main_model.parameters()})

        optimizer = self.optimizer_class(params, **self.optimizer_params)
        if self.lr_scheduler:
            scheduler = self.lr_scheduler(optimizer, **(self.lr_scheduler_params or {}))
            sched_conf = self.lr_config or {"interval": "epoch", "monitor": "val_total_loss"}
            sched_conf["scheduler"] = scheduler
            return {"optimizer": optimizer, "lr_scheduler": sched_conf}
        return optimizer
