import torch
from typing import Optional


class EvolutionarySearch:
    """
    Vectorized population-based global search (per-batch ES-like).
    - Maintains per-batch mean (mu) and per-batch per-dim sigma.
    - At each iteration samples `population_size` candidates per batch from N(mu, sigma).
    - Evaluates candidates in one batched forward pass through transformation_problem.
    - Updates mu and sigma from top-k selected candidates (selection_frac).
    - Keeps track of best found params and best error across iterations.

    Expects transformation_problem to implement:
      - initial_param(batch_size, n_samples) -> tensor (batch, n_samples, dim) or (batch*n_samples, dim)
      - calculate_error(x, params, y=None) -> (errors, other). errors must be shaped (batch*n_samples,) or (batch*n_samples,1)
      - consolidate(x, final_params, final_err, final_other)
    """
    def __init__(
        self,
        population_size: int = 12,
        num_iters: int = 20,
        selection_frac: float = 0.2,
        sigma_decay: float = 0.99,
        sigma_min: float = 1e-6,
        seed: Optional[int] = None,
        project_param: bool = True,  # NEW
    ):
        assert 0.0 < selection_frac <= 1.0, "selection_frac must be > 0 and <= 1"
        self.population_size = int(population_size)
        self.num_iters = int(num_iters)
        self.selection_frac = float(selection_frac)
        self.sigma_decay = float(sigma_decay)
        self.sigma_min = float(sigma_min)
        if seed is not None:
            torch.manual_seed(seed)
        self.project_param = project_param

    def _batched_evaluate(
        self,
        transformation_problem,
        x: torch.Tensor,
        candidates: torch.Tensor,
        y: Optional[torch.Tensor],
    ):
        """
        candidates: (batch, n, dim)
        returns: errs (batch, n), other (batch, n, ... ) or None
        """
        batch, n, dim = candidates.shape
        flat = candidates.reshape(-1, dim)
        # replicate x and y appropriately
        x_rep = x.repeat_interleave(n, dim=0)
        y_rep = y.repeat_interleave(n, dim=0) if y is not None else None

        with torch.no_grad():
            errs, other = transformation_problem.calculate_error(x_rep, flat, y_rep)
            # errs could be shape (batch*n,) or (batch*n,1). Normalize to (batch, n)
            errs = errs.reshape(batch, n)
            if other is not None:
                # try to reshape other to (batch, n, -1) if possible (safe reshape)
                try:
                    other = other.reshape(batch, n, -1)
                except Exception:
                    # fallback: keep as None if reshape fails
                    other = None
        return errs, other

    def optimize(
        self,
        transformation_problem,
        x: torch.Tensor,
        y: torch.Tensor = None,
        verbose: bool = False,
    ):
        device = x.device
        dtype = x.dtype
        batch_size = x.size(0)
        n = self.population_size

        # 1) initialize population via transformation_problem.initial_param (keeps same API)
        with torch.no_grad():
            init = transformation_problem.initial_param(batch_size, n)
        # normalize shapes: expect (batch, n, dim) or (batch*n, dim)
        if init.dim() == 2 and init.size(0) == batch_size * n:
            dim = init.size(1)
            init = init.reshape(batch_size, n, dim)
        else:
            init = init.reshape(batch_size, n, -1)
            dim = init.size(2)
        init = init.to(device=device, dtype=dtype)

        # compute initial mu and sigma (per-batch, per-dim)
        mu = init.mean(dim=1)            # (batch, dim)
        sigma = init.std(dim=1)          # (batch, dim)
        sigma = torch.clamp(sigma, min=self.sigma_min)

        # Evaluate initial population and set bests
        errs, other = self._batched_evaluate(transformation_problem, x, init, y)
        best_vals, best_idx = errs.min(dim=1)  # (batch,), (batch,)
        best_params = init[torch.arange(batch_size, device=device), best_idx]  # (batch, dim)
        best_err = best_vals.clone()  # (batch,)
        best_other = None
        if other is not None:
            best_other = other[torch.arange(batch_size, device=device), best_idx] # (batch, k)

        if verbose:
            print(f"[ES] init best_err mean: {best_err.mean().item():.6f}")

        k = max(1, int(round(self.selection_frac * n)))  # number selected each iteration

        # Main loop
        for it in range(self.num_iters):
            # Sample new population around mu using current sigma
            # shape: (batch, n, dim)
            eps = torch.randn(batch_size, n, dim, device=device, dtype=dtype)
            # sigma is (batch, dim) -> expand to (batch, n, dim)
            samp = mu.unsqueeze(1) + eps * sigma.unsqueeze(1)
            if self.project_param:
                flat = samp.reshape(-1, dim)
                flat = transformation_problem.correct_param(flat)
                samp = flat.reshape(batch_size, n, dim)
            else:
                flat = samp.reshape(-1, dim)
                flat = transformation_problem.normalize(flat)
                samp = flat.reshape(batch_size, n, dim)

            # Evaluate candidates
            errs, other = self._batched_evaluate(transformation_problem, x, samp, y)
            # Update global bests
            cur_best_vals, cur_best_idx = errs.min(dim=1)  # (batch,), (batch,)
            improved_mask = cur_best_vals < best_err
            if improved_mask.any():
                idxs = torch.nonzero(improved_mask, as_tuple=False).squeeze(1)
                best_err[idxs] = cur_best_vals[idxs]
                best_params[idxs] = samp[idxs, cur_best_idx[idxs]]
                if other is not None and best_other is not None:
                    best_other[idxs] = other[idxs, cur_best_idx[idxs]]

            # Select top-k per batch (lowest errors)
            # topk with largest=False gives smallest errors
            topk_vals, topk_idx = torch.topk(errs, k=k, largest=False, dim=1)
            # gather selected candidates: shape (batch, k, dim)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, dim)
            selected = torch.gather(samp, dim=1, index=idx_exp)

            # Update mu and sigma (we use simple recombination and an EMA for sigma)
            new_mu = selected.mean(dim=1)  # (batch, dim)
            sel_std = selected.std(dim=1)

            # EMA update for sigma (keeps stability)
            sigma = torch.clamp(self.sigma_decay * sigma + (1.0 - self.sigma_decay) * sel_std, min=self.sigma_min)

            mu = new_mu

            if verbose and ((it + 1) % max(1, self.num_iters // 10) == 0 or it == 0):
                print(f"[ES] iter {it+1}/{self.num_iters}, best_err mean: {best_err.mean().item():.6f}")

        # Final evaluation of best_params
        with torch.no_grad():
            final_err = best_err
            final_other = best_other
            # ensure final_err is (batch, 1)
            if final_err.ndim == 1:
                final_err = final_err.reshape(batch_size, 1)
            else:
                final_err = final_err.reshape(batch_size, -1)
                if final_err.size(1) != 1:
                    final_err = final_err.mean(dim=1, keepdim=True)

            # final_other -> (batch, 1, -1) if possible
            if final_other is None:
                final_other = torch.zeros(batch_size, 1, 0, device=device)
            else:
                try:
                    final_other = final_other.reshape(batch_size, -1)  # (batch, nfeatures)
                    final_other = final_other.unsqueeze(1)         # (batch, 1, nfeatures)
                except Exception:
                    # fallback: empty placeholder
                    final_other = torch.zeros(batch_size, 1, 0, device=device)

            final_params = best_params.unsqueeze(1)  # (batch, 1, dim)

        return transformation_problem.consolidate(
            x, final_params, final_err, final_other
        )

from typing import Optional
import torch


class CMAES:
    """
    Batched, vectorized CMA-ES that integrates with your transformation_problem API.

    Expects transformation_problem to implement:
      - initial_param(batch_size, n_samples)
      - calculate_error(x, params, y=None)
      - consolidate(x, final_params, final_err, final_other)
      - correct_param(flat_params) and normalize(flat_params) for projecting samples
    """

    def __init__(
        self,
        population_size: Optional[int] = None,   # lambda (default 4 + floor(3*log(d)))
        mu_par: Optional[int] = None,            # number of parents (default lambda//2)
        sigma0: float = 0.5,                     # initial multiplier for sigma (scale of init std)
        num_iters: int = 100,
        seed: Optional[int] = None,
        project_param: bool = True,
        tol: float = 1e-12,                      # used now for a simple early stop (sigma mean)
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.population_size = population_size
        self.mu_par_override = mu_par
        self.sigma0 = float(sigma0)
        self.num_iters = int(num_iters)
        self.project_param = bool(project_param)
        self.tol = float(tol)

        # small constant for numerical stability
        self.EPS = 1e-12

    def _batched_evaluate(self, transformation_problem, x, candidates, y):
        """
        candidates: (batch, n, dim)
        returns: errs (batch, n), other (batch, n, k) or None
        """
        batch, n, dim = candidates.shape
        flat = candidates.reshape(-1, dim)
        x_rep = x.repeat_interleave(n, dim=0)
        y_rep = y.repeat_interleave(n, dim=0) if y is not None else None

        with torch.no_grad():
            errs, other = transformation_problem.calculate_error(x_rep, flat, y_rep)

            # normalize errs to (batch, n)
            try:
                errs = errs.reshape(batch, n)
            except Exception:
                errs = errs.view(batch, n)

            # normalize other if provided
            if other is None:
                other_out = None
            else:
                # try the most common shapes: (batch*n,), (batch*n,k), (batch,n,k)
                try:
                    # prefer (batch, n, k)
                    other_out = other.reshape(batch, n, -1)
                except Exception:
                    try:
                        # maybe other is 1D per sample: (batch*n,)
                        other_tmp = other.reshape(batch, n)
                        other_out = other_tmp.unsqueeze(-1)  # (batch,n,1)
                    except Exception:
                        # Can't reshape in expected way — drop it to None to avoid silent errors
                        other_out = None
            return errs, other_out

    def optimize(self, transformation_problem, x, y=None, verbose=False):
        """
        Main optimizer entry. Returns transformation_problem.consolidate(...) output.
        """

        device, dtype = x.device, x.dtype
        batch_size = x.size(0)

        # ----------------- infer dim & lambda -----------------
        lam_provided = self.population_size
        with torch.no_grad():
            probe_lam = lam_provided or 1
            init_probe = transformation_problem.initial_param(batch_size, probe_lam)

        if init_probe.dim() == 2 and init_probe.size(0) == batch_size * probe_lam:
            dim = init_probe.size(1)
        else:
            dim = init_probe.reshape(batch_size, probe_lam, -1).size(2)

        if lam_provided is None:
            lam = int(4 + torch.floor(3 * torch.log(torch.tensor(dim, dtype=torch.float32))).item())
        else:
            lam = int(lam_provided)

        if self.mu_par_override is None:
            mu_par = max(1, lam // 2)
        else:
            mu_par = int(self.mu_par_override)

        # ensure valid bounds
        mu_par = max(1, min(mu_par, lam))

        # ----------------- recombination weights -----------------
        eps = self.EPS
        ranks = torch.arange(1, mu_par + 1, device=device, dtype=dtype)
        raw = (torch.log(torch.tensor(float(mu_par) + 0.5, device=device, dtype=dtype) + eps)
               - torch.log(ranks + eps))
        raw = torch.clamp(raw, min=0.0)
        if raw.sum() <= 0:
            raw = torch.ones_like(raw)
        weights = (raw / (raw.sum() + eps)).to(device=device, dtype=dtype)  # shape (mu_par,)
        # ensure weights are float and on device
        weights = weights.to(dtype=dtype, device=device)
        mueff = (weights.sum() ** 2) / (weights.pow(2).sum() + eps)

        # ----------------- strategy parameters -----------------
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / (mueff + eps)) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, ((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = (dim ** 0.5) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # ----------------- initialize population samples -----------------
        with torch.no_grad():
            init = transformation_problem.initial_param(batch_size, lam)

        # normalize/reshape init to (batch, lam, dim)
        init = init.reshape(batch_size, lam, -1).to(device=device, dtype=dtype)

        mu = init.mean(dim=1)  # (batch, dim)

        # ----------------- IMPORTANT: initialize sigma as scalar per batch from empirical std of init -----------------
        # use average std across dims for each batch to produce a scalar sigma
        init_std = init.std(dim=1).mean(dim=1)                 # (batch,)
        init_std = torch.clamp(init_std, min=1e-8)
        sigma = (init_std * float(self.sigma0)).unsqueeze(1)   # (batch,1) scalar per batch

        # ----------------- initialize C, B, D and evolution paths -----------------
        C = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        B = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        D = torch.ones((batch_size, dim), device=device, dtype=dtype)

        pc = torch.zeros((batch_size, dim), device=device, dtype=dtype)
        ps = torch.zeros((batch_size, dim), device=device, dtype=dtype)

        # ----------------- evaluate initial population and set bests -----------------
        errs, other = self._batched_evaluate(transformation_problem, x, init, y)
        best_vals, best_idx = errs.min(dim=1)
        best_params = init[torch.arange(batch_size, device=device), best_idx]
        best_err = best_vals.clone()
        best_other = None
        if other is not None:
            best_other = other[torch.arange(batch_size, device=device), best_idx]

        if verbose:
            print(f"[CMA-ES] init best_err mean: {best_err.mean().item():.6f}")

        # ----------------- robust eigendecomposition helper -----------------
        def _robust_eigh_batch(Cmat, min_eig=1e-20, initial_jitter=1e-12, max_tries=6):
            batch_local, dim_local, _ = Cmat.shape
            C_sym = 0.5 * (Cmat + Cmat.transpose(-1, -2))
            eye = torch.eye(dim_local, device=Cmat.device, dtype=Cmat.dtype).unsqueeze(0).repeat(batch_local, 1, 1)
            try:
                eigvals, eigvecs = torch.linalg.eigh(C_sym)
                eigvals = torch.clamp(eigvals, min=min_eig)
                return eigvals, eigvecs
            except RuntimeError:
                jitter = initial_jitter
                for attempt in range(max_tries):
                    try:
                        C_try = C_sym + eye * jitter
                        eigvals, eigvecs = torch.linalg.eigh(C_try)
                        eigvals = torch.clamp(eigvals, min=min_eig)
                        return eigvals, eigvecs
                    except RuntimeError:
                        jitter *= 10.0
                # final fallback to SVD
                C_fallback = C_sym + eye * jitter
                U, S, Vh = torch.linalg.svd(C_fallback)
                eigvals = torch.clamp(S, min=min_eig)
                eigvecs = U
                return eigvals, eigvecs

        # ----------------- main loop -----------------
        for it in range(self.num_iters):
            # draw z ~ N(0,I) and shape via B, D
            z = torch.randn(batch_size, lam, dim, device=device, dtype=dtype)
            z_scaled = z * D.unsqueeze(1)  # (batch, lam, dim)  [D multiplies per-dim]
            # y_samp are "internal coordinates" (B @ (D * z))
            y_samp = torch.einsum('bij,bkj->bki', B, z_scaled)  # (batch, lam, dim)

            # produce samples = mu + sigma * y_samp
            # sigma is (batch,1) -> broadcast to (batch, lam, dim)
            samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * y_samp  # (batch, lam, dim)

            # projection/normalization step
            flat = samples.reshape(-1, dim)
            if self.project_param:
                flat = transformation_problem.correct_param(flat)
            else:
                flat = transformation_problem.normalize(flat)
            samples = flat.reshape(batch_size, lam, dim)

            # evaluate
            errs, other = self._batched_evaluate(transformation_problem, x, samples, y)

            # update bests
            cur_best_vals, cur_best_idx = errs.min(dim=1)
            improved = cur_best_vals < best_err
            if improved.any():
                idxs = torch.nonzero(improved, as_tuple=False).squeeze(1)
                best_err[idxs] = cur_best_vals[idxs]
                best_params[idxs] = samples[idxs, cur_best_idx[idxs]]
                if other is not None and best_other is not None:
                    best_other[idxs] = other[idxs, cur_best_idx[idxs]]

            # select top mu_par (smallest errors)
            topk_vals, topk_idx = torch.topk(errs, k=mu_par, largest=False, dim=1)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, dim)
            sel = torch.gather(samples, 1, idx_exp)      # (batch, mu_par, dim)
            sel_y = torch.gather(y_samp, 1, idx_exp)     # (batch, mu_par, dim)
            sel_z = torch.gather(z, 1, idx_exp)          # (batch, mu_par, dim)

            # recombination weights: expand to batch dims for computation
            w = weights.reshape(1, mu_par, 1).to(device=device, dtype=dtype)

            old_mu = mu.clone()
            mu = torch.sum(w * sel, dim=1)  # (batch, dim)

            # ---- rank-mu covariance contribution (use internal coordinates sel_y) ----
            # cov_mu computed in internal coordinate space y (not multiplied by sigma^2 here).
            # We'll combine with c1 and cmu in full coordinate space (since x = mu + sigma * y).
            cov_mu = torch.einsum('bki,bkj->bij', (sel_y * w), sel_y)  # (batch, dim, dim)

            # ---- evolution path for sigma (ps) ----
            # y_w is weighted shaped displacement in internal coordinates (y)
            y_w = torch.sum(w * sel_y, dim=1)  # (batch, dim)
            # C^{-1/2} * y_w computed via B and D: invsqrt(C) = B diag(1/D) B^T
            Bt_yw = torch.einsum('bij,bj->bi', B.transpose(1, 2), y_w)  # (batch, dim)
            tmp = Bt_yw / (D + self.EPS)
            Cinv_sqrt_yw = torch.einsum('bij,bj->bi', B, tmp)  # (batch, dim)

            # update ps (evolution path for sigma)
            ps = (1 - cs) * ps + torch.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt_yw

            # compute hsig for pc update (use updated ps)
            norm_ps = ps.norm(dim=1)  # (batch,)

            # CORRECT denom: use 2*(it+1) exponent (no division by lam)
            denom = torch.sqrt(torch.clamp(1.0 - (1.0 - cs) ** (2.0 * (it + 1)), min=self.EPS))

            hsig_flag = (norm_ps / (chiN * denom + self.EPS)) < (1.4 + 2.0 / (dim + 1.0))
            hsig = hsig_flag.to(dtype=dtype).unsqueeze(1)  # (batch,1)

            # ---- update pc (rank-1 evolution path) ----
            # Important: divide by scalar sigma (broadcast), not vector sigma
            # (mu - old_mu) / sigma    => (batch, dim) / (batch,1) -> broadcasts correctly
            pc = (1 - cc) * pc + (hsig * torch.sqrt(cc * (2 - cc) * mueff)) * ((mu - old_mu) / (sigma + self.EPS))

            # ---- rank-one contribution ----
            rank_one = torch.einsum('bi,bj->bij', pc, pc)

            # ---- update C: combine base, rank-one and rank-mu ----
            # Note: both rank_one and cov_mu are in internal coordinates scaled by sigma appropriately.
            # We update C in full coordinate representation: use factors c1 and cmu and
            # multiply cov_mu by 1 (because our cov_mu is in internal coords and sigma^2 factor
            # should be applied consistently; given we use x = mu + sigma*y, the contribution
            # to C (in full coordinates) from y is sigma^2 * cov_mu. To remain consistent, multiply cov_mu by sigma^2
            sigma2 = (sigma.squeeze(1) ** 2).reshape(-1, 1, 1)  # (batch,1,1)
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * (cov_mu * sigma2)

            # enforce symmetry to avoid numerical drift
            C = 0.5 * (C + C.transpose(-1, -2))

            # ---- update sigma (global scale) using ps ----
            # standard CMA-ES: sigma *= exp((cs/damps)*(norm(ps)/chiN - 1))
            exponential_update = (norm_ps / (chiN + self.EPS)) - 1.0
            exponential_update = (cs / damps) * exponential_update
            exponential_update = torch.clamp(exponential_update, min=-50.0, max=50.0)
            factor = torch.exp(exponential_update).unsqueeze(1)  # (batch,1)

            sigma = sigma * factor
            # clamp sigma to avoid numerical underflow
            sigma = torch.clamp(sigma, min=1e-12)

            # ---- decompose C to B, D (robust) ----
            eigvals, eigvecs = _robust_eigh_batch(C)
            D = torch.sqrt(eigvals)  # (batch, dim)
            B = eigvecs             # (batch, dim, dim)

            # ---- diagnostics and verbose logging ----
            if verbose and ((it + 1) % max(1, self.num_iters // 10) == 0 or it == 0):
                try:
                    print(
                        f"[CMA-ES] iter {it+1}/{self.num_iters}, best_err mean: {best_err.mean().item():.6f}, "
                        f"sigma mean: {sigma.mean().item():.6e}, D max: {D.max().item():.6e}"
                    )
                except Exception:
                    print(f"[CMA-ES] iter {it+1}/{self.num_iters}, best_err mean: {best_err.mean().item():.6f}")

            # ---- simple early stopping by very small sigma ----
            if sigma.mean().item() < self.tol:
                if verbose:
                    print(f"[CMA-ES] early stop at iter {it+1} due to sigma < tol ({self.tol})")
                break

            # NOTE: no other early stopping requested; loop continues until num_iters or sigma tol

        # ----------------- final evaluation of best_params -----------------
        with torch.no_grad():
            final_err = best_err
            final_other = best_other

            # normalize final_err shape to (batch, 1)
            if final_err.ndim == 1:
                final_err = final_err.reshape(batch_size, 1)
            else:
                final_err = final_err.reshape(batch_size, -1)
                if final_err.size(1) != 1:
                    final_err = final_err.mean(dim=1, keepdim=True)

            # normalize final_other
            if final_other is None:
                final_other_out = None
            else:
                try:
                    # try (batch, k) -> (batch, 1, k)
                    final_other_tmp = final_other.reshape(batch_size, -1)
                    final_other_out = final_other_tmp.unsqueeze(1)
                except Exception:
                    final_other_out = None

            final_params = best_params.unsqueeze(1)  # (batch, 1, dim)

        return transformation_problem.consolidate(x, final_params, final_err, final_other_out)


import torch
import numpy as np
class CMAES_Nevergrad:
    pass