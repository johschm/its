# File: `search/scipy_wrapper.py`
import numpy as np
import torch
from typing import Optional, Dict

from scipy.optimize import dual_annealing, direct, shgo, differential_evolution

def _bounds_np_from_problem(transformation_problem):
    ts = transformation_problem.transform_sequence
    lb = ts.lower_bounds.detach().cpu().numpy().astype(float)
    ub = ts.upper_bounds.detach().cpu().numpy().astype(float)
    bounds = [(float(lb[i]), float(ub[i])) for i in range(lb.shape[0])]
    return lb, ub, bounds

class ScipyDualAnnealing:
    def __init__(self, maxiter: int = 1000, initial_temp: Optional[float] = None, seed: Optional[int] = None, **kwargs):
        self.maxiter = maxiter
        self.initial_temp = initial_temp
        self.seed = seed
        self.kwargs = kwargs

    def optimize(self, transformation_problem, x: torch.Tensor, y: torch.Tensor = None, verbose: bool = False):
        device = x.device
        batch_size = x.size(0)
        dim = transformation_problem.calc_complete_size()
        best_params = torch.zeros(batch_size, dim, device=device)

        _, _, bounds = _bounds_np_from_problem(transformation_problem)

        for b in range(batch_size):
            def obj_np(xx):
                xx = np.asarray(xx, dtype=float)
                p_t = torch.from_numpy(xx).to(device=device,
                                             dtype=transformation_problem.transform_sequence.dtype).view(1, -1)
                with torch.no_grad():
                    err, _ = transformation_problem.calculate_error(
                        x[b:b+1], p_t, y[b:b+1] if y is not None else None
                    )
                return float(err.view(-1)[0].item())

            opt_kwargs = dict(**self.kwargs)
            if self.seed is not None:
                opt_kwargs.setdefault("seed", self.seed)

            res = dual_annealing(obj_np, bounds=bounds, **opt_kwargs)
            best_params[b] = torch.from_numpy(np.asarray(res.x, dtype=float)).to(
                device=device,
                dtype=transformation_problem.transform_sequence.dtype
            )
            if verbose:
                print(f"[DualAnnealing] batch {b} fun {res.fun} success {res.success} msg:{res.message}")

        final_params = best_params
        with torch.no_grad():
            final_err, final_other = transformation_problem.calculate_error(x, final_params, y)
        if final_err.ndim == 1:
            final_err = final_err.view(batch_size, 1)
        if final_other is not None and final_other.ndim == 2:
            final_other = final_other.view(batch_size, 1, -1)
        return transformation_problem.consolidate(x, final_params.unsqueeze(1), final_err, final_other)


class ScipyDIRECT:
    def __init__(self, maxfev: int = 200, locally_biased: bool = True, len_tol: float = 1e-6, eps: float = 1e-4, **kwargs):
        self.maxfev = maxfev
        self.locally_biased = locally_biased
        self.len_tol = len_tol
        self.eps = eps
        self.kwargs = kwargs

    def optimize(self, transformation_problem, x: torch.Tensor, y: torch.Tensor = None, verbose: bool = False):
        device = x.device
        batch_size = x.size(0)
        dim = transformation_problem.calc_complete_size()
        best_params = torch.zeros(batch_size, dim, device=device)

        _, _, bounds = _bounds_np_from_problem(transformation_problem)

        for b in range(batch_size):
            def obj_np(xx):
                xx = np.asarray(xx, dtype=float)
                p_t = torch.from_numpy(xx).to(device=device,
                                             dtype=transformation_problem.transform_sequence.dtype).view(1, -1)
                with torch.no_grad():
                    err, _ = transformation_problem.calculate_error(
                        x[b:b+1], p_t, y[b:b+1] if y is not None else None
                    )
                return float(err.view(-1)[0].item())

            opt_kwargs = dict(
                maxfev=self.maxfev,
                locally_biased=self.locally_biased,
                len_tol=self.len_tol,
                eps=self.eps,
                **self.kwargs
            )

            res = direct(obj_np, bounds=bounds, **opt_kwargs)
            best_params[b] = torch.from_numpy(np.asarray(res.x, dtype=float)).to(
                device=device, dtype=transformation_problem.transform_sequence.dtype
            )
            if verbose:
                print(f"[DIRECT] batch {b} fun {res.fun} success {res.success} msg:{res.message} nfev:{getattr(res,'nfev',None)}")

        final_params = best_params
        with torch.no_grad():
            final_err, final_other = transformation_problem.calculate_error(x, final_params, y)
        if final_err.ndim == 1:
            final_err = final_err.view(batch_size, 1)
        if final_other is not None and final_other.ndim == 2:
            final_other = final_other.view(batch_size, 1, -1)
        return transformation_problem.consolidate(x, final_params.unsqueeze(1), final_err, final_other)


class ScipySHGO:
    def __init__(self, n: int = 100, iters: int = 1, minimizer_kwargs: Optional[Dict] = None, workers: int = 1, **kwargs):
        self.n = n
        self.iters = iters
        self.minimizer_kwargs = minimizer_kwargs or {}
        self.workers = workers
        self.kwargs = kwargs

    def optimize(self, transformation_problem, x: torch.Tensor, y: torch.Tensor = None, verbose: bool = False):
        device = x.device
        batch_size = x.size(0)
        dim = transformation_problem.calc_complete_size()
        best_params = torch.zeros(batch_size, dim, device=device)

        _, _, bounds = _bounds_np_from_problem(transformation_problem)

        for b in range(batch_size):
            def obj_np(xx):
                xx = np.asarray(xx, dtype=float)
                p_t = torch.from_numpy(xx).to(device=device,
                                             dtype=transformation_problem.transform_sequence.dtype).view(1, -1)
                with torch.no_grad():
                    err, _ = transformation_problem.calculate_error(
                        x[b:b+1], p_t, y[b:b+1] if y is not None else None
                    )
                return float(err.view(-1)[0].item())

            opt_kwargs = dict(n=self.n, iters=self.iters,
                              minimizer_kwargs=self.minimizer_kwargs, workers=self.workers, **self.kwargs)

            res = shgo(obj_np, bounds=bounds, **opt_kwargs)
            best_params[b] = torch.from_numpy(np.asarray(res.x, dtype=float)).to(
                device=device, dtype=transformation_problem.transform_sequence.dtype
            )
            if verbose:
                print(f"[SHGO] batch {b} fun {res.fun} success {res.success} nfev:{getattr(res,'nfev',None)} nlfev:{getattr(res,'nlfev',None)}")

        final_params = best_params
        with torch.no_grad():
            final_err, final_other = transformation_problem.calculate_error(x, final_params, y)
        if final_err.ndim == 1:
            final_err = final_err.view(batch_size, 1)
        if final_other is not None and final_other.ndim == 2:
            final_other = final_other.view(batch_size, 1, -1)
        return transformation_problem.consolidate(x, final_params.unsqueeze(1), final_err, final_other)


class ScipyDifferentialEvolution:
    def __init__(self,
                 maxiter: int = 50,
                 popsize: int = 15,
                 tol: float = 1e-6,
                 mutation=(0.5, 1.0),
                 recombination: float = 0.7,
                 strategy: str = 'best1bin',
                 seed: int = None,
                 vectorized: bool = True,
                 polish: bool = True,
                 updating: str = 'immediate',
                 workers = 1,
                 **kwargs):
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.strategy = strategy
        self.seed = seed
        self.vectorized = vectorized
        self.polish = polish
        self.updating = updating
        self.workers = workers
        self.extra_kwargs = kwargs

    def optimize(self, transformation_problem, x: torch.Tensor, y: torch.Tensor = None, verbose: bool = False):
        device = x.device
        batch_size = x.size(0)
        dim = transformation_problem.calc_complete_size()
        best_params = torch.zeros(batch_size, dim, device=device,
                                  dtype=transformation_problem.transform_sequence.dtype)

        _, _, bounds = _bounds_np_from_problem(transformation_problem)

        for b in range(batch_size):
            def obj_np(xx):
                xx = np.asarray(xx, dtype=float)
                if xx.ndim == 1:
                    param_t = torch.from_numpy(xx).to(device=device,
                                                    dtype=transformation_problem.transform_sequence.dtype).view(1, -1)
                    with torch.no_grad():
                        err, _ = transformation_problem.calculate_error(
                            x[b:b+1], param_t, y[b:b+1] if y is not None else None
                        )
                    return float(err.view(-1)[0].item())
                elif xx.ndim == 2:
                    pts = torch.from_numpy(xx).to(device=device,
                                                 dtype=transformation_problem.transform_sequence.dtype).view(1, xx.shape[0], dim)
                    with torch.no_grad():
                        errs, _ = transformation_problem.calculate_error(
                            x[b:b+1], pts, y[b:b+1] if y is not None else None
                        )
                        return errs.view(-1).detach().cpu().numpy().astype(float)
                else:
                    raise ValueError("Unexpected array shape passed to obj_np")

            de_kwargs = dict(
                maxiter=self.maxiter,
                popsize=self.popsize,
                tol=self.tol,
                mutation=self.mutation,
                recombination=self.recombination,
                strategy=self.strategy,
                seed=self.seed,
                vectorized=self.vectorized,
                polish=self.polish,
                updating=self.updating,
                workers=self.workers,
                **self.extra_kwargs
            )

            if verbose:
                print(f"[DE] batch {b} start (dim={dim}) bounds[0]={bounds[0]} ...")
            res = differential_evolution(func=obj_np, bounds=bounds, **de_kwargs)
            if verbose:
                print(f"[DE] batch {b} done fun={res.fun} nit={getattr(res, 'nit', None)} success={res.success}")

            best_params[b] = torch.from_numpy(np.asarray(res.x, dtype=float)).to(
                device=device, dtype=transformation_problem.transform_sequence.dtype
            )

        final_params = best_params
        with torch.no_grad():
            final_err, final_other = transformation_problem.calculate_error(x, final_params, y)
        if final_err.ndim == 1:
            final_err = final_err.view(batch_size, 1)
        if final_other is not None and final_other.ndim == 2:
            final_other = final_other.view(batch_size, 1, -1)
        return transformation_problem.consolidate(x, final_params.unsqueeze(1), final_err, final_other)