from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import yaml
import optuna
import math  # NEW

# Import cost / helper functions from objective_generators (no reverse import there)
from search.objective_generators import (
    _cost_shgo,
    _cost_parallel_sa,
    _cost_es,
    _cost_pso,
    _cost_cd,
    _cost_wcd,
    _cost_pgd,  # added earlier; now expects grad_weight
)

# ---------------------------
# Default allocations
# ---------------------------

def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def default_shgo_params(budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    n_init = int(0.90 * budget)
    #always allow at leaset one local run
    min_n_init = max(0, budget - grad_weight -1)
    if n_init > min_n_init > 0:
        n_init = min_n_init

    max_local_runs = max(1,int((budget - n_init)/(3 * grad_weight +1)))
    local_runs = max(1, int(max_local_runs * 0.5))
    per_run_budget = max(0, ((budget - n_init)- local_runs)// (local_runs * grad_weight)) #at least local_runs steps for convergence
    local_steps = per_run_budget
    # Adjust if cost overflow
    if _cost_shgo(n_init, local_runs, local_steps, grad_weight) > budget:
        raise ValueError("Error in budget allocation for SHGO")
    params = {
        "shgo_initial_samples": n_init,
        "shgo_local_runs": local_runs,
        "shgo_local_steps": local_steps,       # canonical (builder expects this)
        "shgo_selection_method": "knn",
        "shgo_local_opt": "adam",
        "shgo_lr": 1e-1,
        "shgo_acceptance_criterion": "step",
        "grad_weight": grad_weight,
    }
    # NEW: reassign any leftover budget to initial samples (n_init)
    current_cost = _cost_shgo(params["shgo_initial_samples"], params["shgo_local_runs"], params["shgo_local_steps"], grad_weight)
    leftover = budget - current_cost
    if leftover > 0:
        params["shgo_initial_samples"] += leftover
        # sanity check
        final_cost = _cost_shgo(params["shgo_initial_samples"], params["shgo_local_runs"], params["shgo_local_steps"], grad_weight)
        if final_cost != budget:
            raise ValueError(f"SHGO default allocation mismatch after top-up: cost={final_cost}, budget={budget}")
    return params

def default_parallel_sa_params(budget: int) -> Dict[str, Any]:
    parallel_runs = _clip(int(0.05 * budget), 1, max(1, budget // 5))
    max_iter = max(1, budget // max(1, parallel_runs))
    while _cost_parallel_sa(parallel_runs, max_iter) > budget and max_iter > 1:
        max_iter -= 1
        if max_iter < 0:
            raise ValueError("Cannot fit Parallel SA parameters within budget")
    return {
        "psa_parallel_runs": parallel_runs,
        "psa_max_iterations": max_iter,
        "psa_init_temp": 50.0,
        "psa_cooling": 0.95,
        "psa_reinit_interval": 9999999999999,  # Disabled by default
        "psa_reinit_amount": 0.0,  # Disabled by default
        "psa_neighbor_hood_size": 0.1,
    }

def default_parallel_sa_resets_params(budget: int) -> Dict[str, Any]:
    """
    Defaults for PSA-with-resets: same as normal PSA but with reinit enabled.
    """
    base_params = default_parallel_sa_params(budget)
    # Enable reinit with reasonable defaults
    base_params.update({
        "psa_reinit_interval": min(25, max(5, base_params["psa_max_iterations"] // 4)),
        "psa_reinit_amount": 0.2,
    })
    return base_params

def default_evolutionary_params(budget: int, min_pop: int = 4) -> Dict[str, Any]:
    # Choose iterations ~ sqrt(budget) heuristic
    iters = max(5, int(budget ** 0.5))
    if iters + 1 >= budget:  # fallback tiny budgets
        iters = max(1, budget // 2)
    pop_max = max(min_pop, budget // max(1, (iters + 1)))
    pop = pop_max
    while _cost_es(pop, iters) > budget and iters > 1:
        iters -= 1
        if iters < 0:
            raise ValueError("Cannot fit Evolutionary parameters within budget")
    return {
        "es_population": pop,
        "es_iters": iters,
        "es_selection_frac": 0.5,
        "es_sigma_decay": 0.98,
    }


def default_pso_params(budget: int, min_swarm: int = 4) -> Dict[str, Any]:
    # Use middle of the sampling range as default
    max_steps_possible = max(1, budget // min_swarm - 1)
    steps = max(1, max_steps_possible // 2)  # Middle of range [1, max_steps_possible]

    # Compute swarm size given steps
    swarm = max(min_swarm, budget // (steps + 1))

    # Ensure budget is not exceeded
    while _cost_pso(swarm, steps) > budget and steps > 1:
        steps -= 1

    return {
        "pso_swarm_size": swarm,
        "pso_steps": steps,
        "pso_w": 0.6,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
        "pso_clamp_velocity": True,
        "pso_vmax_scale": 0.2,
    }

def default_cd_params(budget: int, dim: Optional[int] = None, samples_min: int = 4) -> Dict[str, Any]:
    # Dimension-agnostic: uniform distribution computable later once dim known.
    # Intentionally do not store dim or samples here.
    return {
        # No cd_number_samples / cd_dim_estimate persisted intentionally.
        # Add a marker if needed for downstream logic (optional).
        # "cd_uniform": True
    }

def default_wcd_params(budget: int, dim: Optional[int] = None) -> Dict[str, Any]:
    # Dimension-agnostic: only persist cycles (rounds) and first-dim weight.
    rounds = max(1, min(5, int(0.1 * (budget ** 0.5))))
    if rounds == 0:
        rounds = 1
    first_factor = 2
    return {
        "wcd_rounds": rounds,
        "wcd_first_dim_factor": first_factor,
        # base and dim omitted; derive later when actual dim known
    }

def default_wcd_lattice_params(budget: int, dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Dimension-agnostic defaults for WCD with lattice initialization.
    Uses same parameter structure as regular WCD.
    """
    return default_wcd_params(budget, dim)

def default_cd_multi_cyclus_params(budget: int, dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Dimension-agnostic defaults for multi-round coordinate descent.
    Like WCD but with first_dim_factor fixed to 1 (no special weighting).
    """
    rounds = max(1, min(5, int(0.1 * (budget ** 0.5))))
    if rounds == 0:
        rounds = 1
    return {
        "wcd_rounds": rounds,
        "wcd_first_dim_factor": 1,  # Fixed to 1 for uniform CD
    }

# --- NEW: parallel gradient descent family defaults ---

def default_pgd_params(budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    # Pick reasonable number of parallel runs
    parallel_runs = _clip(
        int(0.05 * budget),
        2,
        max(2, min(16, budget // 10 if budget >= 20 else 4))
    )

    # Compute theoretical max_iterations that fits within the budget
    # cost = par_runs * (max_iter * grad_weight + 1) <= budget
    max_iterations = int((budget - parallel_runs) / (parallel_runs * grad_weight))
    max_iterations = max(1, max_iterations)

    # Ensure at least a small lower bound for practicality
    if max_iterations < 5 and budget > 20:
        max_iterations = 5

    # Adjust if rounding errors made cost exceed budget
    while _cost_pgd(parallel_runs, max_iterations, grad_weight) > budget and max_iterations > 1:
        max_iterations -= 1

    return {
        "pgd_parallel_runs": parallel_runs,
        "pgd_max_iterations": max_iterations,
        "pgd_learning_rate": 1e-1,
        "pgd_lr_decay_rate": 0.98,
        "pgd_optimizer": "adam",
        "grad_weight": grad_weight,
    }

def default_pgd_restart_params(budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    base = default_pgd_params(budget, grad_weight=grad_weight)
    # choose restart every ~10% of iterations
    interval = max(5, base["pgd_max_iterations"] // 10)
    return {
        **base,
        "pgdr_reinit_interval": interval,
        "pgdr_reinit_amount": 0.25,  # fraction of runs
    }

def default_pgd_window_params(budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    base = default_pgd_params(budget, grad_weight=grad_weight)
    return {
        **base,
        "pgdw_stuck_window_size": 6,
        "pgdw_improvement_threshold": 1e-3,
        "pgdw_flat_threshold": 1e-4,
        "pgdw_oscillation_threshold": 0.6,
    }

def default_random_search_params(budget: int, **kwargs) -> Dict[str, Any]:
    """
    Random search is parameterless; all budget is used for initial samples at build time.
    """
    return {}

def default_its_params(budget: int, dim: int | None = None) -> Dict[str, Any]:
    """
    Dimension-agnostic ITS defaults (handled like cd/wcd).
    Defer n_samples computation to builder (needs problem dimension & budget).
    Provide a conservative default for hypotheses (=1); builder may derive if absent.
    """
    return {
        "its_n_hypotheses": 1,  # keep simple; builder derives n_samples
        "its_mc_steps": 1,
        "its_change_of_mind": "score",
        "its_gaussian_filter_channel_wise": False,
        "its_unique_class_condition": False,  # NEW
    }

def default_its2_params(budget: int, dim: int | None = None) -> Dict[str, Any]:  # NEW
    """
    ITS2 defaults mirror ITS: dimension-agnostic; hypotheses default to 1; n_samples derived at build time.
    """
    return {
        "its_n_hypotheses": 1,
        "its_mc_steps": 1,
        "its_change_of_mind": "score",
        "its_gaussian_filter_channel_wise": False,
        "its_unique_class_condition": False,  # NEW
    }

def default_cmaes_params(budget: int, min_pop: int = 4) -> Dict[str, Any]:
    """
    Simple, budget-respecting defaults for CMA-ES:
    - choose iterations by a small heuristic (sqrt(budget)), then set population so cost <= budget
    - mu default is floor(pop/2)
    """
    iters = max(5, int(budget ** 0.5))
    if iters + 1 >= budget:
        iters = max(1, budget // 2)

    pop = max(min_pop, budget // max(1, (iters + 1)))
    # Ensure cost fits; if not, reduce iterations until it does
    while _cost_es(pop, iters) > budget and iters > 1:
        iters -= 1
        pop = max(min_pop, budget // max(1, (iters + 1)))

    mu_default = max(1, pop // 2)
    return {
        "cmaes_population": int(pop),
        "cmaes_iters": int(iters),
        "cmaes_sigma0": 0.5,
        "cmaes_mu_par": int(mu_default),
    }

# NEW: defaults for CMA-ES using Nevergrad backend (same shape as cmaes defaults)
def default_cmaes_nevergrad_params(budget: int, min_pop: int = 4) -> Dict[str, Any]:
    """
    Defaults for CMA-ES when using Nevergrad optimizer:
    - follow same budget-respecting heuristic as default_cmaes_params
    - expose same keys so downstream filtering / saving is consistent
    """
    iters = max(5, int(budget ** 0.5))
    if iters + 1 >= budget:
        iters = max(1, budget // 2)

    pop = max(min_pop, budget // max(1, (iters + 1)))
    while _cost_es(pop, iters) > budget and iters > 1:
        iters -= 1
        pop = max(min_pop, budget // max(1, (iters + 1)))

    mu_default = max(1, pop // 2)
    return {
        "cmaes_population": int(pop),
        "cmaes_iters": int(iters),
        "cmaes_sigma0": 0.5,
        "cmaes_mu_par": int(mu_default),
    }

# Central registry for default parameter factory functions
ALGO_DEFAULT_PARAM_FACTORIES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "shgo": default_shgo_params,
    "parallel_sa": default_parallel_sa_params,
    "parallel_sa_resets": default_parallel_sa_resets_params,  # NEW
    "evolutionary": default_evolutionary_params,
    "pso": default_pso_params,
    "cd": default_cd_params,
    "wcd": default_wcd_params,
    "wcd_lattice": default_wcd_lattice_params,  # NEW
    "cd_multi_cyclus": default_cd_multi_cyclus_params,  # NEW
    "pgd": default_pgd_params,
    "pgd_restart": default_pgd_restart_params,
    "pgd_window": default_pgd_window_params,
    "random_search": default_random_search_params,  # NEW
    "its": default_its_params,
    "its2": default_its2_params,  # NEW
    "cmaes": default_cmaes_params,  # NEW
    "cmaes_nevergrad": default_cmaes_nevergrad_params,  # NEW
}

# Prefix patterns to identify parameters per algorithm (used for filtering trial params)
PARAM_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "shgo": ("shgo_", "grad_weight"),
    "parallel_sa": ("psa_",),
    "parallel_sa_resets": ("psa_", "psar_"),  # Include both psa_ and psar_ prefixes
    "evolutionary": ("es_",),
    "pso": ("pso_",),
    "cd": ("cd_",),
    "wcd": ("wcd_",),
    "wcd_lattice": ("wcd_",),  # NEW: uses same params as wcd
    "cd_multi_cyclus": ("wcd_",),  # NEW: uses same params as wcd
    "pgd": ("pgd_", "grad_weight"),
    "pgd_restart": ("pgd_", "pgdr_", "grad_weight"),
    "pgd_window": ("pgd_", "pgdw_", "grad_weight"),
    "random_search": ("shgo_",),  # Uses SHGO underneath
    "its": ("its_",),
    "its2": ("its_",),  # NEW: its2 uses same param names as its
    "cmaes": ("cmaes_",),  # NEW
    "cmaes_nevergrad": ("cmaes_",),  # NEW: Nevergrad CMA-ES uses same keys as cmaes
}

def get_default_params(algo: str, budget: int, **kwargs) -> Dict[str, Any]:
    """
    Obtain default parameter dictionary for an algorithm given a budget.
    kwargs are forwarded to the underlying default_* function.
    """
    if algo not in ALGO_DEFAULT_PARAM_FACTORIES:
        raise KeyError(f"Unknown algorithm '{algo}'")
    return ALGO_DEFAULT_PARAM_FACTORIES[algo](budget, **kwargs)

def filter_algo_params(params: Dict[str, Any], algo: str) -> Dict[str, Any]:
    """
    Filter a raw parameter dict (e.g., trial.params) to only those relevant for the algo.
    """
    if algo not in PARAM_PREFIXES:
        raise KeyError(f"Unknown algorithm '{algo}'")
    prefixes = PARAM_PREFIXES[algo]
    out: Dict[str, Any] = {}
    for k, v in params.items():
        for pref in prefixes:
            # exact match (e.g., 'grad_weight') or prefix match 'prefix_'
            if (pref.endswith("_") and k.startswith(pref)) or k == pref:
                out[k] = v
                break
    return out

def save_params(params: Dict[str, Any], path: str | Path):
    """
    Save parameters to YAML (overwrites).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(params, f)

def load_params(path: str | Path) -> Dict[str, Any]:
    """
    Load parameters from YAML. Returns empty dict if file missing.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Parameter file {path} must contain a mapping.")
    return data

def merge_params(base: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Shallow merge helper: overrides win.
    """
    if not overrides:
        return dict(base)
    merged = dict(base)
    merged.update(overrides)
    return merged

# (Optional) convenience wrapper akin to a 'default_allocation'
def default_allocation(algo: str, budget: int, **kwargs) -> Dict[str, Any]:
    """
    Backward-compatible allocation provider.
    """
    return get_default_params(algo, budget, **kwargs)