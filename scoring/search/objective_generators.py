import gc
import math
import torch
import optuna
from typing import Callable, Optional, Sequence, Dict, Any

from search.shgo import SHGO
from search.simulated_anealing import ParallelSimulatedAnnealing
from search.evo import EvolutionarySearch, CMAES, CMAES_Nevergrad  # added CMAES_Nevergrad import
from search.swarm import PSO
from search.tree import CoordinateDescent, WeightedCoordinateDescent
from search.parallel_gradient import (
    ParallelGradientDescent,
    ParallelRestartingDescent,
    WindowStuckDetectionDescent,
)
from utils.eval.ood_performance import evaluate_confidence_and_search, ITSWRAPPER, ConfidenceEvaluator  # NEW
from its.search import InverseTransformationSearch,InverseTransformationSearch2
# NEW
from utils.transform_sequence import TransformSequence
from utils.transformation_problem import TransformationProblem


#Note removed sgd as optimizer as it cant handle different scales of the different detectors.

def _problems_same_dim(problems: Sequence) -> bool:
    try:
        dims = [int(p.calc_complete_size()) for p in problems]
        return all(d == dims[0] for d in dims)
    except Exception:
        return False


def _assert_compatible_problems_for_shared_build(algo: str, problems: Sequence):
    # Only enforce for algorithms that bake-in dimension/transform choices
    if algo not in ("cd", "wcd", "its"):
        return
    # All problems must expose the same parameter dimensionality
    if not _problems_same_dim(problems):
        raise ValueError(f"All problems must have the same calc_complete_size() for shared build of '{algo}'.")

# --------------------------------------------------
# Helper to copy TransformationProblem with different init_method
# --------------------------------------------------
def _copy_problem_with_init_method(problem, init_method: str):
    """
    Create a copy of a TransformationProblem with a different init_method.
    If the problem's transform_sequence already has the requested init_method,
    the original problem is returned.
    """

    original_sequence = problem.transform_sequence

    # If the sequence already has the requested init_method, return the original problem
    if original_sequence.init_method == init_method:
        return problem

    # Copy the sequence with new init_method
    new_sequence = TransformSequence(
        transformations=original_sequence.transformations,
        domains=original_sequence.domains,
        neighbour_hood_size=original_sequence.neighbour_hood_size,
        application_method=original_sequence.application_method,
        device=original_sequence.dummy_param.device,
        dtype=original_sequence.dummy_param.dtype,
        init_method=init_method,
        reflect=original_sequence.reflect,
        invert=original_sequence.invert
    )

    # Create a new TransformationProblem with the new sequence
    new_problem = TransformationProblem(
        confidence_module=problem.confidence_module,
        transform_sequence=new_sequence,
        consolidate_method=problem.consolidate_method,
        max_batch_size=problem.max_batch_size
    )

    return new_problem


class WCD_LATTICE_WRAPPER(torch.nn.Module):
    """
    Wrapper for WCD with lattice initialization that automatically creates
    a lattice-initialized problem during optimize() call.
    Stateless version (no caching).
    """

    def __init__(self, wcd_optimizer):
        super().__init__()
        self.wcd_optimizer = wcd_optimizer

    def optimize(self, problem, x, y=None):
        """
        Optimize using WCD with a lattice-initialized version of the problem.
        """
        lattice_problem = _copy_problem_with_init_method(problem, "permuted_lattice")
        return self.wcd_optimizer.optimize(lattice_problem, x, y)


# --------------------------------------------------
# Cost model helpers
# --------------------------------------------------
def _ensure_min(v: int, lo: int = 1) -> int:
    return max(lo, int(v))
def _safe_mean_accuracy(correct: int, total: int) -> float:
    return 0.0 if total == 0 else correct / total

# --- new helpers for budget assertions ---
def _assert_budget(cost: int, budget: int, label: str):
    if cost > budget:
        raise ValueError(f"{label} budget exceeded: cost={cost} > budget={budget}")

def _cost_shgo(n_init: int, local_runs: int, local_max_steps: int, grad_weight: int) -> int:
    if local_max_steps==0:
        return n_init
    return n_init + local_runs * local_max_steps * grad_weight +local_runs

def _cost_parallel_sa(par_runs: int, max_iter: int) -> int:
    return par_runs * (max_iter+1)
def _cost_es(pop: int, iters: int) -> int:
    return pop * (iters + 1)
def _cost_pso(swarm: int, steps: int) -> int:
    return swarm * (steps + 1)
def _cost_cd(dim: int, samples: int) -> int:
    return dim * samples
def _cost_wcd(dim: int, base: int, first_factor: int, rounds: int) -> int:
    per_round = base * (dim - 1 + first_factor)
    return per_round * rounds

# NEW cost helper for parallel gradient methods
def _cost_pgd(par_runs: int, max_iter: int, grad_weight: int) -> int:
    return par_runs * (max_iter * grad_weight + 1)

# --- ITS cost helper (already defined) ---
def _cost_its(n_samples: int, n_hypotheses: int, dim: int) -> int:
    return n_samples * (1 + n_hypotheses * (dim - 1))

# --------------------------------------------------
# Common evaluation loop (now unified; uses repeats)
# --------------------------------------------------
def _evaluate_model_partial(
    search_obj,
    problem,
    model,
    dataloader,
    device: str = "cuda",
    repeats: int = 1,
    check_percent: float = 0.1,
):
    model.to(device)
    evaluator = ConfidenceEvaluator(
        model=model,
        optimizer=search_obj,
        problem=problem,
        test_loader=dataloader,
        repeats=repeats,
        show_progress=True,
    )

    # Run until first checkpoint
    acc_at_checkpoint = evaluator.run_until(check_percent)["accuracy_mean"]
    return evaluator, acc_at_checkpoint

# --------------------------------------------------
# Generic parameter samplers (add wcd_lattice)
# --------------------------------------------------

def sample_shgo_params(trial: optuna.Trial, budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    max_init = budget -grad_weight -1
    if max_init < 1:
        max_init=budget
    low_max =int(0.5 * budget)
    n_init = trial.suggest_int("shgo_initial_samples", low_max, max_init)
    max_local_runs = max(1, int((budget - n_init)/(3 * grad_weight +1)))
    local_runs = trial.suggest_int("shgo_local_runs", 1, max_local_runs)
    per_run_budget = max(0, ((budget - n_init)- local_runs)// (local_runs * grad_weight))
    local_steps = per_run_budget
    _assert_budget(_cost_shgo(n_init, local_runs, local_steps, grad_weight), budget, "SHGO")
    sel_method = trial.suggest_categorical("shgo_selection_method", ["topk", "knn"])
    opt_choice = trial.suggest_categorical("shgo_local_opt", ["adam", ])
    lr = trial.suggest_float("shgo_lr", 1e-3, 1e-0, log=True)
    params = {
        "shgo_initial_samples": n_init,
        "shgo_local_runs": local_runs,
        "shgo_local_steps": per_run_budget,  # canonical only
        "shgo_selection_method": sel_method,
        "shgo_local_opt": opt_choice,
        "shgo_lr": lr,
        "shgo_acceptance_criterion": trial.suggest_categorical("shgo_acceptance_criterion", ['always', 'step', 'final']),
        "grad_weight": grad_weight,
    }
    if opt_choice == "sgd":
        params["shgo_momentum"] = trial.suggest_float("shgo_momentum", 0.0, 0.95)

    # NEW: reassign any unassigned budget to initial samples (n_init) so total cost == budget
    current_cost = _cost_shgo(params["shgo_initial_samples"], params["shgo_local_runs"], params["shgo_local_steps"], grad_weight)
    leftover = budget - current_cost
    if leftover > 0:
        params["shgo_initial_samples"] += leftover
        _assert_budget(_cost_shgo(params["shgo_initial_samples"], params["shgo_local_runs"], params["shgo_local_steps"], grad_weight), budget, "SHGO(top-up)")

    return params

def sample_parallel_sa_params(trial: optuna.Trial, budget: int) -> Dict[str, Any]:
    max_parallel = max(1, budget // 10) #means at least 10 iters
    parallel_runs = trial.suggest_int("psa_parallel_runs", 1, max_parallel)
    max_iter = max(1, budget // parallel_runs -1) #at least parallel_runs iterations
    _assert_budget(_cost_parallel_sa(parallel_runs, max_iter), budget, "ParallelSA")
    return {
        "psa_parallel_runs": parallel_runs,
        "psa_max_iterations": max_iter,
        "psa_init_temp": trial.suggest_float("psa_init_temp", 1.0, 200.0, log=True),
        "psa_cooling": trial.suggest_float("psa_cooling", 0.80, 0.999),
        "psa_reinit_interval": 9999999999999,  # effectively disabled
        "psa_reinit_amount": 0.0,  # effectively disabled
        "psa_neighbor_hood_size": trial.suggest_float("psa_neighbor_hood_size", 0.01, 1.0, log=True),
    }

def sample_parallel_sa_resets_params(trial: optuna.Trial, budget: int) -> Dict[str, Any]:
    # Sample base PSA parameters first
    base_params = sample_parallel_sa_params(trial, budget)

    # Override with more aggressive reinit settings for resets variant
    max_reinit_interval = base_params["psa_max_iterations"] // 4

    base_params["psa_reinit_interval"] = trial.suggest_int("psar_reinit_interval", 3, max(3, max_reinit_interval))


    base_params["psa_reinit_amount"] = trial.suggest_float("psar_reinit_amount", 0.05, 0.5)

    return base_params

def sample_evolutionary_params(trial: optuna.Trial, budget: int, min_pop: int = 4) -> Dict[str, Any]:
    max_iters_possible = max(1, budget // max(1, min_pop) - 1)
    iters = trial.suggest_int("es_iters", 1, max_iters_possible)
    pop_max = max(min_pop, budget // (iters + 1))
    pop = trial.suggest_int("es_population", pop_max, pop_max)
    _assert_budget(_cost_es(pop, iters), budget, "EvolutionarySearch")
    return {
        "es_population": pop,
        "es_iters": iters,
        "es_selection_frac": trial.suggest_float("es_selection_frac", 0.05, 0.9),
        "es_sigma_decay": trial.suggest_float("es_sigma_decay", 0.90, 0.999),
    }


def sample_pso_params(trial: optuna.Trial, budget: int, min_swarm: int = 4) -> Dict[str, Any]:
    # Compute maximum steps possible given the budget and minimum swarm
    min_steps = 1  # allow very small budgets
    max_steps_possible = max(min_steps, budget // max(1, min_swarm) - 1)

    # Sample number of steps
    steps = trial.suggest_int("pso_steps", min_steps, max_steps_possible)

    # Compute maximum swarm that fits the budget
    swarm_max = max(min_swarm, budget // (steps + 1))

    # Always use the maximum swarm
    swarm = trial.suggest_int("pso_swarm_size", swarm_max, swarm_max)

    # Ensure budget is not exceeded
    _assert_budget(_cost_pso(swarm, steps), budget, "PSO")

    return {
        "pso_swarm_size": swarm,
        "pso_steps": steps,
        "pso_w": trial.suggest_float("pso_w", 0.2, 0.9),
        "pso_c1": trial.suggest_float("pso_c1", 0.5, 3.0),
        "pso_c2": trial.suggest_float("pso_c2", 0.5, 3.0),
        "pso_clamp_velocity": trial.suggest_categorical("pso_clamp_velocity", [True, False]),
        "pso_vmax_scale": trial.suggest_float("pso_vmax_scale", 0.05, 0.5),
    }

def sample_cd_params(trial: optuna.Trial, budget: int, device: str, val_loader, problem, samples_min: int = 1) -> Dict[str, Any]:
    # Dimension-dependent values deferred to finalize_params; return empty (dimension-agnostic)
    return {}

def sample_wcd_params(trial: optuna.Trial, budget: int, device: str, val_loader, problem, rounds_range=(1, 5)) -> Dict[str, Any]:
    # Only store rounds + first-dim weight (dimension/base deferred)
    min_rounds, max_rounds_hint = rounds_range
    # Quick optimistic upper bound (true feasibility rechecked at finalize)
    rounds = trial.suggest_int("wcd_rounds", min_rounds, max_rounds_hint)
    first_factor = trial.suggest_int("wcd_first_dim_factor", 1, 4)
    return {
        "wcd_rounds": rounds,
        "wcd_first_dim_factor": first_factor,
    }

def sample_wcd_lattice_params(trial: optuna.Trial, budget: int, device: str, val_loader, problem, rounds_range=(1, 5)) -> Dict[str, Any]:
    """
    Sample parameters for WCD with lattice initialization.
    Uses same sampling logic as regular WCD.
    """
    return sample_wcd_params(trial, budget, device, val_loader, problem, rounds_range)

def sample_cd_multi_cyclus_params(trial: optuna.Trial, budget: int, device: str, val_loader, problem, rounds_range=(1, 5)) -> Dict[str, Any]:
    """
    Sample parameters for multi-round coordinate descent (CD with cycles).
    This is a WCD variant with first_dim_factor fixed to 1 (no special weighting).
    Only rounds needs to be sampled; first_dim_factor is always 1.
    """
    min_rounds, max_rounds_hint = rounds_range
    rounds = trial.suggest_int("wcd_rounds", min_rounds, max_rounds_hint)
    return {
        "wcd_rounds": rounds,
        "wcd_first_dim_factor": 1,  # Fixed to 1 for uniform coordinate descent
    }

# NEW samplers for parallel gradient descent family ---

def sample_pgd_params(trial: optuna.Trial, budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    """
    Sample projected gradient descent (PGD) parameters from an Optuna trial,
    ensuring that the total compute cost does not exceed the given budget.

    The cost model is:
        cost = parallel_runs * (max_iterations * grad_weight + 1)
    """
    # --- Determine how many parallel runs are affordable -----------------------
    # Require at least ~2 iterations per run (so each run has meaningful work).
    max_parallel_runs = max(1, budget // (2 * grad_weight + 1))
    parallel_runs = trial.suggest_int("pgd_parallel_runs", 1, max_parallel_runs)

    # --- Compute max iterations that fit within the remaining budget ----------
    # Derived directly from the cost equation:
    #     budget = par_runs * (max_iter * grad_weight + 1)
    #  →  max_iter = (budget / par_runs - 1) / grad_weight
    max_iterations = int((budget - parallel_runs) / (parallel_runs * grad_weight))
    max_iterations = max(1, max_iterations)

    # --- Verify the total cost stays within budget ----------------------------
    _assert_budget(
        _cost_pgd(parallel_runs, max_iterations, grad_weight),
        budget,
        "PGD"
    )

    # --- Sample optimization hyperparameters ----------------------------------
    learning_rate = trial.suggest_float("pgd_learning_rate", 1e-3, 1e-0, log=True)
    decay_rate = trial.suggest_float("pgd_lr_decay_rate", 0.90, 0.999)
    optimizer = trial.suggest_categorical("pgd_optimizer", ["adam", ])

    # --- Collect parameters ---------------------------------------------------
    params = {
        "pgd_parallel_runs": parallel_runs,
        "pgd_max_iterations": max_iterations,
        "pgd_learning_rate": learning_rate,
        "pgd_lr_decay_rate": decay_rate,
        "pgd_optimizer": optimizer,
        "grad_weight": grad_weight,
    }

    # SGD uses momentum; Adam doesn’t.
    if optimizer == "sgd":
        params["pgd_momentum"] = trial.suggest_float("pgd_momentum", 0.0, 0.95)

    return params


def sample_pgd_restart_params(trial: optuna.Trial, budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    base = sample_pgd_params(trial, budget, grad_weight=grad_weight)
    interval = trial.suggest_int("pgdr_reinit_interval", 3, max(5, base["pgd_max_iterations"] // 4))
    amount = trial.suggest_float("pgdr_reinit_amount", 0.05, 0.5)
    base.update({
        "pgdr_reinit_interval": interval,
        "pgdr_reinit_amount": amount,
    })
    return base

def sample_pgd_window_params(trial: optuna.Trial, budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    base = sample_pgd_params(trial, budget, grad_weight=grad_weight)
    window = trial.suggest_int("pgdw_stuck_window_size", 3, 12)
    imp_thr = trial.suggest_float("pgdw_improvement_threshold", 1e-5, 1e-3, log=True)
    flat_thr = trial.suggest_float("pgdw_flat_threshold", 1e-5, 1e-3, log=True)
    osc_thr = trial.suggest_float("pgdw_oscillation_threshold", 0.3, 0.9)
    base.update({
        "pgdw_stuck_window_size": window,
        "pgdw_improvement_threshold": imp_thr,
        "pgdw_flat_threshold": flat_thr,
        "pgdw_oscillation_threshold": osc_thr,
    })
    return base

def sample_random_search_params(trial: optuna.Trial, budget: int, **kwargs) -> Dict[str, Any]:
    # Parameterless, budget is used entirely for initial samples at build time.
    return {}

def sample_its_params(trial: optuna.Trial, budget: int, device: str, val_loader, problem, **_) -> Dict[str, Any]:
    # Do NOT use problem.calc_complete_size() or budget here.
    # Only sample the number of hypotheses (1..4). MC steps fixed to 1.
    # Other boolean options can still be sampled.
    n_hyp = trial.suggest_int("its_n_hypotheses", 1, 4)
    return {
        "its_n_hypotheses": int(n_hyp),
        "its_n_samples": None,  # explicit marker: actual n_samples is derived at build time
        "its_mc_steps": 1,  # fixed as requested
        "its_change_of_mind": trial.suggest_categorical("its_change_of_mind", ["score", "off"]),
        "its_gaussian_filter_channel_wise": trial.suggest_categorical("its_gaussian_filter_channel_wise", [False, True]),
        "its_unique_class_condition": trial.suggest_categorical("its_unique_class_condition", [False, True]),  # NEW
    }

def sample_cmaes_params(trial: optuna.Trial, budget: int, min_pop: int = 4) -> Dict[str, Any]:
    """
    Iterations-first CMA-ES sampler.

    - Sample iterations (iters) only.
    - Derive population = floor(budget / (iters + 1)) so cost = pop*(iters+1) <= budget.
    - Choose the sampling range for iters such that the derived population can be >= min_pop
      when possible. Allow iters == 0 to enable small budgets to allocate a reasonable pop.
    - mu_par is sampled in [1, pop].
    """
    # Determine the largest iters that still allows at least min_pop population:
    # pop = floor(budget / (iters + 1)) >= min_pop  => iters + 1 <= floor(budget / min_pop)
    # => iters <= floor(budget / min_pop) - 1
    max_iters_possible = max(0, (budget // max(1, min_pop)) - 1)

    # Sample iterations (allow zero to accommodate small budgets)
    iters = trial.suggest_int("cmaes_iters", 0, max_iters_possible)

    # Derive population deterministically to actually use the budget given iters
    pop = max(1, budget // (iters + 1))

    # If the derived population is below min_pop (possible when budget < min_pop),
    # we fall back to the maximum feasible population while keeping min_pop intention:
    if pop < min_pop:
        # there is no iters >=0 satisfying pop >= min_pop; keep pop as the budget-max and
        # reduce min_pop expectation implicitly (we don't raise here to keep sampling robust)
        pop = max(1, budget // (iters + 1))

    # mu (number of parents) must be in [1, pop]
    mu = trial.suggest_int("cmaes_mu_par", 1, max(1, pop))
    mu = min(mu, pop)

    sigma0 = trial.suggest_float("cmaes_sigma0", 0.1, 2.0)

    # Safety: ensure we didn't exceed budget
    final_cost = _cost_es(pop, iters)
    _assert_budget(final_cost, budget, "CMAES")

    return {
        "cmaes_population": int(pop),
        "cmaes_iters": int(iters),
        "cmaes_sigma0": float(sigma0),
        "cmaes_mu_par": int(mu),
    }

# NEW: sampler for CMA-ES that will be run via Nevergrad backend
def sample_cmaes_nevergrad_params(trial: optuna.Trial, budget: int, min_pop: int = 4) -> Dict[str, Any]:
    """
    Sample iterations-first then derive population so cost = pop*(iters+1) <= budget.
    Same parameter keys as sample_cmaes_params to keep persistence/filtering consistent.
    """
    max_iters_possible = max(0, (budget // max(1, min_pop)) - 1)
    iters = trial.suggest_int("cmaes_iters", 0, max_iters_possible)
    pop = max(1, budget // (iters + 1))
    if pop < min_pop:
        pop = max(1, budget // (iters + 1))
    mu = trial.suggest_int("cmaes_mu_par", 1, max(1, pop))
    mu = min(mu, pop)
    sigma0 = trial.suggest_float("cmaes_sigma0", 0.1, 2.0)
    final_cost = _cost_es(pop, iters)
    _assert_budget(final_cost, budget, "CMAES_Nevergrad")
    return {
        "cmaes_population": int(pop),
        "cmaes_iters": int(iters),
        "cmaes_sigma0": float(sigma0),
        "cmaes_mu_par": int(mu),
    }

# Mapping for sampler dispatch
_PARAM_SAMPLERS = {
    "shgo": sample_shgo_params,
    "parallel_sa": sample_parallel_sa_params,
    "parallel_sa_resets": sample_parallel_sa_resets_params,  # NEW
    "evolutionary": sample_evolutionary_params,
    "pso": sample_pso_params,
    "cd": sample_cd_params,
    "wcd": sample_wcd_params,
    "wcd_lattice": sample_wcd_lattice_params,  # NEW
    "cd_multi_cyclus": sample_cd_multi_cyclus_params,  # NEW
    "pgd": sample_pgd_params,
    "pgd_restart": sample_pgd_restart_params,
    "pgd_window": sample_pgd_window_params,
    "random_search": sample_random_search_params,  # NEW
    "its": sample_its_params,
    "its2": sample_its_params,  # NEW: use same (dimension-agnostic) sampler as ITS
    "cmaes": sample_cmaes_params,
    "cmaes_nevergrad": sample_cmaes_nevergrad_params,  # NEW
}




# --------------------------------------------------
# Builder: create algorithm instance (add wcd_lattice)
# --------------------------------------------------
def build_search_algorithm(algo: str, params: Dict[str, Any], problem=None, budget: Optional[int] = None, model=None):
    # NOTE: For cd / wcd when dimension-dependent fields are absent, caller must have finalized params.
    if algo == "shgo":
        opt_name = params.get("shgo_local_opt", "adam")
        lr = params.get("shgo_lr", 1e-3)
        if opt_name == "sgd":
            momentum = params.get("shgo_momentum", 0.0)
            opt_cls = torch.optim.SGD
            opt_kwargs = {"lr": lr, "momentum": momentum}
        else:
            opt_cls = torch.optim.Adam
            opt_kwargs = {"lr": lr}
        return SHGO(
            initial_samples=params["shgo_initial_samples"],
            local_runs=params["shgo_local_runs"],
            local_max_steps=params["shgo_local_steps"],
            local_opt_class=opt_cls,
            local_opt_kwargs=opt_kwargs,
            selection_method=params.get("shgo_selection_method", "topk"),
            acceptance_criterion=params.get("shgo_acceptance_criterion", "step"),
        )
    if algo == "random_search":
        if budget is None:
            raise ValueError("Random Search build requires budget.")
        return SHGO(
            initial_samples=budget,
            local_runs=1,
            local_max_steps=0,
            local_opt_class=torch.optim.Adam, # Not used, but required
            local_opt_kwargs={},
            selection_method="topk", # Not used
            acceptance_criterion="step", # Not used
        )
    if algo == "parallel_sa":
        return ParallelSimulatedAnnealing(
            initial_temp=params["psa_init_temp"],
            cooling_rate=params["psa_cooling"],
            max_iterations=params["psa_max_iterations"],
            parallel_runs=params["psa_parallel_runs"],
            reinit_interval=params.get("psa_reinit_interval", 9999999999999),
            reinit_amount=params.get("psa_reinit_amount", 0.0),
            neighbor_hood_size=params.get("psa_neighbor_hood_size", 0.1),
        )
    if algo == "parallel_sa_resets":
        # Use normal PSA with reinit parameters enabled
        return ParallelSimulatedAnnealing(
            initial_temp=params["psa_init_temp"],
            cooling_rate=params["psa_cooling"],
            max_iterations=params["psa_max_iterations"],
            parallel_runs=params["psa_parallel_runs"],
            reinit_interval=params["psa_reinit_interval"],
            reinit_amount=params["psa_reinit_amount"],
            neighbor_hood_size=params.get("psa_neighbor_hood_size", 0.1),
        )
    if algo == "evolutionary":
        return EvolutionarySearch(
            population_size=params["es_population"],
            num_iters=params["es_iters"],
            selection_frac=params["es_selection_frac"],
            sigma_decay=params["es_sigma_decay"],
        )
    if algo == "cmaes":
        return CMAES(
            population_size=params.get("cmaes_population"),
            mu_par=params.get("cmaes_mu_par"),              # pass mu (parents) through
            sigma0=params.get("cmaes_sigma0", 0.5),
            num_iters=params.get("cmaes_iters", 100),
            project_param=True,
        )
    if algo == "cmaes_nevergrad":
        # Use the Nevergrad-backed CMA-ES wrapper (per-batch)
        return CMAES_Nevergrad(
            population_size=params.get("cmaes_population"),
            sigma0=params.get("cmaes_sigma0", 0.5),
            num_iters=params.get("cmaes_iters", 100),
            seed=params.get("cmaes_seed", None),
            project_param=True,
        )
    if algo == "pso":
        return PSO(
            swarm_size=params["pso_swarm_size"],
            steps=params["pso_steps"],
            w=params["pso_w"],
            c1=params["pso_c1"],
            c2=params["pso_c2"],
            clamp_velocity=params["pso_clamp_velocity"],
            v_max_scale=params["pso_vmax_scale"],
        )
    if algo == "cd":
        number_samples = params.get("cd_number_samples")
        if number_samples is None:
            if problem is None or budget is None:
                raise ValueError("Missing problem/budget to derive cd_number_samples")
            # derive number_samples taking discreteness into account (only here at instance creation)
            dim = problem.calc_complete_size()
            ts = problem.transform_sequence
            discreteness = ts.get_discreteness_vector().to(torch.long).cpu().tolist()
            if len(discreteness) != dim:
                raise ValueError("TransformSequence discreteness length mismatch with problem dimension.")
            # cost for candidate count s is sum(min(s, n_disc) if n_disc>0 else s)
            def total_cost_for(s):
                total = 0
                for n_disc in discreteness:
                    if n_disc is not None and n_disc > 0 and n_disc <= s:
                        total += int(n_disc)
                    else:
                        total += int(s)
                return total
            # start from conservative baseline
            s = max(1, budget // max(1, dim))
            # try to grow s as long as total cost fits budget
            while s + 1 <= budget:
                if total_cost_for(s + 1) <= budget:
                    s += 1
                else:
                    break
            number_samples = int(s)
            # persist derived value
            params["cd_number_samples"] = number_samples
            params["cd_dim_estimate"] = dim
        return CoordinateDescent(number_samples=number_samples)
    if algo == "wcd":
        if "_wcd_samples_per_dim" in params:
            samples_per_dim = params["_wcd_samples_per_dim"]
        else:
            if problem is None or budget is None:
                raise ValueError("Missing problem/budget to derive WCD samples_per_dim")
            dim = problem.calc_complete_size()
            rounds = params["wcd_rounds"]
            first_factor = params["wcd_first_dim_factor"]

            # Work with a per-round budget to ensure total_cost = per_round_cost * rounds <= budget
            per_round_budget = max(1, budget // max(1, rounds))
            denom_per_round = (dim - 1 + first_factor)
            if denom_per_round <= 0:
                denom_per_round = 1
            base = max(1, per_round_budget // denom_per_round)
            tentative = [base * first_factor] + [base] * (dim - 1)

            ts = problem.transform_sequence
            discreteness = ts.get_discreteness_vector().to(torch.long).cpu().tolist()
            if len(discreteness) != dim:
                raise ValueError("TransformSequence discreteness length mismatch with problem dimension.")

            # assign per-round: discrete dims that fit within tentative use their discrete cost
            assigned = []
            for d in range(dim):
                n_disc = discreteness[d]
                t = tentative[d]
                if n_disc is not None and n_disc > 0 and n_disc <= t:
                    assigned.append(int(n_disc))
                else:
                    assigned.append(int(t))

            per_round_assigned = sum(assigned)
            rem_per_round = per_round_budget - per_round_assigned

            # distribute per-round remainder among dims where assigned == tentative (i.e., expandable)
            free_idxs = [i for i in range(dim) if assigned[i] == tentative[i]]
            if rem_per_round > 0 and len(free_idxs) > 0:
                extra_each = rem_per_round // len(free_idxs)
                for i in free_idxs:
                    assigned[i] += extra_each
                rem_per_round -= extra_each * len(free_idxs)
                # distribute remaining one by one (still per-round)
                for i in free_idxs[:rem_per_round]:
                    assigned[i] += 1
                rem_per_round = 0

            # If per-round assigned somehow exceeds per_round_budget, trim from expandable dims
            per_round_assigned = sum(assigned)
            if per_round_assigned > per_round_budget:
                over = per_round_assigned - per_round_budget
                for i in reversed(free_idxs):
                    if over <= 0:
                        break
                    take = min(over, max(0, assigned[i] - 1))
                    assigned[i] -= take
                    over -= take

            samples_per_dim = [int(v) for v in assigned]  # these are per-round base samples
            # persist derived fields (wcd_base_samples is per-round base)
            params["_wcd_samples_per_dim"] = samples_per_dim
            params["wcd_dim_estimate"] = dim
            params["wcd_base_samples"] = base

        return WeightedCoordinateDescent(samples_per_dim=samples_per_dim, rounds=params["wcd_rounds"])
    if algo == "wcd_lattice":
        # Build regular WCD optimizer first
        if "_wcd_samples_per_dim" in params:
            samples_per_dim = params["_wcd_samples_per_dim"]
        else:
            if problem is None or budget is None:
                raise ValueError("Missing problem/budget to derive WCD lattice samples_per_dim")
            dim = problem.calc_complete_size()
            rounds = params["wcd_rounds"]
            first_factor = params["wcd_first_dim_factor"]

            per_round_budget = max(1, budget // max(1, rounds))
            denom_per_round = (dim - 1 + first_factor)
            if denom_per_round <= 0:
                denom_per_round = 1
            base = max(1, per_round_budget // denom_per_round)
            tentative = [base * first_factor] + [base] * (dim - 1)

            ts = problem.transform_sequence
            discreteness = ts.get_discreteness_vector().to(torch.long).cpu().tolist()
            if len(discreteness) != dim:
                raise ValueError("TransformSequence discreteness length mismatch with problem dimension.")

            assigned = []
            for d in range(dim):
                n_disc = discreteness[d]
                t = tentative[d]
                if n_disc is not None and n_disc > 0 and n_disc <= t:
                    assigned.append(int(n_disc))
                else:
                    assigned.append(int(t))

            per_round_assigned = sum(assigned)
            rem_per_round = per_round_budget - per_round_assigned

            free_idxs = [i for i in range(dim) if assigned[i] == tentative[i]]
            if rem_per_round > 0 and len(free_idxs) > 0:
                extra_each = rem_per_round // len(free_idxs)
                for i in free_idxs:
                    assigned[i] += extra_each
                rem_per_round -= extra_each * len(free_idxs)
                for i in free_idxs[:rem_per_round]:
                    assigned[i] += 1
                rem_per_round = 0

            per_round_assigned = sum(assigned)
            if per_round_assigned > per_round_budget:
                over = per_round_assigned - per_round_budget
                for i in reversed(free_idxs):
                    if over <= 0:
                        break
                    take = min(over, max(0, assigned[i] - 1))
                    assigned[i] -= take
                    over -= take

            samples_per_dim = [int(v) for v in assigned]
            params["_wcd_samples_per_dim"] = samples_per_dim
            params["wcd_dim_estimate"] = dim
            params["wcd_base_samples"] = base

        # Create regular WCD and wrap it
        wcd = WeightedCoordinateDescent(samples_per_dim=samples_per_dim, rounds=params["wcd_rounds"])
        return WCD_LATTICE_WRAPPER(wcd)
    
    if algo == "cd_multi_cyclus":
        # Multi-round CD: same as WCD but first_dim_factor is always 1
        if "_wcd_samples_per_dim" in params:
            samples_per_dim = params["_wcd_samples_per_dim"]
        else:
            if problem is None or budget is None:
                raise ValueError("Missing problem/budget to derive cd_multi_cyclus samples_per_dim")
            dim = problem.calc_complete_size()
            rounds = params["wcd_rounds"]
            first_factor = params["wcd_first_dim_factor"]  # Should be 1

            per_round_budget = max(1, budget // max(1, rounds))
            # All dimensions get same base allocation
            denom_per_round = dim
            if denom_per_round <= 0:
                denom_per_round = 1
            base = max(1, per_round_budget // denom_per_round)
            tentative = [base] * dim

            ts = problem.transform_sequence
            discreteness = ts.get_discreteness_vector().to(torch.long).cpu().tolist()
            if len(discreteness) != dim:
                raise ValueError("TransformSequence discreteness length mismatch with problem dimension.")

            assigned = []
            for d in range(dim):
                n_disc = discreteness[d]
                t = tentative[d]
                if n_disc is not None and n_disc > 0 and n_disc <= t:
                    assigned.append(int(n_disc))
                else:
                    assigned.append(int(t))

            per_round_assigned = sum(assigned)
            rem_per_round = per_round_budget - per_round_assigned

            free_idxs = [i for i in range(dim) if assigned[i] == tentative[i]]
            if rem_per_round > 0 and len(free_idxs) > 0:
                extra_each = rem_per_round // len(free_idxs)
                for i in free_idxs:
                    assigned[i] += extra_each
                rem_per_round -= extra_each * len(free_idxs)
                for i in free_idxs[:rem_per_round]:
                    assigned[i] += 1
                rem_per_round = 0

            per_round_assigned = sum(assigned)
            if per_round_assigned > per_round_budget:
                over = per_round_assigned - per_round_budget
                for i in reversed(free_idxs):
                    if over <= 0:
                        break
                    take = min(over, max(0, assigned[i] - 1))
                    assigned[i] -= take
                    over -= take

            samples_per_dim = [int(v) for v in assigned]
            params["_wcd_samples_per_dim"] = samples_per_dim
            params["wcd_dim_estimate"] = dim
            params["wcd_base_samples"] = base

        return WeightedCoordinateDescent(samples_per_dim=samples_per_dim, rounds=params["wcd_rounds"])
    
    if algo == "pgd":
        opt_cls = torch.optim.Adam if params.get("pgd_optimizer", "adam") == "adam" else torch.optim.SGD
        opt_kwargs = {}
        if opt_cls is torch.optim.SGD and "pgd_momentum" in params:
            opt_kwargs["momentum"] = params["pgd_momentum"]
        return ParallelGradientDescent(
            optimizer_class=opt_cls,
            optimizer_params=opt_kwargs,
            learning_rate=params["pgd_learning_rate"],
            max_iterations=params["pgd_max_iterations"],
            parallel_runs=params["pgd_parallel_runs"],
            lr_decay_rate=params["pgd_lr_decay_rate"],
        )
    if algo == "pgd_restart":
        opt_cls = torch.optim.Adam if params.get("pgd_optimizer", "adam") == "adam" else torch.optim.SGD
        opt_kwargs = {}
        if opt_cls is torch.optim.SGD and "pgd_momentum" in params:
            opt_kwargs["momentum"] = params["pgd_momentum"]
        return ParallelRestartingDescent(
            optimizer_class=opt_cls,
            optimizer_params=opt_kwargs,
            learning_rate=params["pgd_learning_rate"],
            max_iterations=params["pgd_max_iterations"],
            parallel_runs=params["pgd_parallel_runs"],
            lr_decay_rate=params["pgd_lr_decay_rate"],
            reinit_interval=params["pgdr_reinit_interval"],
            reinit_amount=params["pgdr_reinit_amount"],
        )
    if algo == "pgd_window":
        opt_cls = torch.optim.Adam if params.get("pgd_optimizer", "adam") == "adam" else torch.optim.SGD
        opt_kwargs = {}
        if opt_cls is torch.optim.SGD and "pgd_momentum" in params:
            opt_kwargs["momentum"] = params["pgd_momentum"]
        return WindowStuckDetectionDescent(
            optimizer_class=opt_cls,
            optimizer_params=opt_kwargs,
            learning_rate=params["pgd_learning_rate"],
            max_iterations=params["pgd_max_iterations"],
            parallel_runs=params["pgd_parallel_runs"],
            lr_decay_rate=params["pgd_lr_decay_rate"],
            stuck_window_size=params["pgdw_stuck_window_size"],
            improvement_threshold=params["pgdw_improvement_threshold"],
            flat_threshold=params["pgdw_flat_threshold"],
            oscillation_threshold=params["pgdw_oscillation_threshold"],
        )
    if algo == "its":
        if model is None:
            raise ValueError("ITS requires model.")
        if problem is None:
            raise ValueError("ITS requires problem.")
        if budget is None:
            raise ValueError("ITS build requires budget to derive missing parameters.")
        dim = problem.calc_complete_size()

        # Ensure integer & cap hypotheses to 4 (sampling phase only picks 1..4, but clamp here too)
        n_hyp = params.get("its_n_hypotheses")
        if n_hyp is None:
            # choose the largest feasible number of hypotheses in [1..4]
            max_h_allowed = 1
            safe_s = 1
            for h in range(4, 0, -1):
                denom = 1 + h * max(0, (dim - 1))
                s = max(1, budget // max(1, denom))
                # require samples >= hypotheses and that the computed cost fits the budget
                if s >= h and _cost_its(s, h, dim) <= budget:
                    max_h_allowed, safe_s = h, s
                    break
            n_hyp = int(max(1, min(4, max_h_allowed)))
            params["its_n_hypotheses"] = n_hyp
        else:
            # clamp user-provided hypotheses to [1..4]
            n_hyp = int(max(1, min(4, n_hyp)))
            # try reducing until feasible
            while n_hyp > 1:
                denom = 1 + n_hyp * max(0, (dim - 1))
                s = max(1, budget // max(1, denom))
                if s >= n_hyp and _cost_its(s, n_hyp, dim) <= budget:
                    break
                n_hyp -= 1
            params["its_n_hypotheses"] = n_hyp

        # Derive n_samples here (deferred from sampling). Ensure budget compliance.
        n_samples = params.get("its_n_samples")
        if not n_samples:
            denom = 1 + n_hyp * max(0, (dim - 1))
            n_samples = max(1, budget // max(1, denom))
            params["its_n_samples"] = int(n_samples)

        # If provided / derived combination still exceeds budget, attempt to reduce hypotheses to fit.
        if _cost_its(params["its_n_samples"], params["its_n_hypotheses"], dim) > budget:
            # try reducing hypotheses until cost fits (prefer reducing hypotheses rather than samples)
            while params["its_n_hypotheses"] > 1:
                params["its_n_hypotheses"] -= 1
                denom = 1 + params["its_n_hypotheses"] * max(0, (dim - 1))
                params["its_n_samples"] = max(1, budget // max(1, denom))
                if _cost_its(params["its_n_samples"], params["its_n_hypotheses"], dim) <= budget:
                    break
            if _cost_its(params["its_n_samples"], params["its_n_hypotheses"], dim) > budget:
                raise ValueError("Derived / provided ITS parameters exceed budget.")

        its = InverseTransformationSearch(
            model=model,
            transformation=None,
            domain=None,
            n_samples=params["its_n_samples"],
            n_hypotheses=params["its_n_hypotheses"],
            mc_steps=params.get("its_mc_steps", 1),
            change_of_mind=params.get("its_change_of_mind", "score"),
            gaussian_filter_channel_wise=params.get("its_gaussian_filter_channel_wise", False),
            confidence_module=getattr(problem, "confidence_module", None),
            en_unique_class_condition=params.get("its_unique_class_condition", False),  # NEW
        )
        return ITSWRAPPER(its)

    if algo == "its2":
        if model is None:
            raise ValueError("ITS requires model.")
        if problem is None:
            raise ValueError("ITS requires problem.")
        if budget is None:
            raise ValueError("ITS build requires budget to derive missing parameters.")
        dim = problem.calc_complete_size()

        # Ensure integer & cap hypotheses to 4 (sampling phase only picks 1..4, but clamp here too)
        n_hyp = params.get("its_n_hypotheses")
        if n_hyp is None:
            # choose the largest feasible number of hypotheses in [1..4]
            max_h_allowed = 1
            safe_s = 1
            for h in range(4, 0, -1):
                denom = 1 + h * max(0, (dim - 1))
                s = max(1, budget // max(1, denom))
                # require samples >= hypotheses and that the computed cost fits the budget
                if s >= h and _cost_its(s, h, dim) <= budget:
                    max_h_allowed, safe_s = h, s
                    break
            n_hyp = int(max(1, min(4, max_h_allowed)))
            params["its_n_hypotheses"] = n_hyp
        else:
            # clamp user-provided hypotheses to [1..4]
            n_hyp = int(max(1, min(4, n_hyp)))
            # try reducing until feasible
            while n_hyp > 1:
                denom = 1 + n_hyp * max(0, (dim - 1))
                s = max(1, budget // max(1, denom))
                if s >= n_hyp and _cost_its(s, n_hyp, dim) <= budget:
                    break
                n_hyp -= 1
            params["its_n_hypotheses"] = n_hyp

        # Derive n_samples here (deferred from sampling). Ensure budget compliance.
        n_samples = params.get("its_n_samples")
        if not n_samples:
            denom = 1 + n_hyp * max(0, (dim - 1))
            n_samples = max(1, budget // max(1, denom))
            params["its_n_samples"] = int(n_samples)

        # If provided / derived combination still exceeds budget, attempt to reduce hypotheses to fit.
        if _cost_its(params["its_n_samples"], params["its_n_hypotheses"], dim) > budget:
            # try reducing hypotheses until cost fits (prefer reducing hypotheses rather than samples)
            while params["its_n_hypotheses"] > 1:
                params["its_n_hypotheses"] -= 1
                denom = 1 + params["its_n_hypotheses"] * max(0, (dim - 1))
                params["its_n_samples"] = max(1, budget // max(1, denom))
                if _cost_its(params["its_n_samples"], params["its_n_hypotheses"], dim) <= budget:
                    break
            if _cost_its(params["its_n_samples"], params["its_n_hypotheses"], dim) > budget:
                raise ValueError("Derived / provided ITS parameters exceed budget.")

        its = InverseTransformationSearch2(
            model=model,
            transformation=None,
            domain=None,
            n_samples=params["its_n_samples"],
            n_hypotheses=params["its_n_hypotheses"],
            mc_steps=params.get("its_mc_steps", 1),
            change_of_mind=params.get("its_change_of_mind", "score"),
            gaussian_filter_channel_wise=params.get("its_gaussian_filter_channel_wise", False),
            confidence_module=getattr(problem, "confidence_module", None),
            en_unique_class_condition=params.get("its_unique_class_condition", False),  # NEW
        )
        return ITSWRAPPER(its)

    raise KeyError(f"Unknown algorithm '{algo}'")

# --------------------------------------------------
# Generic objective factory (add wcd_lattice support)
# --------------------------------------------------
def make_search_objective(
    algo: str,
    model,
    val_loader,
    problem,
    budget: int,
    device: str = "cuda",
    repeats: int = 1,
    check_percent: float = 0.1,
    **algo_kwargs,
) -> Callable[[optuna.Trial], float]:
    problems_list = list(problem) if isinstance(problem, Sequence) else [problem]

    #Prepare problems for ITS-like algorithms before building anything
    if algo in ("its","its2"):
        problems_list = [ITSWRAPPER._prepare_problem(p) for p in problems_list]


    def clone_loader(loader):
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,  # keep deterministic if needed
            sampler=loader.sampler,  # careful: some samplers aren't reusable!
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            prefetch_factor=getattr(loader, "prefetch_factor", 2),
            persistent_workers=loader.persistent_workers if loader.num_workers > 0 else False,
        )

    loaders_list = [clone_loader(val_loader) for _ in problems_list]

    def objective(trial: optuna.Trial) -> float:
        torch.cuda.empty_cache()

        # Always sample parameters for the trial
        if algo in ("cd", "wcd", "wcd_lattice", "cd_multi_cyclus", "its", "its2", "random_search"):
            params = _PARAM_SAMPLERS[algo](
                trial, budget, device, val_loader, problems_list[0], **algo_kwargs
            )
        else:
            params = _PARAM_SAMPLERS[algo](trial, budget, **algo_kwargs)

        # If fixed_params are provided, ensure the sampled params match for the fixed keys.
        current_fixed_params = trial.system_attrs.get("fixed_params", {})

        if current_fixed_params:
            for key, fixed_value in current_fixed_params.items():
                if key in params:
                    sampled_value = params[key]
                    # Use a tolerance for float comparison
                    if isinstance(fixed_value, float) and isinstance(sampled_value, float):
                        assert abs(fixed_value - sampled_value) < 1e-9, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
                    else:
                        assert sampled_value == fixed_value, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
            # Update params with fixed values to ensure they are used
            params.update(current_fixed_params)

        base_params = params

        # Ensure derived params (like SHGO n_init top-up) are persisted
        trial.set_user_attr("full_params", base_params)


        # ---- Phase 1: run all problems to 0.1 ----
        partials = []
        acc_intermediates = []
        for p,loader in zip(problems_list, loaders_list):
            search_inst = build_search_algorithm(algo, base_params, problem=p, budget=budget, model=model)
            evaluator, acc_ckpt = _evaluate_model_partial(
                search_inst, p, model, loader, device=device, repeats=repeats, check_percent=check_percent
            )
            partials.append(evaluator)
            acc_intermediates.append(acc_ckpt)

        if acc_intermediates[0] is not None:
            avg_acc = sum(acc_intermediates) / len(acc_intermediates)
            trial.report(avg_acc, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # ---- Phase 2: continue all problems to 1.0 ----
        acc_finals = []
        for evaluator in partials:
            acc_finals.append(evaluator.run_until(1.0)["accuracy_mean"])
            gc.collect()
            torch.cuda.empty_cache()

        return sum(acc_finals) / len(acc_finals)

    return objective

# --------------------------------------------------
# Backward-compatible wrappers (add wcd_lattice)
# --------------------------------------------------
def make_shgo_objective(model, val_loader, problem, budget, grad_weight: int = 2, device="cuda", repeats=1, **_):
    return make_search_objective("shgo", model, val_loader, problem, budget, device=device, repeats=repeats, grad_weight=grad_weight)

def make_parallel_sa_objective(model, val_loader, problem, budget, device="cuda", repeats=1, **_):
    return make_search_objective("parallel_sa", model, val_loader, problem, budget, device=device, repeats=repeats)

def make_parallel_sa_resets_objective(model, val_loader, problem, budget, device="cuda", repeats=1, **_):
    return make_search_objective("parallel_sa_resets", model, val_loader, problem, budget, device=device, repeats=repeats)  # NEW

def make_evolutionary_search_objective(model, val_loader, problem, budget, device="cuda", repeats=1, min_pop: int = 4, **_):
    return make_search_objective("evolutionary", model, val_loader, problem, budget, device=device, repeats=repeats, min_pop=min_pop)

def make_pso_objective(model, val_loader, problem, budget, device="cuda", repeats=1, min_swarm: int = 4, **_):
    return make_search_objective("pso", model, val_loader, problem, budget, device=device, repeats=repeats, min_swarm=min_swarm)

def make_coordinate_descent_objective(model, val_loader, problem, budget, device="cuda", repeats=1, samples_min: int = 4, **_):
    return make_search_objective("cd", model, val_loader, problem, budget, device=device, repeats=repeats, samples_min=samples_min)

def make_weighted_coordinate_descent_objective(model, val_loader, problem, budget, device="cuda", repeats=1, rounds_range=(1,5), **_):
    return make_search_objective("wcd", model, val_loader, problem, budget, device=device, repeats=repeats, rounds_range=rounds_range)

def make_weighted_coordinate_descent_lattice_objective(model, val_loader, problem, budget, device="cuda", repeats=1, rounds_range=(1,5), **_):
    return make_search_objective("wcd_lattice", model, val_loader, problem, budget, device=device, repeats=repeats, rounds_range=rounds_range)

def make_cd_multi_cyclus_objective(model, val_loader, problem, budget, device="cuda", repeats=1, rounds_range=(1,5), **_):
    return make_search_objective("cd_multi_cyclus", model, val_loader, problem, budget, device=device, repeats=repeats, rounds_range=rounds_range)

def make_parallel_gd_objective(model, val_loader, problem, budget, device="cuda", repeats=1, grad_weight: int = 2, **_):
    return make_search_objective("pgd", model, val_loader, problem, budget, device=device, repeats=repeats, grad_weight=grad_weight)

def make_parallel_gd_restart_objective(model, val_loader, problem, budget, device="cuda", repeats=1, grad_weight: int = 2, **_):
    return make_search_objective("pgd_restart", model, val_loader, problem, budget, device=device, repeats=repeats, grad_weight=grad_weight)

def make_parallel_gd_window_objective(model, val_loader, problem, budget, device="cuda", repeats=1, grad_weight: int = 2, **_):
    return make_search_objective("pgd_window", model, val_loader, problem, budget, device=device, repeats=repeats, grad_weight=grad_weight)

def make_random_search_objective(model, val_loader, problem, budget, device="cuda", repeats=1, **_):
    return make_search_objective("random_search", model, val_loader, problem, budget, device=device, repeats=repeats)

def make_its_objective(model, val_loader, problem, budget, device="cuda", repeats=1, **_):
    return make_search_objective("its", model, val_loader, problem, budget, device=device, repeats=repeats)

def make_its2_objective(model, val_loader, problem, budget, device="cuda", repeats=1, **_):  # NEW
    return make_search_objective("its2", model, val_loader, problem, budget, device=device, repeats=repeats)

def make_cmaes_objective(model, val_loader, problem, budget, device="cuda", repeats=1, min_pop: int = 4, **_):
    return make_search_objective("cmaes", model, val_loader, problem, budget, device=device, repeats=repeats, min_pop=min_pop)

# NEW wrapper for CMAES Nevergrad
def make_cmaes_nevergrad_objective(model, val_loader, problem, budget, device="cuda", repeats=1, min_pop: int = 4, **_):
    return make_search_objective("cmaes_nevergrad", model, val_loader, problem, budget, device=device, repeats=repeats, min_pop=min_pop)

# --------------------------------------------------
# Persist / restore utilities for best trial parameters
# --------------------------------------------------
def save_best_trial_params(study: optuna.Study, algo: str, path: str, include_unrelated: bool = False):
    """
    Save best trial params including derived values (taken from user_attrs['full_params'] if present).
    """
    from search.config import filter_algo_params, save_params
    trial = study.best_trial
    params = trial.user_attrs.get("full_params")
    if params is None:
        # Fallback: merge suggested params only (derived lost)
        params = dict(trial.params)
    if not include_unrelated:
        params = filter_algo_params(params, algo)
    save_params(params, path)



# --------------------------------------------------
# Placeholder for future algorithms (parallel gradient, window stuck, restarting GD)
# --------------------------------------------------

__all__ = [
    "make_shgo_objective",
    "make_parallel_sa_objective",
    "make_evolutionary_search_objective",
    "make_pso_objective",
    "make_coordinate_descent_objective",
    "make_weighted_coordinate_descent_objective",
    "make_search_objective",
    "build_search_algorithm",
    "save_best_trial_params",
    "default_allocation",
    "make_parallel_gd_objective",
    "make_parallel_gd_restart_objective",
    "make_parallel_gd_window_objective",
    "make_its_objective",  # NEW
    "make_its2_objective",  # NEW
    "make_cmaes_objective",  # NEW
    "make_random_search_objective", # NEW
    "make_cmaes_nevergrad_objective",  # NEW
]

# Replace earlier simple try/except with multi-strategy import + fallback
try:
    from .config import default_allocation  # type: ignore
except Exception:
    try:
        from search.config import default_allocation  # type: ignore
    except Exception:
        def default_allocation(*args, **kwargs):  # type: ignore
            raise RuntimeError(
                "default_allocation not available. Run via 'python -m search.objective_generators' "
                "or ensure package imports succeed."
            )

if __name__ == "__main__":
    # Simple smoke test: show default allocations & computed costs
    print("== objective_generators smoke test ==")
    budgets = {
        "shgo": 80,
        "parallel_sa": 60,
        "evolutionary": 90,
        "pso": 75,
        "cd": 64,
        "wcd": 70,
        "pgd": 64,
        "pgd_restart": 64,
        "pgd_window": 64,
        "random_search": 80, # NEW
        "its": 50,  # NEW
        "its2": 50,  # NEW: include its2 in smoke test sampling
        "cmaes": 90,  # NEW
    }

    def compute_cost(method: str, params):
        if method == "shgo" or method == "random_search":
            return _cost_shgo(params["shgo_initial_samples"],
                              params["shgo_local_runs"],
                              params.get("shgo_local_steps", None),
                              params.get("grad_weight", None))
        if method == "parallel_sa":
            return _cost_parallel_sa(params["psa_parallel_runs"],
                                     params["psa_max_iterations"])
        if method == "evolutionary":
            return _cost_es(params["es_population"], params["es_iters"])
        if method == "cmaes":
            return _cost_es(params["cmaes_population"], params["cmaes_iters"])  # NEW
        if method == "pso":
            return _cost_pso(params["pso_swarm_size"], params["pso_steps"])
        if method == "cd":
            dim = params.get("cd_dim_estimate")  # may be absent now
            n_samp = params.get("cd_number_samples")
            if dim is None or n_samp is None:
                return None  # dimension-agnostic default; cost not computable here
            return _cost_cd(dim, n_samp)
        if method == "wcd":
            dim = params.get("wcd_dim_estimate")
            base = params.get("wcd_base_samples")
            ff = params.get("wcd_first_dim_factor")
            rounds = params.get("wcd_rounds")
            if None in (dim, base, ff, rounds):
                return None
            return _cost_wcd(dim, base, ff, rounds)
        if method in ("pgd", "pgd_restart", "pgd_window"):
            return _cost_pgd(params["pgd_parallel_runs"],
                             params["pgd_max_iterations"],
                             params.get("grad_weight", 2))
        if method == "its" or method == "its2":
            dim = params.get("its_dim_estimate", 5)
            if "its_n_samples" in params and "its_n_hypotheses" in params:
                return _cost_its(params["its_n_samples"], params["its_n_hypotheses"], dim)
            return None
        return None

    for m, b in budgets.items():
        try:
            alloc = default_allocation(m, b)
            cost = compute_cost(m, alloc)
            print(f"[{m}] budget={b} cost={cost} params={ {k:v for k,v in alloc.items() if not k.startswith('_')} }")
            if cost is not None and cost > b:
                print(f"  WARNING: cost exceeds budget")
        except Exception as e:
            print(f"[{m}] ERROR: {e}")

    # --------------------------------------------------
    # Parameter sampling test using Optuna samplers only (no model eval)
    # --------------------------------------------------
    print("\n== parameter sampling test (Optuna) ==")
    import random
    random.seed(0)
    torch.manual_seed(0)
    sampler_seed = 0
    n_trials = 30
    grad_weight_default = 2

    # Dummy dataset / loader / problem to satisfy samplers needing dimension (cd, wcd)
    class _DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 8
        def __getitem__(self, idx):
            x = torch.randn(1, 8, 8)
            y = torch.randint(0, 2, (1,)).item()
            return x, y
    dummy_loader = torch.utils.data.DataLoader(_DummyDataset(), batch_size=2)

    class _DummyProblem:
        def initial_param(self, batch: int, k: int):
            return torch.zeros(batch, k, 10)  # dimension=10
        def transform(self, data, params):
            return data
        def calc_complete_size(self):
            return 10

    dummy_problem = _DummyProblem()

    # Light model (not really used here, only kept for future integration)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))

    grad_weight_algos = {"shgo", "pgd", "pgd_restart", "pgd_window"}
    needs_loader = {"cd", "wcd", "its", "its2", "random_search"}  # added 'its2', 'random_search'

    for algo, budget in budgets.items():
        budget = random.randint(10,100)
        if algo not in _PARAM_SAMPLERS:
            print(f"[{algo}] sampler missing, skipping")
            continue
        print(f"\n[{algo}] sampling trials (budget={budget})")
        optuna_sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        study = optuna.create_study(direction="maximize", sampler=optuna_sampler)

        def sampling_objective(trial: optuna.Trial):
            if algo in needs_loader:
                params = _PARAM_SAMPLERS[algo](trial, budget, "cpu", dummy_loader, dummy_problem)  # type: ignore
            elif algo in grad_weight_algos:
                params = _PARAM_SAMPLERS[algo](trial, budget, grad_weight=grad_weight_default)  # type: ignore
            else:
                params = _PARAM_SAMPLERS[algo](trial, budget)  # type: ignore
            cost = compute_cost(algo, params)
            assert cost is None or cost <= budget, f"Sampled cost {cost} exceeds budget {budget} for {algo}"
            print(f"  trial={trial.number} cost={cost} params={ {k:v for k,v in params.items() if not k.startswith('_')} }")
            return 0.0  # dummy objective value

        study.optimize(sampling_objective, n_trials=n_trials, show_progress_bar=False)

    print("\nParameter sampling test complete.")
    # --------------------------------------------------
    # Objective-based sampling test to ensure derived params persisted
    # --------------------------------------------------
    print("\n== objective sampling persistence test ==")
    small_budgets = {
        "shgo": 30,
        "parallel_sa": 25,
        "evolutionary": 30,
        "pso": 28,
        "cd": 24,
        "wcd": 26,
        "pgd": 24,
        "pgd_restart": 24,
        "pgd_window": 24,
        "random_search": 30, # NEW
        "its": 20,  # NEW
        "its2": 20,  # NEW
        "cmaes": 30,  # NEW
    }

    # Tiny dataset (single batch) for fast objective evaluation
    class _TinyDataset(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self, idx):
            x = torch.randn(1, 8, 8)
            y = torch.randint(0, 2, (1,)).item()
            return x, y
    tiny_loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=1)

    class _TinyProblem:
        def __init__(self):
            self.confidence_module = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
            self.max_batch_size = 1
        def initial_param(self, batch: int, k: int=None):
            if k is None:
                return torch.zeros(batch, 6)
            return torch.zeros(batch, k, 6)
        def transform(self, data, params):
            return data
        # Added: minimal calculate_error satisfying optimizer interface
        def calculate_error(self, x, params, y=None):
            # Return a random error per candidate; other info placeholder zeros
            # params expected shape: (num_candidates, param_dim) or similar
            n = params.shape[0]
            errors = torch.rand(n, device=params.device) +torch.mean(params, dim=1) * 0.1  # slight param dependence
            other = torch.zeros(n, device=params.device)
            return errors, other
        def correct_param(self,params):
            return params

        def consolidate(self,x, best_param, best_error, classes_best):

            # Get indices of minimum error per sample
            best_indices = torch.argmin(best_error, dim=1, keepdim=True)  # shape: (batch_size, 1)
            # Gather best parameters: best_param has shape (batch_size, parallel_runs, param_dim)
            best_param_selected = best_param.gather(
                dim=1,
                index=best_indices.unsqueeze(-1).expand(-1, -1, best_param.size(-1))
            ).squeeze(1)
            # Gather best errors: best_error has shape (batch_size, parallel_runs)
            best_error_selected = best_error.gather(dim=1, index=best_indices).squeeze(1)
            # Gather best classes: classes_best has shape (batch_size, parallel_runs, class_dim)
            best_classes = classes_best.gather(
                dim=1,
                index=best_indices.unsqueeze(-1).expand(-1, -1, classes_best.size(-1))
            ).squeeze(1)
            return best_param_selected, best_error_selected, best_classes

        def sample_neighbor(self, params, amount=1.0):
            noise = torch.randn_like(params) * amount
            return params + noise

        def calc_complete_size(self):
            return 6

    tiny_problem = _TinyProblem()
    tiny_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))

    wrappers = {
        "shgo": make_shgo_objective,
        "parallel_sa": make_parallel_sa_objective,
        "evolutionary": make_evolutionary_search_objective,
        "pso": make_pso_objective,
        "cd": make_coordinate_descent_objective,
        "wcd": make_weighted_coordinate_descent_objective,
        "pgd": make_parallel_gd_objective,
        "pgd_restart": make_parallel_gd_restart_objective,
        "pgd_window": make_parallel_gd_window_objective,
        "its": make_its_objective,  # NEW
        "its2": make_its2_objective,  # NEW
        "cmaes": make_cmaes_objective,  # NEW
        "random_search": make_random_search_objective, # NEW
    }

    for algo, b in small_budgets.items():
        if algo not in wrappers:
            continue
        print(f"\n[{algo}] running study (budget={b})")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
        objective = wrappers[algo](tiny_model, tiny_loader, tiny_problem, b, device="cpu", repeats=1)
        study.optimize(objective, n_trials=2, show_progress_bar=False)
        full = study.best_trial.user_attrs.get("full_params", {})
        print(f" full_params: { {k:v for k,v in full.items()} }")
        # Assertions for key derived fields
        if algo == "shgo":
            assert "shgo_local_steps" in full, "Missing derived shgo_local_steps"
        if algo == "wcd":
            assert "wcd_rounds" in full and "wcd_first_dim_factor" in full, "Loaded WCD missing required weighting params"

    print("\nPersistence test complete.")

    # --------------------------------------------------
    # Save / Load cycle test to ensure derived params persist and cost matches
    # --------------------------------------------------
    print("\n== save/load verification test ==")
    from pathlib import Path
    from search.config import load_params, filter_algo_params
    temp_dir = Path("tmp_param_tests")
    temp_dir.mkdir(exist_ok=True)

    for algo, b in small_budgets.items():
        path = temp_dir / f"{algo}.yml"
        # Retrieve study from earlier loop? We rerun tiny quick study for isolation.
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1))
        objective = wrappers[algo](tiny_model, tiny_loader, tiny_problem, b, device="cpu", repeats=1)
        study.optimize(objective, n_trials=2, show_progress_bar=False)
        save_best_trial_params(study, algo, path)
        loaded = load_params(path)
        # Ensure derived keys still present where applicable
        if algo == "shgo":
            assert "shgo_local_steps" in loaded, "Loaded SHGO missing shgo_local_steps"
        # (No dimension-dependent assertions for cd / wcd anymore)

        # Cost consistency
        def _cost_for(algo: str, p: Dict[str, Any], dim: int, budget: int) -> int:
            if algo == "shgo":
                return _cost_shgo(p["shgo_initial_samples"], p["shgo_local_runs"], p["shgo_local_steps"], p.get("grad_weight", 2))
            if algo == "random_search":
                return budget # It's parameterless, cost is budget
            if algo == "parallel_sa":
                return _cost_parallel_sa(p["psa_parallel_runs"], p["psa_max_iterations"])
            if algo == "evolutionary":
                return _cost_es(p["es_population"], p["es_iters"])
            if algo == "cmaes":
                return _cost_es(p["cmaes_population"], p["cmaes_iters"], p.get("grad_weight", 2))  # NEW
            if algo == "pso":
                return _cost_pso(p["pso_swarm_size"], p["pso_steps"])
            if algo == "cd":
                # Derive uniform samples if not stored
                n = p.get("cd_number_samples", max(1, budget // max(1, dim)))
                return _cost_cd(dim, n)
            if algo == "wcd":
                rounds = p["wcd_rounds"]
                first_factor = p["wcd_first_dim_factor"]
                denom = rounds * (dim - 1 + first_factor)
                if denom <= 0:
                    denom = 1
                base = max(1, budget // denom)
                return _cost_wcd(dim, base, first_factor, rounds)
            if algo in ("pgd", "pgd_restart", "pgd_window"):
                return _cost_pgd(p["pgd_parallel_runs"], p["pgd_max_iterations"], p.get("grad_weight", 2))
            if algo == "its" or algo == "its2":
                if "its_n_samples" in p and "its_n_hypotheses" in p:
                    return _cost_its(p["its_n_samples"], p["its_n_hypotheses"], dim)
                return None

        def _extract_instance_cost(algo: str, inst, params: Dict[str, Any], dim: int, budget: int) -> Optional[int]:
            try:
                if algo == "shgo":
                    return _cost_shgo(inst.initial_samples, inst.local_runs, inst.local_max_steps, params.get("grad_weight", 2))
                if algo == "random_search":
                    return inst.initial_samples
                if algo == "parallel_sa":
                    return _cost_parallel_sa(inst.parallel_runs, inst.max_iterations)
                if algo == "evolutionary":
                    return _cost_es(inst.population_size, inst.num_iters)
                if algo == "cmaes":
                    return _cost_es(params["cmaes_population"], params["cmaes_iters"], params.get("grad_weight", 2))  # NEW
                if algo == "pso":
                    return _cost_pso(inst.swarm_size, inst.steps)
                if algo in ("pgd", "pgd_restart", "pgd_window"):
                    return _cost_pgd(params["pgd_parallel_runs"], params["pgd_max_iterations"], params.get("grad_weight", 2))
                # Skip cd / wcd (dimension & allocation reconstructed on build)
            except Exception:
                return None
            return None

        saved_cost = _cost_for(algo, loaded, dim=tiny_problem.calc_complete_size(), budget=b)
        assert saved_cost <= b, f"Loaded params exceed budget: {saved_cost}>{b} ({algo})"
        inst = build_search_algorithm(algo, loaded, problem=tiny_problem, budget=b)
        inst_cost = _extract_instance_cost(algo, inst, loaded, dim=tiny_problem.calc_complete_size(), budget=b)
        if inst_cost is not None:
            assert inst_cost == saved_cost, f"Instance cost mismatch {inst_cost}!={saved_cost} for {algo}"
        print(f"[{algo}] saved+loaded OK (cost={saved_cost}, budget={b}) file={path}")

    print("\nSave/Load verification complete.")
    print("Done.")
