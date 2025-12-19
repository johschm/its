from __future__ import annotations

import gc
import warnings
from typing import Dict, Any, Callable, Optional, Tuple, List, Union
import optuna
import torch
from torch.utils.data import DataLoader

from confidence.base_confidence import ConfidenceModule
from embedding_cache import LayerEmbeddingCache
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer
from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images  # << new import
from utils.eval.ood_performance import progressive_confidence_evaluation, ConfidenceEvaluator, \
    evaluate_search_with_progressive_reporting
import optuna  # ensure optuna available locally
from utils.transformation_problem import TransformationProblem
from confidence.utils import ModelInputOutputWrapper

# --- Registries for OOD Detectors ---

OOD_DEFAULT_PARAM_FACTORIES: Dict[str, Callable[[], Dict[str, Any]]] = {}
OOD_PARAM_SAMPLERS: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {}
OOD_PROBLEM_FACTORIES: Dict[str, Callable[..., TransformationProblem]] = {}
OOD_MODEL_PARAM_EXTRACTORS: Dict[str, Callable[[ConfidenceModule], List[Any]]] = {}

# Prefix patterns for OOD detectors
OOD_PARAM_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "knn": ("knn_",),
    "per_class_knn": ("pcknn_",),
    "energy": ("energy_",),
    "vim": ("vim_",),
    "she": ("she_",),    # added SHE prefix
    "pca": ("pca_",),    # added PCA prefix
    "react": ("react_",),
}

# detectors that have no tunable hyperparameters and therefore should be skipped for Optuna runs
OOD_PARAMLESS = {
}

# --- Helper Functions ---

def get_default_ood_params(detector_name: str) -> Dict[str, Any]:
    """Get default parameters for a given OOD detector."""
    if detector_name not in OOD_DEFAULT_PARAM_FACTORIES:
        raise KeyError(f"Unknown OOD detector '{detector_name}'")
    return OOD_DEFAULT_PARAM_FACTORIES[detector_name]()

def sample_ood_params(trial: optuna.Trial, detector_name: str, **kwargs) -> Dict[str, Any]:
    """Sample parameters for a given OOD detector using Optuna."""
    if detector_name not in OOD_PARAM_SAMPLERS:
        raise KeyError(f"No parameter sampler for OOD detector '{detector_name}'")
    return OOD_PARAM_SAMPLERS[detector_name](trial, **kwargs)

def create_ood_problem(
    detector_name: str,
    params: Dict[str, Any],
    **factory_kwargs,
) -> TransformationProblem:
    """
    Create a TransformationProblem for a given OOD detector using its factory.
    This function is used both during optimization (with sampled params) and
    after optimization (with the best loaded params).

    If callers omit 'transform_seq' we attempt to resolve it from dataset_info
    (first look for dataset_info.transform_seq, then use dataset_info.transform_seq_name
    and dataset_info.resample_method via get_transformation_sequence_images).
    This centralizes transform_seq handling so specific OOD prepare modules
    don't have to repeat the logic.
    """
    if detector_name not in OOD_PROBLEM_FACTORIES:
        raise KeyError(f"No problem factory for OOD detector '{detector_name}'")
    
    # Resolve transform_seq if missing or None
    if "transform_seq" not in factory_kwargs or factory_kwargs.get("transform_seq") is None:
        dataset_info = factory_kwargs.get("dataset_info")
        resolved = None
        if dataset_info is not None:
            # try direct attribute first
            try:
                resolved = getattr(dataset_info, "transform_seq", None)
            except Exception:
                resolved = None
            if resolved is None:
                # fallback to name-based construction if available
                seq_name = getattr(dataset_info, "transform_seq_name", None)
                resample = getattr(dataset_info, "resample_method", None)
                if seq_name not in (None, "", "none"):
                    try:
                        resolved = get_transformation_sequence_images(name=seq_name, resample_method=resample,init_method="sobol")
                    except Exception:
                        resolved = None
        if resolved is not None:
            factory_kwargs["transform_seq"] = resolved

    # Build the problem via factory
    problem = OOD_PROBLEM_FACTORIES[detector_name](params=params, **factory_kwargs)

    # If a device hint was provided, move the problem to CUDA by calling .cuda()
    # (best-effort; some problems may not implement cuda())
    device_arg = factory_kwargs.get("device", None)
    if device_arg is not None:
            problem.confidence_module.to(device_arg)
    dataset_info = factory_kwargs.get("dataset_info", None)
    problem.max_batch_size = dataset_info.batch_size_search

    return problem

# --- Save/Load and Objective Functions ---

def load_best_problem_from_study(
    study_name: str,
    storage_path: str,
    detector_name: str,
    **factory_kwargs,
) -> TransformationProblem:
    """
    Loads a completed study, gets the best parameters, and creates the TransformationProblem.

    Args:
        study_name: The name of the study to load.
        storage_path: Path to the Optuna storage file (e.g., "sqlite:///ood_studies.db").
        detector_name: The name of the OOD detector that was tuned.
        **factory_kwargs: The same arguments needed to build the problem (model, caches, etc.).

    Returns:
        The fully configured TransformationProblem with the best found hyperparameters.
    """
    study = optuna.load_study(study_name=study_name, storage=storage_path)
    best_params = get_best_ood_params_from_study(study)
    
    # Get model state from best trial's user_attrs and add to factory_kwargs
    if "model_params" in study.best_trial.user_attrs:
        # This part is tricky because user_attrs might contain raw tensors
        # which are not part of the main DB storage. This is more for live studies.
        # The file-based approach in the main script is more robust for persistence.
        pass

    print("Loading problem with best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    problem = create_ood_problem(
        detector_name=detector_name,
        params=best_params,
        **factory_kwargs,
    )

    # The factory itself is now responsible for loading any model state
    # passed via factory_kwargs.
    return problem


def get_best_ood_params_from_study(study: optuna.Study) -> Dict[str, Any]:
    """Extracts the best hyperparameter set from a completed Optuna study."""
    # The 'full_params' user attribute is where we stored the complete dict.
    if "full_params" in study.best_trial.user_attrs:
        return study.best_trial.user_attrs["full_params"]
    else:
        # Fallback for search objective where params are split
        params = study.best_params
        if "full_ood_params" in study.best_trial.user_attrs:
            params.update(study.best_trial.user_attrs["full_ood_params"])
        return params

@torch.no_grad()
def run_ood_study(
    study_name: str,
    storage_path: str,
    detector_name: str,
    objective_type: str,
    objective_kwargs: Dict[str, Any],
    n_trials: int = 50,
    enqueue_params: Optional[List[Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]]] = None,
) -> optuna.Study:
    """
    High-level wrapper to create, run, and save an Optuna study for an OOD detector.
    Uses median pruning and supports enqueuing initial trials.

    Args:
        ...
        enqueue_params: An optional list of trials to enqueue. Each item can be either
                        a parameter dictionary, or a tuple of (params_dict, user_attrs_dict).
                        If None, the detector's default parameters will be enqueued.
    """
    # (The initial logic for parameterless detectors and objective creation remains the same)
    metric_objectives = {
        "auc": "auroc", "auroc": "auroc", "aupr_in": "aupr_in", "aupr_out": "aupr_out",
        "fpr95": "fpr95", "tnr95": "tnr95", "detection_error": "detection_error",
        "paired_ood_acc": "paired_ood_acc",
    }
    direction = "maximize"
    if objective_type in ["fpr95", "detection_error"]:
        direction = "minimize"

    if objective_type == "search":
        objective = make_ood_search_objective(detector_name=detector_name, **objective_kwargs)
    elif objective_type in metric_objectives:
        metric = metric_objectives[objective_type]
        objective = make_ood_auc_objective(detector_name=detector_name, **{**objective_kwargs, "metric": metric})
    else:
        raise ValueError(f"Unknown objective_type '{objective_type}'.")


    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, len(enqueue_params) if enqueue_params else 1))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction=direction,
        load_if_exists=True,
        pruner=pruner,
    )

    # Enqueue initial trials only if the study is new
    if len(study.trials) == 0:
        params_to_enqueue = []
        if enqueue_params is None:
            try:
                default_params = get_default_ood_params(detector_name)
                if default_params:
                    print(f"Enqueuing trial with default parameters for '{detector_name}'.")
                    params_to_enqueue.append(default_params)
            except Exception as e:
                print(f"Warning: Failed to enqueue default parameters for '{detector_name}': {e}")
        else:
            params_to_enqueue.extend(enqueue_params)

        for trial_info in params_to_enqueue:
            params, user_attrs = None, None
            if isinstance(trial_info, tuple):
                params, user_attrs = trial_info
            else:
                params = trial_info

            if params:
                try:
                    study.enqueue_trial(params, user_attrs=user_attrs)
                    print(f"Enqueued trial with parameters: {params}")
                    if user_attrs:
                        print(f"  ... with user_attrs: {list(user_attrs.keys())}")
                except Exception as e:
                    print(f"Warning: Failed to enqueue trial with params {params}: {e}")

    study.optimize(objective, n_trials=n_trials,gc_after_trial=True)

    print(f"Study {study_name} complete.")
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params: ")
    for key, value in get_best_ood_params_from_study(study).items():
        print(f"    {key}: {value}")

    gc.collect()
    torch.cuda.empty_cache()

    return study


def _resolve_safe_batch_size(dataset_info=None, train_cache=None, default: int = 128) -> int:
    """
    Resolve a safe max_batch_size for a TransformationProblem.
    Preference order:
      1. dataset_info.batch_size_search
    """
    return dataset_info.batch_size_search

@torch.no_grad()
def make_ood_auc_objective(
    detector_name: str,
    model: torch.nn.Module,
    train_cache: LayerEmbeddingCache,
    id_loader,                # DataLoader for in-distribution (validation/true) samples
    ood_loader,               # DataLoader for out-of-distribution samples
    transform_seq: Any,
    dataset_info: Dict,
    architecture: str,
    device: str = "cuda",
    metric: str = "auroc",
    check_percent: float = 0.1,
    prune_at: Optional[float] = 0.1,
    max_batches: Optional[int] = None,
    show_progress: bool = False,
    val_id_loader: Optional[DataLoader] = None,  # Added for fitting
    val_ood_loader: Optional[DataLoader] = None,  # Added for fitting
) -> Callable[[optuna.Trial], float]:
    """
    Creates an objective function to optimize OOD detector hyperparameters based on an
    OOD detection metric (default AUROC). Uses progressive_confidence_evaluation to
    support intermediate reports and Optuna pruning. The frequency of pruning checks
    is controlled by `check_percent` (fraction of dataset, default 0.1 => 10%).
    """
    def objective(trial: optuna.Trial) -> float:

        sampler_kwargs = {"train_cache": train_cache, "architecture": architecture,"dataset_info": dataset_info}
        factory_kwargs = {
            "model": model, "train_cache": train_cache, "transform_seq": transform_seq,
            "dataset_info": dataset_info, "architecture": architecture, "device": device,
            "val_id_loader": val_id_loader, "val_ood_loader": val_ood_loader,
        }

        # Check if model parameters were passed via user_attrs (from an enqueued trial)
        if "model_params" in trial.user_attrs:
            print(f"Trial {trial.number}: Found pre-loaded model_params in user_attrs.")
            factory_kwargs["model_params"] = trial.user_attrs["model_params"]

        # Sample parameters for the trial
        params = sample_ood_params(trial, detector_name, **sampler_kwargs)

        # If fixed_params are provided, ensure the sampled params match for the fixed keys.
        # This is a safeguard, especially for enqueued default trials.
        #ignore future warnings for fixed_params
        with warnings.catch_warnings():
            fixed_params  = trial.system_attrs.get("fixed_params", None)
        if fixed_params:
            for key, fixed_value in fixed_params.items():
                if key in params:
                    sampled_value = params[key]
                    # Use a tolerance for float comparison
                    if isinstance(fixed_value, float) and isinstance(sampled_value, float):
                        assert abs(fixed_value - sampled_value) < 1e-9, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
                    else:
                        assert sampled_value == fixed_value, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
                else:
                    print(f"Warning: Fixed parameter '{key}' not found in sampled params for trial {trial.number}. Adding it.")
                    print(f"Warning: Fixed parameter '{key}' not found in sampled params for trial {trial.number}. Adding it.")
                    print(f"Warning: Fixed parameter '{key}' not found in sampled params for trial {trial.number}. Adding it.")
            # Update params with fixed values to ensure they are used, especially if a sampler
            # doesn't sample a parameter that is in fixed_params.
            params.update(fixed_params)

        trial.set_user_attr("full_params", params)

        # Create fully-wired problem via factory
        problem = create_ood_problem(
            detector_name,
            params,
            **factory_kwargs,
        )
        # ensure module on device
        try:
            problem.confidence_module.to(device)
        except Exception:
            pass

        # Progressive evaluation with pruning support (checkpoints controlled by check_percent)
        res = progressive_confidence_evaluation(
            confidence_module=problem.confidence_module,
            id_loader=id_loader,
            ood_loader=ood_loader,
            device=device,
            metric=metric,
            trial=trial,
            check_percent=check_percent,
            prune_at=prune_at,
            max_batches=max_batches,
            show_progress=show_progress,
        )

        # Use registered extractor to get model parameters, if any
        if detector_name in OOD_MODEL_PARAM_EXTRACTORS:
            extractor = OOD_MODEL_PARAM_EXTRACTORS[detector_name]
            model_params = extractor(problem.confidence_module)
            if model_params:
                trial.set_user_attr("model_params", model_params)

        # store some info for inspection
        trial.set_user_attr("full_params", params)
        trial.set_user_attr("ood_eval_info", {
            "id_count": res.get("id_count"),
            "ood_count": res.get("ood_count"),
            "metric": res.get("metric"),
        })

        # Explicit cleanup before returning
        for name, module in problem.confidence_module.named_modules():
            if isinstance(module, ModelInputOutputWrapper):
                module.clear()

        #here maybe check wether one of the named children is a modelinputoutputwrapper and if so, maybe delete the hooks, do for search also
        del problem
        return float(res["metric"])

    return objective


# python
import copy
from typing import Optional, Callable, Any, Dict
# (place the import near other top-level imports in `experiment_thesis/ood/base_prepare.py`)

def make_ood_search_objective(
    detector_name: str,
    optimizer: Optional[Any] = None,
    model: torch.nn.Module = None,
    train_cache: LayerEmbeddingCache = None,
    val_loader = None,
    transform_seq: Any = None,
    dataset_info: Dict = None,
    architecture: str = None,
    device: str = "cuda",
    fixed_params: Optional[Dict[str, Any]] = None,
    report_fraction: float = 0.1,
    repeats: Union[int, float] = 1,
    val_id_loader: Optional[DataLoader] = None,  # Added for fitting
    val_ood_loader: Optional[DataLoader] = None,  # Added for fitting
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective that tunes OOD detector params by evaluating their
    performance inside the downstream search procedure.
    """

    def objective(trial: optuna.Trial) -> float:
        sampler_kwargs = {"train_cache": train_cache, "architecture": architecture,"dataset_info": dataset_info}
        factory_kwargs = {
            "model": model, "train_cache": train_cache, "transform_seq": transform_seq,
            "dataset_info": dataset_info, "architecture": architecture, "device": device,
            "val_id_loader": val_id_loader, "val_ood_loader": val_ood_loader,
        }

        # Check if model parameters were passed via user_attrs (from an enqueued trial)
        if "model_params" in trial.user_attrs:
            print(f"Trial {trial.number}: Found pre-loaded model_params in user_attrs.")
            factory_kwargs["model_params"] = trial.user_attrs["model_params"]

        # Sample parameters for the trial
        params = sample_ood_params(trial, detector_name, **sampler_kwargs)

        # If fixed_params are provided, ensure the sampled params match for the fixed keys.
        if fixed_params:
            for key, fixed_value in fixed_params.items():
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
            params.update(fixed_params)

        trial.set_user_attr("full_params", params)

        # Build problem with current OOD params
        problem = create_ood_problem(
            detector_name,
            params,
            **factory_kwargs,
        )




        # Use ConfidenceEvaluator to run partial evaluation (supports run_until with early stopping)
        evaluator = ConfidenceEvaluator(
            model=model,
            optimizer=optimizer,
            problem=problem,
            test_loader=val_loader,
            repeats=repeats,
            show_progress=True,
        )

        # Phase 1: run until early checkpoint and report intermediate accuracy

        intermediate_acc = evaluator.run_until(report_fraction)["accuracy_mean"]
        trial.report(intermediate_acc, step=1)
        if trial.should_prune():
            # Clean up before pruning
            del evaluator, problem
            raise optuna.TrialPruned()


        # Phase 2: finish evaluation to completion
        final_res = evaluator.run_until(1.0)
        final_acc = final_res["accuracy_mean"]

        # Use registered extractor to get model parameters, if any
        if detector_name in OOD_MODEL_PARAM_EXTRACTORS:
            extractor = OOD_MODEL_PARAM_EXTRACTORS[detector_name]
            model_params = extractor(problem.confidence_module)
            if model_params:
                trial.set_user_attr("model_params", model_params)

        # store info for inspection
        trial.set_user_attr("full_params", params)
        trial.set_user_attr("search_eval_info", {
            "intermediate_acc": intermediate_acc,
            "final_acc": final_acc,
        })

        # Explicit cleanup before returning
        for name, module in problem.confidence_module.named_modules():
            if isinstance(module, ModelInputOutputWrapper):
                module.clear()
                pass

        del evaluator, problem
        return float(final_acc)

    return objective



@torch.no_grad()
def run_ood_study_halving(
    study_name: str,
    storage_path: str,
    detector_name: str,
    objective_type: str,
    objective_kwargs: Dict[str, Any],
    n_trials: int = 50,
    enqueue_params: Optional[List[Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]]] = None,
    min_resource: int = 1,
    reduction_factor: int = 3,
    check_every_n_batches_percentage: float = 0.02,
    load_if_exists: bool = False,
) -> optuna.Study:
    """
    Run Optuna study with SuccessiveHalvingPruner for search optimization.
    Reports at fixed intervals (every N batches) for fair comparison.
    """
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        min_early_stopping_rate=0,
    )
    direction = "maximize"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=load_if_exists,
        direction=direction,
        pruner=pruner,
    )

    # Enqueue initial trials only if the study is new
    if len(study.trials) == 0 and enqueue_params:
        for trial_info in enqueue_params:
            params, user_attrs = None, None
            if isinstance(trial_info, tuple):
                params, user_attrs = trial_info
            else:
                params = trial_info
            if params:
                study.enqueue_trial(params, user_attrs=user_attrs)

    # Use the progressive reporting objective for search
    objective = make_ood_search_objective_with_progressive_reporting(
        detector_name=detector_name,
        check_every_n_batches_percentage=check_every_n_batches_percentage,
        **objective_kwargs
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print(f"Study {study_name} complete.")
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params: ")
    for key, value in get_best_ood_params_from_study(study).items():
        print(f"    {key}: {value}")

    gc.collect()
    torch.cuda.empty_cache()

    return study



def make_ood_search_objective_with_progressive_reporting(
    detector_name: str,
    optimizer: Optional[Any] = None,
    model: torch.nn.Module = None,
    train_cache: LayerEmbeddingCache = None,
    val_loader = None,
    transform_seq: Any = None,
    dataset_info: Dict = None,
    architecture: str = None,
    device: str = "cuda",
    fixed_params: Optional[Dict[str, Any]] = None,
    check_every_n_batches_percentage: float = 0.01,
    repeats: Union[int, float] = 1,
    val_id_loader: Optional[DataLoader] = None,
    val_ood_loader: Optional[DataLoader] = None,
        *args,**kwargs,
) -> Callable[[optuna.Trial], float]:
    """
    Optuna objective for OOD search, reporting accuracy every N batches for SuccessiveHalvingPruner.
    """
    def objective(trial: optuna.Trial) -> float:
        sampler_kwargs = {"train_cache": train_cache, "architecture": architecture, "dataset_info": dataset_info}
        factory_kwargs = {
            "model": model, "train_cache": train_cache, "transform_seq": transform_seq,
            "dataset_info": dataset_info, "architecture": architecture, "device": device,
            "val_id_loader": val_id_loader, "val_ood_loader": val_ood_loader,
        }

        if "model_params" in trial.user_attrs:
            factory_kwargs["model_params"] = trial.user_attrs["model_params"]

        params = sample_ood_params(trial, detector_name, **sampler_kwargs)
        # Calculate batch interval based on percentage
        num_batches = len(val_loader)
        check_every_n_batches = max(1, int(num_batches * check_every_n_batches_percentage))

        if fixed_params:
            for key, fixed_value in fixed_params.items():
                if key in params:
                    sampled_value = params[key]
                    if isinstance(fixed_value, float) and isinstance(sampled_value, float):
                        assert abs(fixed_value - sampled_value) < 1e-9, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
                    else:
                        assert sampled_value == fixed_value, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
            params.update(fixed_params)

        trial.set_user_attr("full_params", params)

        problem = create_ood_problem(
            detector_name,
            params,
            **factory_kwargs,
        )

        result = evaluate_search_with_progressive_reporting(
            model=model,
            optimizer=optimizer,
            problem=problem,
            test_loader=val_loader,
            trial=trial,
            check_every_n_batches=check_every_n_batches,
        )

        if detector_name in OOD_MODEL_PARAM_EXTRACTORS:
            extractor = OOD_MODEL_PARAM_EXTRACTORS[detector_name]
            model_params = extractor(problem.confidence_module)
            if model_params:
                trial.set_user_attr("model_params", model_params)

        trial.set_user_attr("search_eval_info", {
            "final_acc": result["accuracy_mean"],
        })

        for name, module in problem.confidence_module.named_modules():
            if isinstance(module, ModelInputOutputWrapper):
                module.clear()

        del problem
        return float(result["accuracy_mean"])

    return objective


