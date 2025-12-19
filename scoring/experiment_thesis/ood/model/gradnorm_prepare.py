from typing import Dict, Any
import optuna
from typing import Callable, Optional
import torch

from experiment_thesis.ood import base_prepare
from utils.transformation_problem import TransformationProblem  # kept for typing hints

from confidence.model.gradnorm import (
    GradNormConfidence,
    FuncGradNormConfidence,
    BackpackGradNormConfidence,
)

# Default params
def default_gradnorm_params() -> Dict[str, Any]:
    return {
        "variant": "func",      # choices: "backpack", "func", "orig"
        "index": None,              # index to select logits if needed
        "param_filter_names": [],   # optional substrings to filter parameter names
    }

# Sampler (accepts extra kwargs from base_prepare.sample_ood_params)
def sample_gradnorm_params(trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
    return {
        "variant": trial.suggest_categorical("variant", ["backpack", "func"]),
        "index": None,
        "param_filter_names": [],
    }

def create_gradnorm_problem(params: Dict[str, Any], model, transform_seq=None, dataset_info=None, architecture=None, **kwargs) -> TransformationProblem:
    """
    Create and return a GradNorm-based confidence module (nn.Module).
    Returning the raw nn.Module allows base_prepare to centrally wrap it
    into a problem-like object (object.confidence_module) if needed.
    """
    variant = params.get("variant", "backpack")
    index = params.get("index", None)
    pf_names = params.get("param_filter_names", [])

    # construct param filter
    if pf_names:
        def param_filter(name):
            return any(pat in name for pat in pf_names)
    else:
        param_filter = lambda name: True

    if variant == "backpack":
        conf_mod = BackpackGradNormConfidence(model, confidence=None, param_filter=param_filter, index=index)
    elif variant == "func":
        conf_mod = FuncGradNormConfidence(model, confidence=None, param_filter=param_filter, index=index)
    else:
        conf_mod = GradNormConfidence(model, confidence=None, param_filter=param_filter, index=index)

    # Return a TransformationProblem using the confidence module directly (consistent with energy/knn prepares)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Register into base_prepare registries ---
base_prepare.OOD_DEFAULT_PARAM_FACTORIES["gradnorm"] = default_gradnorm_params
base_prepare.OOD_PARAM_SAMPLERS["gradnorm"] = sample_gradnorm_params
base_prepare.OOD_PROBLEM_FACTORIES["gradnorm"] = create_gradnorm_problem