from typing import Dict, Any
import optuna

from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import MaxLogitConfidence

# --- MaxLogitConfidence ---

def default_max_logit_params() -> Dict[str, Any]:
    return {}

def sample_max_logit_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {}

def create_max_logit_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    conf_mod = SinglePassConfidence(model, MaxLogitConfidence(mean=False))
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- MeanLogitConfidence ---

def default_mean_logit_params() -> Dict[str, Any]:
    return {}

def sample_mean_logit_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {}

def create_mean_logit_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    conf_mod = SinglePassConfidence(model, MaxLogitConfidence(mean=True))
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["max_logit"] = default_max_logit_params
OOD_PARAM_SAMPLERS["max_logit"] = sample_max_logit_params
OOD_PROBLEM_FACTORIES["max_logit"] = create_max_logit_problem
