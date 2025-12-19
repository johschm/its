from typing import Dict, Any
import optuna
import torch

from confidence.unsupervised.classic.gmm import GaussianMixtureConfidence
from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import PredictedSplitConfidence
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer, get_max_layer_index

# If transform sequence needs to be constructed from dataset_info:
from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images  # used only as fallback


def default_gmm_params() -> Dict[str, Any]:
    return {
        "n_components": 2,
        "covariance_type": "full",
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }

def sample_gmm_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    max_layer = get_max_layer_index(dataset_info, architecture)
    layer_index = trial.suggest_int("layer_index", 0, max_layer)
    reducer_names = train_cache.reducer_name if train_cache else None
    reducer_names = [None] + reducer_names if reducer_names else [None]
    reducer_name = trial.suggest_categorical("reducer_name", reducer_names)
    if train_cache:
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    return {
        "n_components": trial.suggest_int("n_components", 1, 10),
        "covariance_type": trial.suggest_categorical("covariance_type", ["full", "diag", "spherical", "tied"]),
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_gmm_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    # 1. Get embeddings
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    if params.get("use_correct_only", False):
        embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        embeddings_t, _, classes_t = train_cache(
            layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True, reducer_select=reducer_name
        )

    # 2. Build and fit detector
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    gmm_detector = GaussianMixtureConfidence(
        n_components=params.get("n_components", 2),
        covariance_type=params.get("covariance_type", "full"),
        input_transform=None,
        reg_covar=1e-6
    )
    device = kwargs.get("device", torch.device("cpu"))
    gmm_detector.to(device)
    gmm_detector._fit(embeddings_t.to(dtype_map.get(params.get("dtype", "float32"), torch.float32)))
    gmm_detector.to(device)

    # 3. Create the full confidence module structure
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True)
    conf_split = PredictedSplitConfidence(gmm_detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Registration
OOD_DEFAULT_PARAM_FACTORIES["gmm"] = default_gmm_params
OOD_PARAM_SAMPLERS["gmm"] = sample_gmm_params
OOD_PROBLEM_FACTORIES["gmm"] = create_gmm_problem