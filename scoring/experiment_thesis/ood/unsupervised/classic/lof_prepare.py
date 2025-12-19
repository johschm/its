from typing import Dict, Any
import optuna
import torch

from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.control.classify import ClassifyingConfidence
from confidence.unsupervised.classic.lof import LOFTorchConfidence
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer, get_max_layer_index


# --- LOF ---

def default_lof_params(train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    return {
        "n_neighbors": 20,
        "layer_index": 0,
        "reducer_name": None,
        "use_correct_only": False,
    }

def sample_lof_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
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
        "n_neighbors": n_neighbors,
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_lof_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.get("use_correct_only", False):
        embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        embeddings_t, _, classes_t = train_cache(
            layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True, reducer_select=reducer_name
        )
    embeddings_t = embeddings_t.to(device)
    classes_t = classes_t.to(device)
    lof_detector = LOFTorchConfidence(n_neighbors=params.get("n_neighbors", 20))
    lof_detector = lof_detector.to(device)
    lof_detector.fit(embeddings_t, y=classes_t)
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_split = ClassifyingConfidence(lof_detector, index=1, index_confidence=0)
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Registration ---
OOD_DEFAULT_PARAM_FACTORIES["lof"] = default_lof_params
OOD_PARAM_SAMPLERS["lof"] = sample_lof_params
OOD_PROBLEM_FACTORIES["lof"] = create_lof_problem