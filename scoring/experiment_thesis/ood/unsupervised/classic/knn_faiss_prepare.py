from typing import Dict, Any
import optuna

from confidence.unsupervised.classic.nn import KNNConfidence as FaissKNNConfidence, PerClassKNNConfidence as FaissPerClassKNNConfidence
from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import PredictedSplitConfidence
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer, get_max_layer_index

# --- Faiss KNN (minimal params) ---

def default_faiss_knn_params() -> Dict[str, Any]:
    return {
        "k": 3,
        "metric": "cosine",
        "layer_index": 0,
        "reducer_name": None,
        "use_gpu": True,   # forced
        "split_b": 0.0,    # forced
        "use_correct_only": False,
    }

def sample_faiss_knn_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
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
        "k": trial.suggest_int("k", 1, 50),
        "metric": trial.suggest_categorical("metric", ["euclidean", "cosine"]),
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "use_gpu": True,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_faiss_knn_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    # get layer and embeddings
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
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    # build and fit Faiss KNN (use defaults except k and metric, force GPU)
    knn_detector = FaissKNNConfidence(
        metric=params.get("metric", "euclidean"),
        number_of_neighbors=int(params.get("k", 5)),
        use_gpu=True,
    )
    knn_detector = knn_detector.fit(embeddings_t)
    # wrap into single-pass confidence pipeline
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select= reducer_name)
    conf_split = PredictedSplitConfidence(knn_detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Faiss Per-Class KNN (minimal params) ---

def default_faiss_per_class_knn_params() -> Dict[str, Any]:
    return {
        "k": 3,
        "metric": "euclidean",
        "layer_index": 0,
        "reducer_name": None,
        "use_gpu": True,   # forced
        "split_b": 0.0,    # forced
        "class_penalty": None,
        "debug_class_match": False,
        "use_correct_only": False,
    }

def sample_faiss_per_class_knn_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
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
        "k": trial.suggest_int("k", 1, 50),
        "metric": trial.suggest_categorical("metric", ["euclidean", "cosine"]),
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "use_gpu": True,
        "split_b": 0.0,
        "class_penalty": None,
        "debug_class_match": False,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_faiss_per_class_knn_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    # get layer and embeddings
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
    # build and fit per-class Faiss detector (defaults used, force GPU)
    pcknn_detector = FaissPerClassKNNConfidence(
        class_penalty=params.get("class_penalty", None),
        debug_class_match=params.get("debug_class_match", False),
        metric=params.get("metric", "euclidean"),
        number_of_neighbors=int(params.get("k", 5)),
        use_gpu=True,
    )
    pcknn_detector = pcknn_detector.fit(embeddings_t, classes_t)
    # wrap into single-pass confidence pipeline
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select= reducer_name)
    conf_split = PredictedSplitConfidence(pcknn_detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Register factories and samplers ---
OOD_DEFAULT_PARAM_FACTORIES["faiss_knn"] = default_faiss_knn_params
OOD_PARAM_SAMPLERS["faiss_knn"] = sample_faiss_knn_params
OOD_PROBLEM_FACTORIES["faiss_knn"] = create_faiss_knn_problem

OOD_DEFAULT_PARAM_FACTORIES["faiss_per_class_knn"] = default_faiss_per_class_knn_params
OOD_PARAM_SAMPLERS["faiss_per_class_knn"] = sample_faiss_per_class_knn_params
OOD_PROBLEM_FACTORIES["faiss_per_class_knn"] = create_faiss_per_class_knn_problem