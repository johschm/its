from typing import Dict, Any, List
import optuna
import torch

from confidence.unsupervised.classic.gram import FeatureGramTorchConfidence
from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import PredictedSplitConfidence
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer, get_max_layer_index

# If transform sequence needs to be constructed from dataset_info:
from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images  # used only as fallback
#TODO this likely needs fixing as gram likes a list input.

def default_gram_params() -> Dict[str, Any]:
    return {
        "num_poles_list": None,
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }

def sample_gram_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
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
    # sample up to 3 poles sets as an example; here we pick a single list or None
    poles_choice = trial.suggest_categorical("num_poles_list", [None, [1], [1,2], [1,2,3]])
    return {
        "num_poles_list": poles_choice,
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_gram_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    # 1. Get feature maps (do not flatten; gram expects per-channel features)
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    device = kwargs.get("device", torch.device("cpu"))
    if params.get("use_correct_only", False):
        features_t, _, classes_t = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=False, reducer_select=reducer_name
        )
    else:
        features_t, _, classes_t = train_cache(
            layer, capture_modes=layer_io, flatten=False, return_y=True, return_final=True, reducer_select=reducer_name
        )

    # features_t might be a single tensor; Gram accepts a list
    features_list = features_t if isinstance(features_t, list) else [features_t]

    # determine num_classes (fallback to dataset_info if available)
    if hasattr(dataset_info, "num_classes"):
        num_classes = dataset_info.num_classes
    else:
        num_classes = int(classes_t.max().item() + 1)

    # 2. Build and fit detector
    gram_detector = FeatureGramTorchConfidence(
        num_classes=num_classes,
        num_poles_list=params.get("num_poles_list", None),
        input_transform=None
    )
    gram_detector.to(device)
    gram_detector._fit(features_list, classes_t)
    gram_detector.to(device)

    # 3. Create the full confidence module structure
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=False)
    conf_split = PredictedSplitConfidence(gram_detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Registration
OOD_DEFAULT_PARAM_FACTORIES["gram"] = default_gram_params
OOD_PARAM_SAMPLERS["gram"] = sample_gram_params
OOD_PROBLEM_FACTORIES["gram"] = create_gram_problem