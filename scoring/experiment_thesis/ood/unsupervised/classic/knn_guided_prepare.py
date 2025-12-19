from typing import Dict, Any
import optuna
import torch
import torch.nn as nn

from confidence.direct.logit_based import EnergyConfidence
from confidence.direct.prob_based import EntropyConfidence
from confidence.model.single_pass import SinglePassConfidence
from confidence.control.split import NNGuideSplitConfidence
from utils.transformation_problem import TransformationProblem
from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer, get_max_layer_index

def default_nn_guided_params() -> Dict[str, Any]:
    return {
        "k": 3,
        "base_confidence": "energy",  # "energy" or "entropy"
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }

def sample_nn_guided_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
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

    params = {
        "k": int(trial.suggest_int("k", 1, 50)),
        "base_confidence": trial.suggest_categorical("base_confidence", ["energy", "entropy"]),
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    return params

def create_nn_guided_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)

    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None

    # get embeddings and logits (final outputs) from train cache
    if params.get("use_correct_only", False):
        embeddings_t, logits_t, _ = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        # train_cache(...) convention: returns (embeddings, final_logits, classes) when return_final=True & return_y maybe
        embeddings_t, logits_t, _ = train_cache(
            layer, capture_modes=layer_io, flatten=True, return_final=True, reducer_select=reducer_name
        )

    # build base confidence module
    base_type = params.get("base_confidence", "energy")
    if base_type == "energy":
        base_conf = EnergyConfidence()
    elif base_type == "entropy":
        # EntropyConfidence expects logits if input_logits=True
        base_conf = EntropyConfidence(input_logits=True)
    else:
        raise ValueError(f"Unknown base_confidence: {base_type}")

    device = kwargs.get("device", torch.device("cpu"))
    # ensure correct dtype
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
        logits_t = logits_t.half()

    # create NN-guided split confidence and fit on (embeddings, logits)
    conf_split = NNGuideSplitConfidence(base_confidence=base_conf, k=int(params.get("k", 5)))
    conf_split.to(device)
    conf_split.fit((embeddings_t.to(device), logits_t.to(device)))
    # ensure conf_split on device
    conf_split.to(device)

    # wrap model and return TransformationProblem
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- New: simple "ones" confidence for debugging ---
class OneConfidence(nn.Module):
    """
    A tiny confidence module that always returns ones (per-sample).
    Useful for NN-guided debug where the bank confidence should be neutral.
    """
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, logits, y=None):
        # logits: shape [batch, classes] (or any where first dim is batch)
        batch = logits.shape[0] if hasattr(logits, "shape") and len(logits.shape) > 0 else 1
        return torch.ones(batch, dtype=logits.dtype if hasattr(logits, "dtype") else self.dtype, device=logits.device)

def create_nn_guided_one_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    Special NN-guided problem that uses OneConfidence for the bank logits (always returns 1).
    """
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)

    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None

    # get embeddings and logits (final outputs) from train cache
    if params.get("use_correct_only", False):
        embeddings_t, logits_t, _ = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        # train_cache(...) convention: returns (embeddings, final_logits, classes)
        # do NOT pass unsupported kwargs like return_final here
        embeddings_t, logits_t, _ = train_cache(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )

    # Use OneConfidence for the bank scores (always ones)
    base_conf = OneConfidence()

    device = kwargs.get("device", torch.device("cpu"))
    # ensure correct dtype
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
        logits_t = logits_t.half()

    # create NN-guided split confidence and fit on (embeddings, logits)
    conf_split = NNGuideSplitConfidence(base_confidence=base_conf, k=int(params.get("k", 3)))
    conf_split.to(device)
    conf_split.fit((embeddings_t.to(device), logits_t.to(device)))
    conf_split.to(device)

    # wrap model and return TransformationProblem
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- New: tiny wrapper to invert entropy outputs for debug (-entropy) ---
class InvertedEntropyConfidence(nn.Module):
    """
    Debug variant: wraps EntropyConfidence and returns the negated value.
    Accepts same input_logits semantics as EntropyConfidence.
    """
    def __init__(self, input_logits: bool = True):
        super().__init__()
        self.inner = EntropyConfidence(input_logits=input_logits)

    def forward(self, logits, y=None):
        return -self.inner(logits, y)


def create_nn_guided_neg_entropy_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    Explicit NN-guided special-case that uses negated entropy as the base confidence.
    """
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)

    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None

    # get embeddings and logits (final outputs) from train cache
    if params.get("use_correct_only", False):
        embeddings_t, logits_t, _ = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        embeddings_t, logits_t, _ = train_cache(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )

    # Use inverted entropy as base confidence
    base_conf = InvertedEntropyConfidence(input_logits=True)

    device = kwargs.get("device", torch.device("cpu"))
    # ensure correct dtype
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
        logits_t = logits_t.half()

    # create NN-guided split confidence and fit on (embeddings, logits)
    conf_split = NNGuideSplitConfidence(base_confidence=base_conf, k=int(params.get("k", 3)))
    conf_split.to(device)
    conf_split.fit((embeddings_t.to(device), logits_t.to(device)))
    conf_split.to(device)

    # wrap model and return TransformationProblem
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# register factories
OOD_DEFAULT_PARAM_FACTORIES["nn_guided"] = default_nn_guided_params
OOD_PARAM_SAMPLERS["nn_guided"] = sample_nn_guided_params
OOD_PROBLEM_FACTORIES["nn_guided"] = create_nn_guided_problem

# register factories for the special-case nn-guided-one (reuse defaults/sampler)
OOD_DEFAULT_PARAM_FACTORIES["nn_guided_one"] = default_nn_guided_params
OOD_PARAM_SAMPLERS["nn_guided_one"] = sample_nn_guided_params
OOD_PROBLEM_FACTORIES["nn_guided_one"] = create_nn_guided_one_problem

# register factories for the explicit neg-entropy special-case (reuse defaults/sampler)
OOD_DEFAULT_PARAM_FACTORIES["nn_guided_neg_entropy"] = default_nn_guided_params
OOD_PARAM_SAMPLERS["nn_guided_neg_entropy"] = sample_nn_guided_params
OOD_PROBLEM_FACTORIES["nn_guided_neg_entropy"] = create_nn_guided_neg_entropy_problem