from typing import Dict, Any
import optuna
import torch

from confidence.unsupervised.classic.pca import PCATorchConfidence
from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import PredictedSplitConfidence
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer, get_max_layer_index

# If transform sequence needs to be constructed from dataset_info:
from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images  # used only as fallback


def default_pca_params(train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    params = {
        # Either specify absolute n_components (int) OR a ratio in (0,1] via n_components_ratio.
        "n_components": None,
        "n_components_ratio": 0.75,   # new: relative fraction of embedding dim to keep
        "vim_scaling": False,
        "square": False,
        "use_residual": False,
        # removed 'dtype' — embeddings are used as-is
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }
    # Double-check reducer_name validity if train_cache is provided
    if train_cache is not None:
        layer_index = params["layer_index"]
        reducer_name = params["reducer_name"]
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            params["reducer_name"] = None
    return params

def sample_pca_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    reducer_names = [None]
    if train_cache and hasattr(train_cache, "reducers") and train_cache.reducers:
        reducer_names.extend(list(train_cache.reducers.keys()))

    max_layer = get_max_layer_index(kwargs.get("dataset_info", None), architecture)
    layer_index = trial.suggest_int("layer_index", 0, max_layer)

    n_components_ratio = trial.suggest_float("n_components_ratio", 0.05, 0.95)

    return {
        "n_components": None,  # leave to factory to compute from ratio (keeps compatibility)
        "n_components_ratio": n_components_ratio,
        "vim_scaling": trial.suggest_categorical("vim_scaling", [False, True]),
        "square": trial.suggest_categorical("square", [False, True]),
        "use_residual": trial.suggest_categorical("use_residual", [False, True]),
        "layer_index": layer_index,
        "reducer_name": trial.suggest_categorical("reducer_name", reducer_names),
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_pca_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """Factory for creating a TransformationProblem with PCATorchConfidence."""
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # --- Reducer validity check ---
    if train_cache is not None:
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

    # infer embedding dimension and number of classes
    D = max(1, embeddings_t.shape[1])
    num_classes = getattr(dataset_info, "num_classes", None)
    if num_classes is None:
        try:
            num_classes = int(len(set(classes_t.tolist())))
        except Exception:
            num_classes = None

    # Determine n_components:
    n_components = params.get("n_components", None)
    ratio = params.get("n_components_ratio", None)

    if n_components is None:
        if ratio is not None:
            # convert ratio to integer count and clamp to [1, D-1]
            n = max(1, int(round(ratio * D)))
            n = min(n, max(1, D - 1))
            n_components = n
        else:
            # fallback heuristic similar to ViM: if num_classes is small relative to embed dim, use num_classes, else use half dim
            if num_classes is not None and num_classes < 0.9 * D:
                n_components = max(1, num_classes)
            else:
                n_components = max(1, D // 2)

    # Safety check
    if not (0 < n_components < D):
        n_components = max(1, min(n_components, D - 1))

    # Do not cast embeddings dtype here — use cache-provided embeddings as-is

    pca = PCATorchConfidence(
        n_components=n_components,
        map_function=None,
        vim_scaling=params.get("vim_scaling", False),
        square=params.get("square", False),
        use_residual=params.get("use_residual", False),
    )

    device = kwargs.get("device", torch.device("cpu"))
    pca.to(device)
    pca.fit(embeddings_t.to(device), classes_t.to(device))
    pca.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True)
    conf_split = PredictedSplitConfidence(pca, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Register
OOD_DEFAULT_PARAM_FACTORIES["pca"] = default_pca_params
OOD_PARAM_SAMPLERS["pca"] = sample_pca_params
OOD_PROBLEM_FACTORIES["pca"] = create_pca_problem