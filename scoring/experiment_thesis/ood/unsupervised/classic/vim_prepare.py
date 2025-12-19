from typing import Dict, Any
import optuna

from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.unsupervised.classic.VIM import ViMTorchConfidence
from confidence.control.classify import ClassifyingConfidence
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer

# --- ViMConfidence ---

def default_vim_params(train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    """
    Default principal subspace percentage for ViM:
    - feature_percent: None (infer from dataset_info)
    """
    return {
        "feature_percent": -1,
        "use_energy": True,
        "layer_index": 0,
        "reducer_name": None,
        "use_correct_only": False,
    }

def sample_vim_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    """
    Search range for principal subspace percentage: 1/3 to 2/3
    """
    feature_percent = trial.suggest_float("feature_percent", 0.01, 0.99)
    return {
        "feature_percent": feature_percent,
        "use_energy": trial.suggest_categorical("use_energy", [True, False]),
        "layer_index": 0,
        "reducer_name": None,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_vim_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """Factory for creating a TransformationProblem with ViM."""
    # 1. Get embeddings
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    
    if params.get("use_correct_only", False):
        embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        embeddings_t, _, classes_t = train_cache(
            layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True, reducer_select=reducer_name
        )
    embed_dim = embeddings_t.shape[1]
    num_classes = getattr(dataset_info, "num_classes", None)
    if num_classes is None:
        num_classes = len(set(classes_t.tolist()))

    # 2. Determine principal subspace dimension
    feature_percent = params.get("feature_percent", None)
    if feature_percent is None or feature_percent == -1:
        if num_classes < 0.9 * embed_dim:
            n_dim = num_classes
        else:
            n_dim = int(0.5 * embed_dim)
    else:
        n_dim = int(feature_percent * embed_dim) if feature_percent is not None else int(0.5 * embed_dim)

    # 3. Build and fit detector
    vim_detector = ViMTorchConfidence(
        model=train_cache.model, # ViM needs the base model
        n_dim=n_dim,
        use_energy=params["use_energy"]
    )
    device = kwargs.get("device", "cpu")
    embeddings_t = embeddings_t.to(device)
    classes_t = classes_t.to(device)
    vim_detector.fit(embeddings_t, y=classes_t)

    # 4. Create the full confidence module structure
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True)
    conf_split = ClassifyingConfidence(vim_detector, index=1, index_confidence=0)
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)
    
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["vim"] = default_vim_params
OOD_PARAM_SAMPLERS["vim"] = sample_vim_params
OOD_PROBLEM_FACTORIES["vim"] = create_vim_problem