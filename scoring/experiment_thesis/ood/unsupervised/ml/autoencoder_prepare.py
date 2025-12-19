from typing import Dict, Any
import optuna
import torch
from torch import nn

from confidence.input_transform import InputTransform
from confidence.unsupervised.ml.autoencoder import BasicAutoencoderConfidence
from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import PredictedSplitConfidence
from experiment_thesis.dataset_preperation.basic_networks import get_network_layer, get_max_layer_index, get_network, get_encoder_for_resnet

# --- Autoencoder on Embeddings ---

def default_autoencoder_embeddings_params() -> Dict[str, Any]:
    return {
        "latent_fraction": 0.5,
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
    }

def sample_autoencoder_embeddings_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    latent_fraction = trial.suggest_float("latent_fraction", 0.1, 0.9)
    params = {
        "latent_fraction": latent_fraction,
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
    }
    return params

def create_autoencoder_embeddings_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """Factory for creating a TransformationProblem with Autoencoder on embeddings."""
    # 1. Get embeddings
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    embeddings_t, _, classes_t = train_cache(
        layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True, reducer_select=reducer_name
    )
    input_dim = embeddings_t.shape[1]
    latent_dim = int(input_dim * params["latent_fraction"])
    hidden_dim = input_dim
    
    # 2. Build encoder and decoder (3-layer fully connected GELU network)
    encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, latent_dim),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, input_dim),
    )
    
    # 3. Build and fit autoencoder
    autoencoder = BasicAutoencoderConfidence(
        encoder=encoder,
        decoder=decoder,
        input_transform=None,
        trainer_kwargs={"max_epochs": 10, "enable_progress_bar": False},
        dataloader_kwargs={"batch_size": 128},
        optimizer_type=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    )
    device = kwargs.get("device", torch.device("cpu"))
    autoencoder.to(device)
    autoencoder.fit(embeddings_t, classes_t)
    
    # 4. Create the full confidence module structure
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_split = PredictedSplitConfidence(autoencoder, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)
    
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Autoencoder on Images ---

def default_autoencoder_images_params() -> Dict[str, Any]:
    return {
        "latent_dim": 128,
        "split_b": 0.0,
    }

def sample_autoencoder_images_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    latent_dim = trial.suggest_int("latent_dim", 32, 512)
    params = {
        "latent_dim": latent_dim,
        "split_b": 0.0,
    }
    return params

def create_autoencoder_images_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """Factory for creating a TransformationProblem with Autoencoder on images."""
    # 1. Get model and encoder/decoder
    model = get_network(dataset_info, architecture)
    encoder, decoder = get_encoder_for_resnet(model, dim=params["latent_dim"])
    
    # 2. Build autoencoder
    autoencoder = BasicAutoencoderConfidence(
        encoder=encoder,
        decoder=decoder,
        input_transform=None,
        trainer_kwargs={"max_epochs": 10, "enable_progress_bar": False},
        dataloader_kwargs={"batch_size": 128},
        optimizer_type=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    )
    
    # 3. Get train and val loaders and fit
    train_loader = kwargs.get("train_loader")
    val_loader = kwargs.get("val_loader")
    device = kwargs.get("device", torch.device("cpu"))
    autoencoder.to(device)
    autoencoder.fit(train_loader, val_loader)
    
    # 4. Create the full confidence module structure
    dual_output_model = nn.Identity()  # Input is images directly
    conf_split = PredictedSplitConfidence(autoencoder, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)
    
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["autoencoder_embeddings"] = default_autoencoder_embeddings_params
OOD_PARAM_SAMPLERS["autoencoder_embeddings"] = sample_autoencoder_embeddings_params
OOD_PROBLEM_FACTORIES["autoencoder_embeddings"] = create_autoencoder_embeddings_problem

OOD_DEFAULT_PARAM_FACTORIES["autoencoder_images"] = default_autoencoder_images_params
OOD_PARAM_SAMPLERS["autoencoder_images"] = sample_autoencoder_images_params
OOD_PROBLEM_FACTORIES["autoencoder_images"] = create_autoencoder_images_problem