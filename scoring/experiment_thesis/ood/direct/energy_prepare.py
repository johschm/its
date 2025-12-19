from typing import Dict, Any
import optuna
import torch

from experiment_thesis.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.scaler.calibration import TemperatureCalibrationModule

# --- EnergyConfidence ---

def default_energy_params() -> Dict[str, Any]:
    return {"t":1.0}

def sample_energy_params(trial: optuna.Trial,**kwargs) -> Dict[str, Any]:
    # Add temperature sampling for energy
    params = {"t": trial.suggest_float("t", 0.5, 2.0)}
    return params

def create_energy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    Factory for creating a TransformationProblem with EnergyConfidence.
    This detector does not use embeddings or fitting.
    """
    energy_conf = EnergyConfidence(t=params.get("t", 1.0))
    
    # Wrap in SinglePassConfidence to use the base model directly
    conf_mod = SinglePassConfidence(model, energy_conf)
    
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Energy_TS (Temperature Scaled Energy) ---

def default_energy_ts_params() -> Dict[str, Any]:
    return {}

def sample_energy_ts_params(trial: optuna.Trial,**kwargs) -> Dict[str, Any]:
    # No hyperparameters for energy_ts by default (temperature scaling is handled by the wrapper)
    return {}

def create_energy_ts_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    Factory for creating a TransformationProblem with temperature-scaled EnergyConfidence.
    Wraps the model in TemperatureCalibrationModule and fits it on the calibration_loader (expected in kwargs).
    """
    # Wrap model with temperature scaler
    temp_scaler = TemperatureCalibrationModule(model)
    # Fit on calibration data if provided
    calibration_loader = kwargs.get('train_cache', None).dataloader
    if calibration_loader is not None:
        temp_scaler.fit(calibration_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError("calibration_loader must be provided in kwargs for energy_ts")
    
    energy_conf = EnergyConfidence(t=params.get("t", 1.0))
    
    # Wrap in SinglePassConfidence using the temperature-scaled model
    conf_mod = SinglePassConfidence(temp_scaler, energy_conf)
    
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["energy"] = default_energy_params
OOD_PARAM_SAMPLERS["energy"] = sample_energy_params
OOD_PROBLEM_FACTORIES["energy"] = create_energy_problem

OOD_DEFAULT_PARAM_FACTORIES["energy_ts"] = default_energy_ts_params
OOD_PARAM_SAMPLERS["energy_ts"] = sample_energy_ts_params
OOD_PROBLEM_FACTORIES["energy_ts"] = create_energy_ts_problem