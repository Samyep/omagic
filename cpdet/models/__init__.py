from __future__ import annotations

from typing import Dict, Type

from .base import ChangePointModel
from .lotka_volterra import LotkaVolterraPoissonModel
from .lorenz import LorenzRhoModel, LorenzUniformCPModel
from .seird import SEIRDBetaModel, SEIRDPdModel

_MODEL_REGISTRY: Dict[str, Type[ChangePointModel]] = {
    "lotka_volterra": LotkaVolterraPoissonModel,
    "lv_poisson": LotkaVolterraPoissonModel,
    "lorenz": LorenzRhoModel,
    "lorenz_uniform": LorenzUniformCPModel,
    "seird": SEIRDBetaModel,
    "seird_beta": SEIRDBetaModel,
    "seird_pd": SEIRDPdModel,
}


def get_model(name: str) -> ChangePointModel:
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[key]()


__all__ = [
    "ChangePointModel",
    "get_model",
    "LotkaVolterraPoissonModel",
    "LorenzRhoModel",
    "LorenzUniformCPModel",
    "SEIRDBetaModel",
    "SEIRDPdModel",
]
