from __future__ import annotations

from typing import Protocol, Callable, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf


FVec = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


class ChangePointModel(Protocol):
    """Protocol for models that support change-point inference."""

    name: str
    param_dim_null: int
    param_dim_alt: int

    def simulate(self, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        ...

    def null_f_vec(self) -> FVec:
        ...

    def alt_f_vec(self, tau_tf: tf.Tensor) -> FVec:
        ...

    def alt_theta_init(self, theta_null: np.ndarray) -> np.ndarray:
        ...
