from __future__ import annotations

from typing import Protocol, Callable, Tuple, List
import pandas as pd
import numpy as np
import tensorflow as tf


FVec = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


class ChangePointModel(Protocol):
    """Protocol for models that support change-point inference."""

    name: str
    param_dim_null: int
    param_dim_alt: int
    obs_columns: List[str]

    def simulate(self, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        ...

    def null_f_vec(self) -> FVec:
        ...

    def alt_f_vec(self, tau_tf: tf.Tensor) -> FVec:
        ...

    def alt_theta_init(self, theta_null: np.ndarray) -> np.ndarray:
        ...

    def prepare_window(self, window_df: pd.DataFrame, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ts_obs, X_obs) for the current window; apply any preprocessing (e.g., log)."""
        ...

    def infer_change_value(self, theta_null: np.ndarray | None, best_alt: dict, T_stat: float, threshold: float) -> float:
        """Optional helper to extract an inferred post-change parameter; default NaN."""
        ...
