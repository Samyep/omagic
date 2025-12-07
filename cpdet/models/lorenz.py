from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf

from .base import ChangePointModel, FVec


@dataclass
class LorenzRhoModel(ChangePointModel):
    """Lorenz system with a change in rho at unknown tau."""

    sigma: float = 10.0
    rho_left: float = 28.0
    rho_mid: float = 35.0
    rho_right: float = 20.0
    beta: float = 8.0 / 3.0
    t_cp1: float = 7.0
    t_cp2: float = 14.0
    dt: float = 0.01
    t_max: float = 95.0
    obs_stride: int = 10  # decimate every 10 steps
    noise_std: float = 0.05

    name: str = "lorenz"
    param_dim_null: int = 3  # (sigma, rho, beta)
    param_dim_alt: int = 4   # (sigma, beta, rho_L, rho_R)
    obs_columns: List[str] = None

    def __post_init__(self):
        if self.obs_columns is None:
            self.obs_columns = ["x_noisy", "y_noisy", "z_noisy"]

    def simulate(self, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        t = np.arange(0.0, self.t_max + self.dt, self.dt)
        n_steps = t.size

        def rho_time(tt):
            rho_t = np.full_like(tt, self.rho_left, dtype=float)
            rho_t[tt > self.t_cp1] = self.rho_mid
            rho_t[tt > self.t_cp2] = self.rho_right
            return rho_t

        def lorenz(state, tt):
            x, y, z = state
            current_rho = rho_time(np.array([tt]))[0]
            dx = self.sigma * (y - x)
            dy = x * (current_rho - z) - y
            dz = x * y - self.beta * z
            return np.array([dx, dy, dz])

        states = np.zeros((n_steps, 3), dtype=float)
        states[0] = np.array([1.0, 1.0, 1.0], dtype=float)
        for i in range(n_steps - 1):
            s = states[i]
            ti = t[i]
            k1 = self.dt * lorenz(s, ti)
            k2 = self.dt * lorenz(s + 0.5 * k1, ti + 0.5 * self.dt)
            k3 = self.dt * lorenz(s + 0.5 * k2, ti + 0.5 * self.dt)
            k4 = self.dt * lorenz(s + k3, ti + self.dt)
            states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        noisy = states * rng.normal(1.0, self.noise_std, size=states.shape)

        states_obs = states[:: self.obs_stride]
        noisy_obs = noisy[:: self.obs_stride]
        t_obs = t[:: self.obs_stride]

        cp_times = np.array([self.t_cp1, self.t_cp2], dtype=float)
        cp_idx = np.searchsorted(t_obs, cp_times)

        df = pd.DataFrame(
            {
                "t": t_obs,
                "rho_t": rho_time(t_obs),
                "x_clean": states_obs[:, 0],
                "y_clean": states_obs[:, 1],
                "z_clean": states_obs[:, 2],
                "x_noisy": noisy_obs[:, 0],
                "y_noisy": noisy_obs[:, 1],
                "z_noisy": noisy_obs[:, 2],
            }
        )
        df["is_cp"] = 0
        df.loc[cp_idx, "is_cp"] = 1

        return df, cp_idx, cp_times

    def null_f_vec(self) -> FVec:
        def f_vec(t, X, theta):
            sigma, rho, beta = theta[0], theta[1], theta[2]
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return tf.stack([dx, dy, dz], axis=1)

        return f_vec

    def alt_f_vec(self, tau_tf: tf.Tensor) -> FVec:
        def f_vec(t, X, theta):
            sigma = theta[0]
            beta = theta[1]
            rho_L = theta[2]
            rho_R = theta[3]
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            rho_t = tf.where(t[:, 0] < tau_tf, rho_L, rho_R)
            dx = sigma * (y - x)
            dy = x * (rho_t - z) - y
            dz = x * y - beta * z
            return tf.stack([dx, dy, dz], axis=1)

        return f_vec

    def alt_theta_init(self, theta_null: np.ndarray) -> np.ndarray:
        sigma0 = float(theta_null[0])
        rho0 = float(theta_null[1])
        beta0 = float(theta_null[2])
        return np.array([sigma0, beta0, rho0, rho0], dtype=np.float64)

    def prepare_window(self, window_df: pd.DataFrame, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        ts_obs = window_df["t"].to_numpy(dtype=np.float64).reshape(-1, 1)
        X_obs = window_df[self.obs_columns].to_numpy(dtype=np.float64)
        return ts_obs, X_obs

    def infer_change_value(self, theta_null: np.ndarray | None, best_alt: dict, T_stat: float, threshold: float) -> float:
        null_rho = np.nan
        try:
            null_rho = float(theta_null[1]) if theta_null is not None else np.nan
        except Exception:
            pass

        if (
            best_alt.get("theta_hat") is not None
            and best_alt.get("k_time") is not None
            and np.isfinite(T_stat)
            and T_stat > threshold
        ):
            try:
                return float(best_alt["theta_hat"][3])
            except Exception:
                return null_rho
        return null_rho


@dataclass
class LorenzUniformCPModel(ChangePointModel):
    """Lorenz system with three rho changes: start at 28, then three levels sampled in [18, 28]."""

    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    baseline_rho: float = 28.0
    rho_levels: Optional[Tuple[float, float, float, float]] = None  # (base, change1, change2, change3)
    rho_min: float = 18.0
    rho_max: float = 28.0
    n_obs: int = 950
    dt: float = 0.01
    noise_std: float = 0.01
    cp_min_idx: int = 60
    cp_max_idx: int = 900

    name: str = "lorenz_uniform"
    param_dim_null: int = 3  # (sigma, rho, beta)
    param_dim_alt: int = 4   # (sigma, beta, rho_L, rho_R)
    obs_columns: Tuple[str, str, str] = ("x_noisy", "y_noisy", "z_noisy")

    def simulate(self, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        # Choose three change points uniformly in the specified index range and sort them
        cp_idx = np.sort(rng.integers(self.cp_min_idx, self.cp_max_idx + 1, size=3))

        t = np.linspace(0.0, self.dt * (self.n_obs - 1), self.n_obs)

        if self.rho_levels is None:
            sampled = rng.uniform(self.rho_min, self.rho_max, size=3)
            rho_levels = np.concatenate(([self.baseline_rho], sampled))
        else:
            rho_levels = np.array(self.rho_levels, dtype=float)

        if rho_levels.size != 4:
            raise ValueError("rho_levels must have four entries (baseline + 3 changes)")

        def rho_time(idx_array: np.ndarray) -> np.ndarray:
            rho_t = np.full_like(idx_array, rho_levels[0], dtype=float)
            rho_t[idx_array >= cp_idx[0]] = rho_levels[1]
            rho_t[idx_array >= cp_idx[1]] = rho_levels[2]
            rho_t[idx_array >= cp_idx[2]] = rho_levels[3]
            return rho_t

        def lorenz_rhs(state: np.ndarray, idx: int) -> np.ndarray:
            x, y, z = state
            rho_t = rho_time(np.array([idx]))[0]
            dx = self.sigma * (y - x)
            dy = x * (rho_t - z) - y
            dz = x * y - self.beta * z
            return np.array([dx, dy, dz])

        states = np.zeros((self.n_obs, 3), dtype=float)
        states[0] = np.array([1.0, 1.0, 1.0], dtype=float)
        for i in range(self.n_obs - 1):
            s = states[i]
            k1 = self.dt * lorenz_rhs(s, i)
            k2 = self.dt * lorenz_rhs(s + 0.5 * k1, i)
            k3 = self.dt * lorenz_rhs(s + 0.5 * k2, i)
            k4 = self.dt * lorenz_rhs(s + k3, i + 1)
            states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        noisy = states + rng.normal(0.0, self.noise_std, size=states.shape)

        cp_times = t[cp_idx]

        df = pd.DataFrame(
            {
                "t": t,
                "x_clean": states[:, 0],
                "y_clean": states[:, 1],
                "z_clean": states[:, 2],
                "x_noisy": noisy[:, 0],
                "y_noisy": noisy[:, 1],
                "z_noisy": noisy[:, 2],
                "rho_true": rho_time(np.arange(self.n_obs)),
            }
        )
        df["is_cp"] = 0
        df.loc[cp_idx, "is_cp"] = 1

        return df, cp_idx.astype(np.int64), cp_times

    def null_f_vec(self) -> FVec:
        def f_vec(t, X, theta):
            sigma = theta[0]
            rho = theta[1]
            beta = theta[2]
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return tf.stack([dx, dy, dz], axis=1)

        return f_vec

    def alt_f_vec(self, tau_tf: tf.Tensor) -> FVec:
        def f_vec(t, X, theta):
            sigma = theta[0]
            beta = theta[1]
            rho_L = theta[2]
            rho_R = theta[3]
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            rho_t = tf.where(t[:, 0] < tau_tf, rho_L, rho_R)
            dx = sigma * (y - x)
            dy = x * (rho_t - z) - y
            dz = x * y - beta * z
            return tf.stack([dx, dy, dz], axis=1)

        return f_vec

    def alt_theta_init(self, theta_null: np.ndarray) -> np.ndarray:
        sigma0 = float(theta_null[0])
        rho0 = float(theta_null[1])
        beta0 = float(theta_null[2])
        return np.array([sigma0, beta0, rho0, rho0], dtype=np.float64)

    def prepare_window(self, window_df: pd.DataFrame, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        ts_obs = window_df["t"].to_numpy(dtype=np.float64).reshape(-1, 1)
        X_obs = window_df[list(self.obs_columns)].to_numpy(dtype=np.float64)
        return ts_obs, X_obs

    def infer_change_value(self, theta_null: np.ndarray | None, best_alt: dict, T_stat: float, threshold: float) -> float:
        null_rho = np.nan
        try:
            null_rho = float(theta_null[1]) if theta_null is not None else np.nan
        except Exception:
            pass

        if (
            best_alt.get("theta_hat") is not None
            and best_alt.get("k_time") is not None
            and np.isfinite(T_stat)
            and T_stat > threshold
        ):
            try:
                return float(best_alt["theta_hat"][3])
            except Exception:
                return null_rho
        return null_rho


__all__ = ["LorenzRhoModel", "LorenzUniformCPModel"]
