from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf

from .base import ChangePointModel, FVec


@dataclass
class LotkaVolterraPoissonModel(ChangePointModel):
    """Lotka-Volterra model with Poisson change-points on gamma(t)."""

    alpha: float = 0.6
    beta: float = 0.75
    delta: float = 1.0
    gamma_low: float = 0.6
    gamma_high: float = 1.0
    lambda_rate: float = 0.08  # per year
    obs_per_year: int = 12
    n_obs: int = 1000
    min_no_cp_obs: int = 60
    step_sub: int = 10
    noise_std: float = 0.05

    name: str = "lotka_volterra"
    param_dim_null: int = 4
    param_dim_alt: int = 5
    obs_columns: List[str] = None

    def __post_init__(self):
        if self.obs_columns is None:
            self.obs_columns = ["x_noisy", "y_noisy"]

    def simulate(self, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        obs_dt = 1.0 / self.obs_per_year
        dt = obs_dt / self.step_sub
        n_steps = self.n_obs * self.step_sub

        t = np.arange(n_steps, dtype=float) * dt
        t_obs = t[:: self.step_sub]
        assert len(t_obs) == self.n_obs
        t_max = t_obs[-1]

        min_cp_time = t_obs[self.min_no_cp_obs]

        cp_times = []
        t_cp = min_cp_time
        while True:
            t_cp += rng.exponential(1.0 / self.lambda_rate)
            if t_cp >= t_max:
                break
            cp_times.append(t_cp)
        cp_times = np.array(cp_times, dtype=float)

        def gamma_time(tt):
            tt_arr = np.asarray(tt, dtype=float)
            if cp_times.size == 0:
                return np.full_like(tt_arr, self.gamma_low, dtype=float)
            k = np.searchsorted(cp_times, tt_arr, side="right")
            return np.where(k % 2 == 0, self.gamma_low, self.gamma_high)

        def lotka_volterra(state, tt):
            x, y = state
            g = gamma_time(tt)
            dx = self.alpha * x - self.beta * x * y
            dy = self.delta * x * y - g * y
            return np.array([dx, dy])

        states = np.zeros((n_steps, 2), dtype=float)
        states[0] = np.array([2.0, 1.0], dtype=float)

        for i in range(n_steps - 1):
            s = states[i]
            ti = t[i]
            k1 = dt * lotka_volterra(s, ti)
            k2 = dt * lotka_volterra(s + 0.5 * k1, ti + 0.5 * dt)
            k3 = dt * lotka_volterra(s + 0.5 * k2, ti + 0.5 * dt)
            k4 = dt * lotka_volterra(s + k3, ti + dt)
            states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        noisy = states * rng.normal(1.0, self.noise_std, size=states.shape)

        states_obs = states[:: self.step_sub]
        noisy_obs = noisy[:: self.step_sub]

        if cp_times.size > 0:
            cp_idx = np.searchsorted(t_obs, cp_times)
            mask = cp_idx < self.n_obs
            cp_idx = cp_idx[mask]
            cp_times_local = cp_times[mask]
        else:
            cp_idx = np.array([], dtype=int)
            cp_times_local = np.array([], dtype=float)

        df = pd.DataFrame(
            {
                "t": t_obs,
                "gamma_t": gamma_time(t_obs),
                "x_clean": states_obs[:, 0],
                "y_clean": states_obs[:, 1],
                "x_noisy": noisy_obs[:, 0],
                "y_noisy": noisy_obs[:, 1],
            }
        )
        df["is_cp"] = 0
        if cp_idx.size > 0:
            df.loc[cp_idx, "is_cp"] = 1

        return df, cp_idx, cp_times_local

    def null_f_vec(self) -> FVec:
        def f_vec(t, X, theta):
            alpha, beta, delta, gamma = theta[0], theta[1], theta[2], theta[3]
            r, f = X[:, 0], X[:, 1]
            dr = alpha - beta * tf.exp(f)
            df = delta * tf.exp(r) - gamma
            return tf.stack([dr, df], axis=1)

        return f_vec

    def alt_f_vec(self, tau_tf: tf.Tensor) -> FVec:
        def f_vec(t, X, theta):
            alpha, beta, delta = theta[0], theta[1], theta[2]
            gamma_L, gamma_R = theta[3], theta[4]
            r, f = X[:, 0], X[:, 1]
            gamma_t = tf.where(t[:, 0] < tau_tf, gamma_L, gamma_R)
            dr = alpha - beta * tf.exp(f)
            df = delta * tf.exp(r) - gamma_t
            return tf.stack([dr, df], axis=1)

        return f_vec

    def alt_theta_init(self, theta_null: np.ndarray) -> np.ndarray:
        alpha0, beta0, delta0, gamma0 = (
            float(theta_null[0]),
            float(theta_null[1]),
            float(theta_null[2]),
            float(theta_null[3]),
        )
        return np.array([alpha0, beta0, delta0, gamma0, gamma0], dtype=np.float64)

    def prepare_window(self, window_df: pd.DataFrame, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        ts_obs = window_df["t"].to_numpy(dtype=np.float64).reshape(-1, 1)
        vals = window_df[self.obs_columns].to_numpy(dtype=np.float64)
        X_obs = np.log(np.maximum(vals, eps))
        return ts_obs, X_obs

    def infer_change_value(self, theta_null: np.ndarray | None, best_alt: dict, T_stat: float, threshold: float) -> float:
        null_gamma = np.nan
        try:
            null_gamma = float(theta_null[3]) if theta_null is not None else np.nan
        except Exception:
            pass

        if (
            best_alt.get("theta_hat") is not None
            and best_alt.get("k_time") is not None
            and np.isfinite(T_stat)
            and T_stat > threshold
        ):
            try:
                return float(best_alt["theta_hat"][4])
            except Exception:
                return null_gamma
        return null_gamma
