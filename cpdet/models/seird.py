from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf

from .base import ChangePointModel, FVec


@dataclass
class SEIRDBetaModel(ChangePointModel):
    """SEIRD model with a change in transmission beta at tau."""

    ve: float = 0.1
    vi: float = 0.1
    beta_low: float = 0.8
    beta_high: float = 0.1
    pd_const: float = 0.02
    pd_late: float = 0.05  # currently unused (pd fixed); kept for future extension
    t0_min: float = 50.0
    t0_max: float = 70.0
    t_max: float = 150.0
    dt: float = 0.1
    obs_interval: float = 0.5  # days
    noise_std: float = 0.05
    S0: float = 1_000_000.0
    E0: float = 1000.0
    I0: float = 500.0
    D0: float = 50.0

    name: str = "seird_beta"
    param_dim_null: int = 4  # (beta, ve, vi, pd)
    param_dim_alt: int = 5   # (ve, vi, pd, beta_L, beta_R)
    obs_columns: List[str] = None

    def __post_init__(self):
        if self.obs_columns is None:
            self.obs_columns = ["S_noisy", "E_noisy", "I_noisy", "D_noisy"]

    def simulate(self, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        t0_cp = rng.uniform(self.t0_min, self.t0_max)

        t = np.arange(0.0, self.t_max + self.dt, self.dt)
        n_steps = t.size

        beta1, beta2 = self.beta_low, self.beta_high
        pd_const = self.pd_const

        def beta_time(tt):
            out = np.full_like(tt, beta1, dtype=float)
            out[tt >= t0_cp] = beta2
            return out

        def seird_rhs(state, tt):
            S, E, I, D = state
            beta_t = beta_time(np.array([tt]))[0]
            dS = -beta_t * I * S / self.N_total
            dE = beta_t * I * S / self.N_total - self.ve * E
            dI = self.ve * E - self.vi * I
            dD = self.vi * I * pd_const
            return np.array([dS, dE, dI, dD])

        self.N_total = self.S0 + self.E0 + self.I0 + self.D0

        states = np.zeros((n_steps, 4), dtype=float)
        states[0] = np.array([self.S0, self.E0, self.I0, self.D0], dtype=float)
        for i in range(n_steps - 1):
            s = states[i]
            ti = t[i]
            k1 = self.dt * seird_rhs(s, ti)
            k2 = self.dt * seird_rhs(s + 0.5 * k1, ti + 0.5 * self.dt)
            k3 = self.dt * seird_rhs(s + 0.5 * k2, ti + 0.5 * self.dt)
            k4 = self.dt * seird_rhs(s + k3, ti + self.dt)
            states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        noisy = states * rng.normal(1.0, self.noise_std, size=states.shape)

        step_obs = int(round(self.obs_interval / self.dt))
        t_obs = t[::step_obs]
        states_obs = states[::step_obs]
        noisy_obs = noisy[::step_obs]

        cp_times = np.array([t0_cp], dtype=float)
        cp_idx = np.searchsorted(t_obs, cp_times)

        df = pd.DataFrame(
            {
                "t": t_obs,
                "S_clean": states_obs[:, 0],
                "E_clean": states_obs[:, 1],
                "I_clean": states_obs[:, 2],
                "D_clean": states_obs[:, 3],
                "S_noisy": noisy_obs[:, 0],
                "E_noisy": noisy_obs[:, 1],
                "I_noisy": noisy_obs[:, 2],
                "D_noisy": noisy_obs[:, 3],
                "beta_true": beta_time(t_obs),
            }
        )
        df["is_cp"] = 0
        df.loc[cp_idx, "is_cp"] = 1

        return df, cp_idx, cp_times

    def null_f_vec(self) -> FVec:
        def f_vec(t, X_log, theta):
            beta = theta[0]
            ve = theta[1]
            vi = theta[2]
            pd = theta[3]
            s, e, i, d = X_log[:, 0], X_log[:, 1], X_log[:, 2], X_log[:, 3]
            N_total = tf.exp(X_log[0, 0]) + tf.exp(X_log[0, 1]) + tf.exp(X_log[0, 2]) + tf.exp(X_log[0, 3])
            ds = -beta * tf.exp(i) / N_total
            de = beta * tf.exp(s + i - e) / N_total - ve
            di = ve * tf.exp(e - i) - vi
            dd = vi * pd * tf.exp(i - d)
            return tf.stack([ds, de, di, dd], axis=1)

        return f_vec

    def alt_f_vec(self, tau_tf: tf.Tensor) -> FVec:
        def f_vec(t, X_log, theta):
            ve = theta[0]
            vi = theta[1]
            pd = theta[2]
            beta_L = theta[3]
            beta_R = theta[4]
            s, e, i, d = X_log[:, 0], X_log[:, 1], X_log[:, 2], X_log[:, 3]
            beta_t = tf.where(t[:, 0] < tau_tf, beta_L, beta_R)
            N_total = tf.exp(X_log[0, 0]) + tf.exp(X_log[0, 1]) + tf.exp(X_log[0, 2]) + tf.exp(X_log[0, 3])
            ds = -beta_t * tf.exp(i) / N_total
            de = beta_t * tf.exp(s + i - e) / N_total - ve
            di = ve * tf.exp(e - i) - vi
            dd = vi * pd * tf.exp(i - d)
            return tf.stack([ds, de, di, dd], axis=1)

        return f_vec

    def alt_theta_init(self, theta_null: np.ndarray) -> np.ndarray:
        beta0 = float(theta_null[0])
        ve0 = float(theta_null[1])
        vi0 = float(theta_null[2])
        pd0 = float(theta_null[3])
        return np.array([ve0, vi0, pd0, beta0, beta0], dtype=np.float64)

    def prepare_window(self, window_df: pd.DataFrame, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        ts_obs = window_df["t"].to_numpy(dtype=np.float64).reshape(-1, 1)
        vals = window_df[self.obs_columns].to_numpy(dtype=np.float64)
        vals[vals <= 0] = eps
        X_obs = np.log(vals)
        return ts_obs, X_obs

    def infer_change_value(self, theta_null: np.ndarray | None, best_alt: dict, T_stat: float, threshold: float) -> float:
        null_beta = np.nan
        try:
            null_beta = float(theta_null[0]) if theta_null is not None else np.nan
        except Exception:
            pass

        if (
            best_alt.get("theta_hat") is not None
            and best_alt.get("k_time") is not None
            and np.isfinite(T_stat)
            and T_stat > threshold
        ):
            try:
                # alt theta: (ve, vi, pd, beta_L, beta_R)
                return float(best_alt["theta_hat"][4])
            except Exception:
                return null_beta
        return null_beta


@dataclass
class SEIRDPdModel(ChangePointModel):
    """SEIRD model with a change in death probability pd at tau (uniform in [90,110])."""

    ve: float = 0.1
    vi: float = 0.1
    beta_const: float = 0.8
    pd_low: float = 0.02
    pd_high: float = 0.05
    t1_min: float = 90.0
    t1_max: float = 110.0
    t_max: float = 150.0
    dt: float = 0.1
    obs_interval: float = 0.5  # days
    noise_std: float = 0.05
    S0: float = 1_000_000.0
    E0: float = 1000.0
    I0: float = 500.0
    D0: float = 50.0

    name: str = "seird_pd"
    param_dim_null: int = 4  # (beta, ve, vi, pd)
    param_dim_alt: int = 5   # (ve, vi, beta, pd_L, pd_R)
    obs_columns: List[str] = None

    def __post_init__(self):
        if self.obs_columns is None:
            self.obs_columns = ["S_noisy", "E_noisy", "I_noisy", "D_noisy"]

    def simulate(self, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        t_cp = rng.uniform(self.t1_min, self.t1_max)

        t = np.arange(0.0, self.t_max + self.dt, self.dt)
        n_steps = t.size

        beta_const = self.beta_const

        def pd_time(tt):
            out = np.full_like(tt, self.pd_low, dtype=float)
            out[tt >= t_cp] = self.pd_high
            return out

        def seird_rhs(state, tt):
            S, E, I, D = state
            pd_t = pd_time(np.array([tt]))[0]
            dS = -beta_const * I * S / self.N_total
            dE = beta_const * I * S / self.N_total - self.ve * E
            dI = self.ve * E - self.vi * I
            dD = self.vi * I * pd_t
            return np.array([dS, dE, dI, dD])

        self.N_total = self.S0 + self.E0 + self.I0 + self.D0

        states = np.zeros((n_steps, 4), dtype=float)
        states[0] = np.array([self.S0, self.E0, self.I0, self.D0], dtype=float)
        for i in range(n_steps - 1):
            s = states[i]
            ti = t[i]
            k1 = self.dt * seird_rhs(s, ti)
            k2 = self.dt * seird_rhs(s + 0.5 * k1, ti + 0.5 * self.dt)
            k3 = self.dt * seird_rhs(s + 0.5 * k2, ti + 0.5 * self.dt)
            k4 = self.dt * seird_rhs(s + k3, ti + self.dt)
            states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        noisy = states * rng.normal(1.0, self.noise_std, size=states.shape)

        step_obs = int(round(self.obs_interval / self.dt))
        t_obs = t[::step_obs]
        states_obs = states[::step_obs]
        noisy_obs = noisy[::step_obs]

        cp_times = np.array([t_cp], dtype=float)
        cp_idx = np.searchsorted(t_obs, cp_times)

        df = pd.DataFrame(
            {
                "t": t_obs,
                "S_clean": states_obs[:, 0],
                "E_clean": states_obs[:, 1],
                "I_clean": states_obs[:, 2],
                "D_clean": states_obs[:, 3],
                "S_noisy": noisy_obs[:, 0],
                "E_noisy": noisy_obs[:, 1],
                "I_noisy": noisy_obs[:, 2],
                "D_noisy": noisy_obs[:, 3],
                "pd_true": pd_time(t_obs),
            }
        )
        df["is_cp"] = 0
        df.loc[cp_idx, "is_cp"] = 1

        return df, cp_idx, cp_times

    def null_f_vec(self) -> FVec:
        def f_vec(t, X_log, theta):
            beta = theta[0]
            ve = theta[1]
            vi = theta[2]
            pd = theta[3]
            s, e, i, d = X_log[:, 0], X_log[:, 1], X_log[:, 2], X_log[:, 3]
            N_total = tf.exp(X_log[0, 0]) + tf.exp(X_log[0, 1]) + tf.exp(X_log[0, 2]) + tf.exp(X_log[0, 3])
            ds = -beta * tf.exp(i) / N_total
            de = beta * tf.exp(s + i - e) / N_total - ve
            di = ve * tf.exp(e - i) - vi
            dd = vi * pd * tf.exp(i - d)
            return tf.stack([ds, de, di, dd], axis=1)

        return f_vec

    def alt_f_vec(self, tau_tf: tf.Tensor) -> FVec:
        def f_vec(t, X_log, theta):
            ve = theta[0]
            vi = theta[1]
            beta = theta[2]
            pd_L = theta[3]
            pd_R = theta[4]
            s, e, i, d = X_log[:, 0], X_log[:, 1], X_log[:, 2], X_log[:, 3]
            pd_t = tf.where(t[:, 0] < tau_tf, pd_L, pd_R)
            N_total = tf.exp(X_log[0, 0]) + tf.exp(X_log[0, 1]) + tf.exp(X_log[0, 2]) + tf.exp(X_log[0, 3])
            ds = -beta * tf.exp(i) / N_total
            de = beta * tf.exp(s + i - e) / N_total - ve
            di = ve * tf.exp(e - i) - vi
            dd = vi * pd_t * tf.exp(i - d)
            return tf.stack([ds, de, di, dd], axis=1)

        return f_vec

    def alt_theta_init(self, theta_null: np.ndarray) -> np.ndarray:
        beta0 = float(theta_null[0])
        ve0 = float(theta_null[1])
        vi0 = float(theta_null[2])
        pd0 = float(theta_null[3])
        return np.array([ve0, vi0, beta0, pd0, pd0], dtype=np.float64)

    def prepare_window(self, window_df: pd.DataFrame, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        ts_obs = window_df["t"].to_numpy(dtype=np.float64).reshape(-1, 1)
        vals = window_df[self.obs_columns].to_numpy(dtype=np.float64)
        vals[vals <= 0] = eps
        X_obs = np.log(vals)
        return ts_obs, X_obs

    def infer_change_value(self, theta_null: np.ndarray | None, best_alt: dict, T_stat: float, threshold: float) -> float:
        null_pd = np.nan
        try:
            null_pd = float(theta_null[3]) if theta_null is not None else np.nan
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
                return null_pd
        return null_pd


__all__ = ["SEIRDBetaModel", "SEIRDPdModel"]
