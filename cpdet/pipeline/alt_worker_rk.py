from __future__ import annotations

import gc
import os
from typing import Any, Tuple

import numpy as np
from scipy.optimize import minimize

from cpdet.models import get_model


def _configure_tf_single_thread():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import tensorflow as tf

    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(False)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass


def rk4(ts, x0, theta, rhs_fn):
    x = np.asarray(x0, dtype=np.float64)
    traj = [x]
    ts_flat = ts.reshape(-1)
    for j in range(1, len(ts_flat)):
        t_prev = float(ts_flat[j - 1])
        dt = float(ts_flat[j] - ts_flat[j - 1])
        if dt <= 0:
            traj.append(x)
            continue
        k1 = rhs_fn(t_prev, x)
        k2 = rhs_fn(t_prev + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = rhs_fn(t_prev + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = rhs_fn(t_prev + dt, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x)
    return np.vstack(traj)


def tf_fvec_to_numpy(f_vec, theta_vec):
    import tensorflow as tf

    theta_tf = tf.constant(theta_vec, dtype=tf.float64)

    def rhs(t_scalar, x_vec):
        t_tf = tf.constant([[t_scalar]], dtype=tf.float64)
        x_tf = tf.constant(x_vec.reshape(1, -1), dtype=tf.float64)
        return np.asarray(f_vec(t_tf, x_tf, theta_tf)).reshape(-1)

    return rhs


def gaussian_nll(y, yhat, sigma2):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    n, d = y.shape
    resid = y - yhat
    sse = float(np.sum(resid * resid))
    sigma2 = float(max(sigma2, 1e-6))
    return (sse / (2 * sigma2)) + (n * d / 2) * np.log(sigma2) + (n * d / 2) * np.log(2 * np.pi)


def fit_theta(obj_fn, theta0):
    res = minimize(obj_fn, theta0, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-9})
    theta_hat = np.maximum(res.x, 1e-8)
    nll = float(obj_fn(theta_hat))
    return theta_hat, nll


def alt_map_worker_rk(args: Tuple[Any, ...]):
    """Run RK GLR alt optimization for one tau candidate in a separate process."""

    _configure_tf_single_thread()
    import tensorflow as tf

    (
        model_name,
        ts_obs,
        X_obs,
        theta_null,
        tau,
        k_idx,
        cp_idx,
        sigma2_hat,
    ) = args

    model = get_model(model_name)
    f_vec_alt_tf = model.alt_f_vec(tf.constant(tau, dtype=tf.float64))
    theta_alt0 = model.alt_theta_init(theta_null)

    def nll_alt_fn(theta_vec):
        theta_c = np.maximum(theta_vec, 1e-8)
        rhs = tf_fvec_to_numpy(f_vec_alt_tf, theta_c)
        traj = rk4(ts_obs[:, 0], X_obs[0], theta_c, rhs)
        return gaussian_nll(X_obs, traj, sigma2_hat)

    try:
        theta_alt_hat, nll_alt = fit_theta(nll_alt_fn, theta_alt0)
    except Exception:
        theta_alt_hat, nll_alt = theta_alt0, np.inf

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return {
        "nll": float(nll_alt),
        "theta_hat": theta_alt_hat,
        "k_idx": k_idx,
        "k_time": tau,
        "cp_idx": cp_idx,
    }
