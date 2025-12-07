from __future__ import annotations

import gc
import os
from typing import Any, Tuple

import numpy as np

from cpdet.models import get_model
from cpdet.utils.results import shrink_result


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


def alt_map_worker(args: Tuple[Any, ...]):
    """Run MAP for one alternative tau candidate in a separate process."""

    _configure_tf_single_thread()

    import tensorflow as tf

    (
        model_name,
        ts_obs,
        X_obs,
        theta_null,
        X_init_null,
        tau,
        k_idx,
        cp_idx,
        map_iters,
        map_patience,
        map_tol,
        map_lr_X,
        map_lr_theta,
    ) = args

    try:
        from python_magi.magi import MAGI
    except ImportError:
        print("[alt_worker] ERROR: python_magi.magi module not found.", flush=True)
        return {
            "loglik": -np.inf,
            "k_idx": k_idx,
            "k_time": tau,
            "cp_idx": cp_idx,
        }

    model = get_model(model_name)
    tau_tf = tf.constant(tau, dtype=tf.float64)
    f_vec_alt = model.alt_f_vec(tau_tf)

    magi_alt = MAGI(D_thetas=model.param_dim_alt, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec_alt)
    magi_alt.initial_fit(discretization=0, verbose=False)

    theta_init_alt = np.clip(model.alt_theta_init(theta_null), 1e-6, None)

    X_init_alt = X_init_null if X_init_null is not None else magi_alt.Xhat_init

    alt_res = magi_alt.predict(
        method="map",
        map_iters=map_iters,
        map_patience=map_patience,
        map_tol=map_tol,
        verbose=False,
        optimizer="adam",
        map_lr_X=map_lr_X,
        map_lr_theta=map_lr_theta,
        X_init=X_init_alt,
        theta_init=theta_init_alt,
    )

    alt_small = shrink_result(alt_res)
    alt_small["k_idx"] = k_idx
    alt_small["k_time"] = tau
    alt_small["cp_idx"] = cp_idx

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    print(f"[alt_worker] finished k={k_idx}, loglik={alt_small.get('loglik', -np.inf):.2f}", flush=True)
    return alt_small
