#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
import pickle
import subprocess
import shutil
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import random
import gc

# -----------------------------------------
# Shared utility
# -----------------------------------------

def shrink_result(res: dict) -> dict:
    """Drop heavy fields from MAGI results."""
    if res is None:
        return {}
    small, drop = {}, {"X_samps", "sigma_sqs_samps",
                       "thetas_samps", "kernel_results", "sample_results"}
    for k, v in res.items():
        if k in drop:
            continue
        # we don't need full X_hat for plotting (optional)
        if k == "X_hat":
            continue
        small[k] = v
    return small


def set_seeds(seed: int):
    """Set seeds for determinism in parent/child."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        # TF imported lazily in child/worker
        pass

# -----------------------------------------
# SEIRD Simulation (paper model)
# -----------------------------------------

def simulate_df():
    """
    Simulate the paper SEIRD model:

      dS/dt = - beta(t) * I S / N
      dE/dt =   beta(t) * I S / N - ve * E
      dI/dt =   ve * E - vi * I
      dD/dt =   vi * I * pd(t)

    with:
      beta(t) = 0.8 for t < t0, 0.1 for t >= t0, t0 ~ U[50,70]
      pd(t)   = 0.02 for t < t1, 0.05 for t >= t1, t1 ~ U[90,110]
      ve = vi = 0.1
      initial [S0,E0,I0,D0] = [1e6, 1000, 500, 50]
      R0 = 0, N = S0+E0+I0+D0+R0 constant
      observations: **twice daily** with 5% multiplicative noise
    """
    # Fixed parameters
    ve = 0.1
    vi = 0.1

    # Random CPs
    t0 = np.random.uniform(50.0, 70.0)   # beta CP
    t1 = np.random.uniform(90.0, 110.0)  # pd CP
    print(t0,t1)
    beta1, beta2 = 0.8, 0.1
    pd1, pd2     = 0.02, 0.05

    # Time grid for ODE integration
    t0_sim, t_max, dt = 0.0, 150.0, 0.1
    t = np.arange(t0_sim, t_max + dt, dt)
    n_steps = t.size

    # Initial states and total N
    S0, E0, I0, D0 = 1_000_000.0, 1000.0, 500.0, 50.0
    R0 = 0.0
    N_total = S0 + E0 + I0 + D0 + R0

    def beta_time(tt):
        if isinstance(tt, np.ndarray):
            out = np.full_like(tt, beta1, dtype=float)
            out[tt >= t0] = beta2
            return out
        else:
            return beta1 if tt < t0 else beta2

    def pd_time(tt):
        if isinstance(tt, np.ndarray):
            out = np.full_like(tt, pd1, dtype=float)
            out[tt >= t1] = pd2
            return out
        else:
            return pd1 if tt < t1 else pd2

    def seird_rhs(state, tt):
        S, E, I, D = state
        beta_t = beta_time(tt)
        pd_t   = pd_time(tt)
        dS = -beta_t * I * S / N_total
        dE =  beta_t * I * S / N_total - ve * E
        dI =  ve * E - vi * I
        dD =  vi * I * pd_t
        return np.array([dS, dE, dI, dD])

    # RK4 integration
    states = np.zeros((n_steps, 4))
    states[0] = np.array([S0, E0, I0, D0], dtype=float)
    for i in range(n_steps - 1):
        s = states[i]
        ti = t[i]
        k1 = dt * seird_rhs(s, ti)
        k2 = dt * seird_rhs(s + 0.5 * k1, ti + 0.5 * dt)
        k3 = dt * seird_rhs(s + 0.5 * k2, ti + 0.5 * dt)
        k4 = dt * seird_rhs(s + k3, ti + dt)
        states[i + 1] = s + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # 5% multiplicative noise
    noise_std = 0.05
    noisy = states * np.random.normal(1.0, scale=noise_std, size=states.shape)

    # Twice-daily observations: sampling interval = 0.5 day => step = 0.5 / dt
    obs_interval = 0.5  # days
    step_obs = int(round(obs_interval / dt))
    t_obs = t[::step_obs]
    states_obs = states[::step_obs]
    noisy_obs  = noisy[::step_obs]

    beta_true_obs = beta_time(t_obs)
    pd_true_obs   = pd_time(t_obs)

    df = pd.DataFrame({
        "t": t_obs,
        "S_clean": states_obs[:, 0],
        "E_clean": states_obs[:, 1],
        "I_clean": states_obs[:, 2],
        "D_clean": states_obs[:, 3],
        "S_noisy": noisy_obs[:, 0],
        "E_noisy": noisy_obs[:, 1],
        "I_noisy": noisy_obs[:, 2],
        "D_noisy": noisy_obs[:, 3],
        "beta_true": beta_true_obs,
        "pd_true": pd_true_obs,
    })
    return df

# -----------------------------------------
# Parent/Orchestrator – CHUNKED
# -----------------------------------------

def parent_main():
    ap = argparse.ArgumentParser(
        description="Paper SEIRD CP detection with jump-and-refill (chunked)."
    )
    ap.add_argument("--start", type=int, required=True, help="inclusive start index")
    ap.add_argument("--stop",  type=int, required=True, help="exclusive stop index")
    ap.add_argument("--seed",  type=int, default=2025720)
    ap.add_argument("--chi2",  type=float, default=20)
    ap.add_argument("--prefix", type=str, default="seird_jump_refill")
    ap.add_argument("--alt_param", type=str, choices=["beta", "pd"], default="beta")
    ap.add_argument("--window", type=int, default=60, help="sliding window length")
    ap.add_argument("--clean_every", type=int, default=50, help="deep-clean interval in #windows")
    # NEW: chunking for memory control
    ap.add_argument("--chunk", type=int, default=150,
                    help="points per child (span); must be >= window")
    ap.add_argument("--stride", type=int, default=100,
                    help="child start step; overlap = chunk - stride")
    args = ap.parse_args()

    t_start_parent = time.perf_counter()

    set_seeds(args.seed)

    tmpdir = tempfile.mkdtemp(prefix="cpdet_seird_")
    data_path = os.path.join(tmpdir, "data.pkl")
    warm_in  = os.path.join(tmpdir, "warm_in.pkl")
    warm_out = os.path.join(tmpdir, "warm_out.pkl")

    print(f"[parent] temp dir: {tmpdir}", flush=True)
    df = simulate_df()
    with open(data_path, "wb") as f:
        pickle.dump(df, f)

    first_full_done = False
    last_theta = None
    last_X     = None

    start_idx = args.start
    stop_idx  = args.stop
    total = stop_idx - start_idx
    if total <= 0:
        print("[parent] empty slice.", flush=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)

    if args.chunk < args.window:
        print(f"[parent] ERROR: --chunk ({args.chunk}) must be >= --window ({args.window}).", flush=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(2)

    # --- chunked loop over [start_idx, stop_idx) ---
    S = start_idx
    while S < stop_idx:
        E = min(S + args.chunk, stop_idx)

        # warm_in: carry over MCMC/MAP state and whether we've ever done the first full window
        with open(warm_in, "wb") as f:
            pickle.dump({
                "FIRST_FULL_WINDOW_DONE": first_full_done,
                "LAST_NULL_INIT": {"theta": last_theta, "X": last_X},
                "seed": args.seed
            }, f)

        cmd = [
            sys.executable, __file__, "--child",
            "--data_path", data_path,
            "--start", str(S),
            "--stop", str(E),
            "--seed", str(args.seed),
            "--chi2", str(args.chi2),
            "--prefix", args.prefix,
            "--alt_param", args.alt_param,
            "--clean_every", str(args.clean_every),
            "--window", str(args.window),
            "--warm_in", warm_in,
            "--warm_out", warm_out,
        ]
        print(f"[parent] spawn child for [{S}:{E})", flush=True)
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[parent] child failed for [{S}:{E})", flush=True)
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(ret.returncode)

        # Read back final warm state from this child
        if os.path.exists(warm_out):
            try:
                with open(warm_out, "rb") as f:
                    w = pickle.load(f)
                first_full_done = bool(w.get("FIRST_FULL_WINDOW_DONE", first_full_done))
                ni = w.get("LAST_NULL_INIT", {})
                last_theta = ni.get("theta", last_theta)
                last_X     = ni.get("X", last_X)
            except Exception as e:
                print(f"[parent] warning: could not read warm_out: {e}", flush=True)

        # Advance S by stride, with a little guard to avoid infinite loops
        if S + args.stride >= E and E < stop_idx:
            # if stride is too large for last chunk, just jump to E
            S = E
        else:
            S += args.stride

    print("[parent] all chunks complete.", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    t_end_parent = time.perf_counter()
    print(f"[parent] TOTAL DURATION: {t_end_parent - t_start_parent:.2f} seconds", flush=True)

# -----------------------------------------
# Alternative worker
# -----------------------------------------

def alt_map_worker(args):
    """
    Worker for one alternative tau (beta or pd change).

    Null params: theta_null = (beta, ve, vi, pd).
    For alt_param == "beta":
        theta_alt = (ve, vi, pd, beta_L, beta_R).
    For alt_param == "pd":
        theta_alt = (ve, vi, beta, pd_L, pd_R).
    """
    import os
    import gc
    import numpy as np

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

    try:
        from python_magi.magi import MAGI
    except ImportError:
        print("[alt_worker] ERROR: python_magi.magi module not found.", flush=True)
        return {
            "loglik": -np.inf,
            "k_idx": args[-5],   # slightly hacky, but just to fill
            "k_time": args[-7],
            "cp_idx": args[-4],
        }

    (ts_obs,
     X_obs,
     theta_null,
     X_init_null,
     tau,
     alt_param,
     k_idx,
     cp_idx,
     map_iters,
     map_patience,
     map_tol,
     map_lr_X,
     map_lr_theta) = args

    print(f"[alt_worker] starting k={k_idx}, tau={tau:.3f}", flush=True)

    beta0, ve0, vi0, pd0 = (
        float(theta_null[0]),
        float(theta_null[1]),
        float(theta_null[2]),
        float(theta_null[3]),
    )

    tau_tf = tf.constant(tau, dtype=tf.float64)

    # Recover approximate N from first row
    S0 = tf.exp(X_obs[0, 0])
    E0 = tf.exp(X_obs[0, 1])
    I0 = tf.exp(X_obs[0, 2])
    D0 = tf.exp(X_obs[0, 3])
    N_total = S0 + E0 + I0 + D0  # R0≈0 in simulation

    def f_vec_alt_beta_tauvar_local(t, X_log, theta):
        # theta = (ve, vi, pd, beta_L, beta_R)
        ve = theta[0]; vi = theta[1]; pd = theta[2]
        beta_L = theta[3]; beta_R = theta[4]
        s, e, i, d = X_log[:, 0], X_log[:, 1], X_log[:, 2], X_log[:, 3]
        beta_t = tf.where(t[:, 0] < tau_tf, beta_L, beta_R)
        ds = -beta_t * tf.exp(i) / N_total
        de =  beta_t * tf.exp(s + i - e) / N_total - ve
        di =  ve * tf.exp(e - i) - vi
        dd =  vi * pd * tf.exp(i - d)
        return tf.stack([ds, de, di, dd], axis=1)

    def f_vec_alt_pd_tauvar_local(t, X_log, theta):
        # theta = (ve, vi, beta, pd_L, pd_R)
        ve = theta[0]; vi = theta[1]; beta = theta[2]
        pd_L = theta[3]; pd_R = theta[4]
        s, e, i, d = X_log[:, 0], X_log[:, 1], X_log[:, 2], X_log[:, 3]
        pd_t = tf.where(t[:, 0] < tau_tf, pd_L, pd_R)
        ds = -beta * tf.exp(i) / N_total
        de =  beta * tf.exp(s + i - e) / N_total - ve
        di =  ve * tf.exp(e - i) - vi
        dd =  vi * pd_t * tf.exp(i - d)
        return tf.stack([ds, de, di, dd], axis=1)

    f_alt_fn = f_vec_alt_beta_tauvar_local if alt_param == "beta" else f_vec_alt_pd_tauvar_local

    magi_alt = MAGI(D_thetas=5, ts_obs=ts_obs, X_obs=X_obs,
                    bandsize=None, f_vec=f_alt_fn)
    magi_alt.initial_fit(discretization=0, verbose=False)

    if X_init_null is not None:
        X_init_alt = X_init_null
    else:
        X_init_alt = magi_alt.Xhat_init

    if alt_param == "beta":
        theta_init_alt = np.array([ve0, vi0, pd0, beta0, beta0], dtype=np.float64)
    else:
        theta_init_alt = np.array([ve0, vi0, beta0, pd0, pd0], dtype=np.float64)
    theta_init_alt = np.clip(theta_init_alt, 1e-6, None)

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
        theta_init=theta_init_alt
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

# -----------------------------------------
# Child – jump-and-refill loop
# -----------------------------------------

def child_main(child_args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import csv
    import numpy as np
    import tensorflow as tf

    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(False)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    try:
        from python_magi.magi import MAGI
    except ImportError:
        print("[child] ERROR: python_magi.magi module not found.", flush=True)
        sys.exit(1)

    set_seeds(child_args.seed)
    tf.random.set_seed(child_args.seed)

    SAVE_DIR = "cache"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Config ---
    WINDOW = int(child_args.window)
    SCAN   = 10
    DETECTION_THRESHOLD = child_args.chi2
    MIN_WINDOW_FOR_DETECTION = min(WINDOW, SCAN)

    D_null = 4   # (beta, ve, vi, pd)
    EPS    = 1e-8

    def deep_cleanup(tag=""):
        print(f"[child deep-clean] {tag}", flush=True)
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

    def write_csv_row(csv_path: str, row_dict: dict):
        fieldnames = sorted(list(row_dict.keys()))
        header_needed = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if header_needed:
                w.writeheader()
            w.writerow(row_dict)

    with open(child_args.data_path, "rb") as f:
        df = pickle.load(f)
    with open(child_args.warm_in, "rb") as f:
        warm = pickle.load(f)

    FIRST_FULL_WINDOW_DONE = bool(warm.get("FIRST_FULL_WINDOW_DONE", False))
    LAST_NULL_INIT = warm.get("LAST_NULL_INIT", {"theta": None, "X": None})

    # Process pool for alternative windows
    num_workers = min(10, max(1, (os.cpu_count() or 2) // 2))
    ctx = mp.get_context("spawn")
    alt_pool = ctx.Pool(processes=num_workers, maxtasksperchild=5)

    batch_df = df.iloc[child_args.start:child_args.stop].reset_index(drop=True)
    n = len(batch_df)
    print(f"[child] slice {child_args.start}:{child_args.stop} (len={n})", flush=True)

    ALT_PARAM = child_args.alt_param

    def f_vec_seird_const(t, X_log, theta):
        """
        Null model in log space for (S,E,I,D) with
        theta = (beta, ve, vi, pd).
        """
        beta = theta[0]; ve = theta[1]; vi = theta[2]; pd = theta[3]
        s, e, i, d = X_log[:, 0], X_log[:, 1], X_log[:, 2], X_log[:, 3]

        # approximate N from first time step (R0≈0)
        S0 = tf.exp(X_log[0, 0])
        E0 = tf.exp(X_log[0, 1])
        I0 = tf.exp(X_log[0, 2])
        D0 = tf.exp(X_log[0, 3])
        N_total = S0 + E0 + I0 + D0

        ds = -beta * tf.exp(i) / N_total
        de =  beta * tf.exp(s + i - e) / N_total - ve
        di =  ve * tf.exp(e - i) - vi
        dd =  vi * pd * tf.exp(i - d)
        return tf.stack([ds, de, di, dd], axis=1)

    csv_path = "results_seird_jump_refill_new.csv"

    # --- jump-and-refill over indices ---

    window_start = 0
    i = 0
    num_windows = 0

    while i < n:
        current_len = i - window_start + 1

        if (not FIRST_FULL_WINDOW_DONE) and (window_start == 0) and (current_len < WINDOW):
            i += 1
            continue

        if current_len > WINDOW:
            window_start = i - WINDOW + 1
            current_len = WINDOW

        if current_len < MIN_WINDOW_FOR_DETECTION:
            i += 1
            continue

        win_df = batch_df.iloc[window_start : i + 1]
        ts_obs = win_df["t"].to_numpy(dtype=np.float64).reshape(-1, 1)
        vals   = win_df[["S_noisy", "E_noisy", "I_noisy", "D_noisy"]].to_numpy(dtype=np.float64)
        vals[vals <= 0] = EPS
        X_obs = np.log(vals)

        current_t = float(batch_df["t"].iloc[i])

        magi_null = MAGI(D_thetas=D_null, ts_obs=ts_obs, X_obs=X_obs,
                         bandsize=None, f_vec=f_vec_seird_const)
        magi_null.initial_fit(discretization=0, verbose=False)

        # --- warm start logic ---
        X_init_null = None
        theta_init_null = None

        if not FIRST_FULL_WINDOW_DONE:
            print(f"[child] calling magi_null.predict (MCMC) for initial warm start. "
                  f"Window indices {window_start}-{i}", flush=True)
            mcmc_res = magi_null.predict(
                num_results=1000,
                num_burnin_steps=1000,
                verbose=True
            )
            X_init_null     = mcmc_res["X_samps"].mean(axis=0)
            theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
            theta_init_null = np.clip(theta_init_null, 1e-6, None)
            del mcmc_res
            FIRST_FULL_WINDOW_DONE = True
        else:
            X_prev = LAST_NULL_INIT.get("X")
            theta_prev = LAST_NULL_INIT.get("theta")
            if X_prev is not None and theta_prev is not None:
                prev_len = X_prev.shape[0]
                if prev_len == current_len:
                    print(f"[child] Using previous MAP values as init ({prev_len}→{current_len}).", flush=True)
                    X_init_null     = X_prev
                    theta_init_null = theta_prev
                elif prev_len == current_len - 1:
                    print(f"[child] Using refill warm start ({prev_len}→{current_len}).", flush=True)
                    pad = X_prev[-1:, :]
                    X_init_null     = np.concatenate([X_prev, pad], axis=0)
                    theta_init_null = theta_prev
                else:
                    print(f"[child] WARN: shape mismatch warm start "
                          f"(prev_len={prev_len}, cur_len={current_len}). MCMC again.", flush=True)
                    mcmc_res = magi_null.predict(
                        num_results=1000,
                        num_burnin_steps=1000,
                        verbose=True
                    )
                    X_init_null     = mcmc_res["X_samps"].mean(axis=0)
                    theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
                    theta_init_null = np.clip(theta_init_null, 1e-6, None)
                    del mcmc_res
            else:
                print("[child] WARN: LAST_NULL_INIT empty; running MCMC.", flush=True)
                mcmc_res = magi_null.predict(
                    num_results=1000,
                    num_burnin_steps=1000,
                    verbose=True
                )
                X_init_null     = mcmc_res["X_samps"].mean(axis=0)
                theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
                theta_init_null = np.clip(theta_init_null, 1e-6, None)
                del mcmc_res

        # --- Null MAP ---
        t0 = time.perf_counter()
        print(f"[child] starting null MAP at t={current_t:.3f} (len={current_len}, start={window_start})",
              flush=True)
        null_res = magi_null.predict(
            method="map",
            map_iters=100000,
            map_patience=200,
            map_tol=1e-12,
            verbose=False,
            optimizer="adam",
            map_lr_X=1e-4,
            map_lr_theta=1e-4,
            X_init=X_init_null,
            theta_init=theta_init_null,
        )
        dt_s = time.perf_counter() - t0
        print(f"[child] finished null MAP in {dt_s:.1f}s", flush=True)

        if null_res.get("X_hat") is not None and null_res.get("theta_hat") is not None:
            LAST_NULL_INIT["X"] = null_res["X_hat"]
            LAST_NULL_INIT["theta"] = null_res["theta_hat"]

        null_small = shrink_result(null_res)
        del magi_null, null_res

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

        # -------------- PARALLEL ALT PART -----------------
        best_alt = {"loglik": -np.inf, "k_idx": None,
                    "theta_hat": None, "k_time": None, "cp_idx": None}
        theta_null = LAST_NULL_INIT.get("theta")
        if theta_null is None or len(theta_null) < D_null:
            theta_null = np.zeros((D_null,), dtype=np.float64)

        scan_len = min(SCAN, current_len)
        candidate_start = i - scan_len + 1
        candidate_indices = list(range(candidate_start, i + 1))

        jobs = []
        first_alt_tau = float(batch_df["t"].iloc[candidate_indices[0]]) if candidate_indices else np.nan

        for idx_local in candidate_indices:
            tau = float(batch_df["t"].iloc[idx_local])
            k_idx = i - idx_local + 1
            jobs.append((
                ts_obs,
                X_obs,
                theta_null,
                LAST_NULL_INIT.get("X"),
                tau,
                ALT_PARAM,
                k_idx,
                idx_local,
                100000,   # map_iters
                200,      # map_patience
                1e-12,    # map_tol
                1e-4,     # map_lr_X
                1e-4      # map_lr_theta
            ))

        alt_results = alt_pool.map(alt_map_worker, jobs)
        for alt_small in alt_results:
            if alt_small.get("loglik", -np.inf) > best_alt.get("loglik", -np.inf):
                best_alt = alt_small

        # -------------- END PARALLEL ALT PART -------------

        T = -2 * (null_small.get("loglik", -np.inf) -
                  best_alt.get("loglik", -np.inf))

        # Extract null params
        null_beta = np.nan
        null_ve   = np.nan
        null_vi   = np.nan
        null_pd   = np.nan
        try:
            null_beta = float(LAST_NULL_INIT["theta"][0])
            null_ve   = float(LAST_NULL_INIT["theta"][1])
            null_vi   = float(LAST_NULL_INIT["theta"][2])
            null_pd   = float(LAST_NULL_INIT["theta"][3])
        except Exception:
            pass

        # Alt params
        alt_beta_L = np.nan
        alt_beta_R = np.nan
        alt_pd_L   = np.nan
        alt_pd_R   = np.nan

        if best_alt.get("theta_hat") is not None:
            th = best_alt["theta_hat"]
            try:
                if ALT_PARAM == "beta":
                    # th = (ve, vi, pd, beta_L, beta_R)
                    alt_beta_L = float(th[3])
                    alt_beta_R = float(th[4])
                else:
                    # ALT_PARAM == "pd": th = (ve, vi, beta, pd_L, pd_R)
                    alt_pd_L = float(th[3])
                    alt_pd_R = float(th[4])
            except Exception:
                pass

        inferred_beta = null_beta
        inferred_pd   = null_pd
        if (best_alt.get("theta_hat") is not None and
            best_alt.get("k_time") is not None and
            np.isfinite(T) and T > DETECTION_THRESHOLD):

            if ALT_PARAM == "beta" and np.isfinite(alt_beta_R):
                inferred_beta = alt_beta_R
            if ALT_PARAM == "pd" and np.isfinite(alt_pd_R):
                inferred_pd = alt_pd_R

        # --- Write results row ---
        row_out = {
            "t": float(current_t),
            "T": float(T),
            "alt_loglik": float(best_alt.get("loglik", -np.inf)),
            "null_loglik": float(null_small.get("loglik", -np.inf)),
            "k_idx": best_alt.get("k_idx"),
            "k_time": best_alt.get("k_time"),
            "first_alt_tau": float(first_alt_tau),
            "slice_start": child_args.start,
            "slice_stop": child_args.stop,
            "prefix": child_args.prefix,
            "mode": ALT_PARAM,
            "null_beta": null_beta,
            "null_ve": null_ve,
            "null_vi": null_vi,
            "null_pd": null_pd,
            "inferred_beta": inferred_beta,
            "inferred_pd": inferred_pd,
            "sec_per_window": dt_s,
            "cp_idx_local": best_alt.get("cp_idx"),
            "win_start_local": window_start,
            "win_end_local": i,
        }

        if ALT_PARAM == "beta":
            row_out["alt_beta_L"] = alt_beta_L
            row_out["alt_beta_R"] = alt_beta_R
        else:
            row_out["alt_pd_L"] = alt_pd_L
            row_out["alt_pd_R"] = alt_pd_R

        write_csv_row(csv_path, row_out)

        print(f"[child window] t={current_t:7.3f}  T={T:8.3f}  "
              f"alt_k={best_alt.get('k_idx')}  "
              f"alt_ll={best_alt.get('loglik', -np.inf):.2f}  "
              f"null_ll={null_small.get('loglik', -np.inf):.2f}  "
              f"mode={ALT_PARAM}  win=[{window_start},{i}]", flush=True)

        win_idx_global = child_args.start + i
        with open(os.path.join(SAVE_DIR, f"{child_args.prefix}_window_{win_idx_global:05d}.pkl"), "wb") as f:
            pickle.dump({"null": null_small, "alt": best_alt}, f)

        num_windows += 1
        if (num_windows % max(1, child_args.clean_every)) == 0:
            deep_cleanup(f"global_window_idx {win_idx_global}")

        # --- Jump or slide ---
        if (np.isfinite(T) and T > DETECTION_THRESHOLD and
                best_alt.get("cp_idx") is not None):
            cp_idx_local = int(best_alt["cp_idx"])
            print(f"[child] DETECTION: T={T:.2f}, jumping to cp_idx_local={cp_idx_local}", flush=True)

            window_start = cp_idx_local
            new_i = cp_idx_local + MIN_WINDOW_FOR_DETECTION - 1
            new_i = min(new_i, n - 1)
            i = new_i
        else:
            i += 1

    alt_pool.close()
    alt_pool.join()

    with open(child_args.warm_out, "wb") as f:
        pickle.dump({
            "FIRST_FULL_WINDOW_DONE": FIRST_FULL_WINDOW_DONE,
            "LAST_NULL_INIT": LAST_NULL_INIT
        }, f)

    print("[child] done.", flush=True)

# -----------------------------------------
# Entrypoint
# -----------------------------------------

def parse_child_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--data_path", type=str)
    ap.add_argument("--start", type=int)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--seed", type=int,  default=2025720)
    ap.add_argument("--chi2", type=float, default=20.0)
    ap.add_argument("--prefix", type=str)
    ap.add_argument("--alt_param", type=str, choices=["beta", "pd"])
    ap.add_argument("--clean_every", type=int, default=50)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--warm_in", type=str)
    ap.add_argument("--warm_out", type=str)
    return ap.parse_args(argv)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    if "--child" in sys.argv:
        ca = parse_child_args(sys.argv[1:])
        child_main(ca)
    else:
        parent_main()
