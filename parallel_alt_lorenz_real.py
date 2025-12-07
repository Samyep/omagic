#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import pickle
import random
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp


# -----------------------------------------
# Shared utility
# -----------------------------------------

def shrink_result(res: dict) -> dict:
    """Drop heavy fields from MAGI results."""
    if res is None:
        return {}
    small, drop = {}, {"X_samps", "sigma_sqs_samps", "thetas_samps",
                       "kernel_results", "sample_results"}
    for k, v in res.items():
        if k in drop:
            continue
        small[k] = v
    return small


def set_seeds(seed: int):
    """Set seeds for determinism."""
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
        pass


# -----------------------------------------
# Data Simulation (Lorenz with changing rho)
# -----------------------------------------

def simulate_df():
    """Simulate Lorenz data with changing rho(t)."""
    sigma = 10.0
    rho1, rho2, rho3 = 28.0, 35.0, 20.0
    t_cp1, t_cp2 = 7.0, 14.0
    beta = 8.0 / 3.0

    t0, t_max, dt = 0.0, 20.0, 0.01
    t = np.arange(t0, t_max + dt, dt)
    n_steps = t.size

    def rho_time(tt):
        if isinstance(tt, np.ndarray):
            rho_t = np.full_like(tt, rho1, dtype=float)
            rho_t[tt > t_cp1] = rho2
            rho_t[tt > t_cp2] = rho3
            return rho_t
        else:
            if tt > t_cp2:
                return rho3
            if tt > t_cp1:
                return rho2
            return rho1

    def lorenz(state, tt):
        x, y, z = state
        current_rho = rho_time(tt)
        dx = sigma * (y - x)
        dy = x * (current_rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])

    # RK4 integration
    states = np.zeros((n_steps, 3))
    states[0] = np.array([1.0, 1.0, 1.0])
    for i in range(n_steps - 1):
        s = states[i]
        ti = t[i]
        k1 = dt * lorenz(s, ti)
        k2 = dt * lorenz(s + 0.5 * k1, ti + 0.5 * dt)
        k3 = dt * lorenz(s + 0.5 * k2, ti + 0.5 * dt)
        k4 = dt * lorenz(s + k3, ti + dt)
        states[i + 1] = s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    noise_std = 0.05
    noisy = states * np.random.normal(1, noise_std, size=states.shape)

    # Decimate ::10
    df = pd.DataFrame({
        "t": t[::10],
        "rho_t": rho_time(t[::10]),
        "x_clean": states[::10, 0],
        "y_clean": states[::10, 1],
        "z_clean": states[::10, 2],
        "x_noisy": noisy[::10, 0],
        "y_noisy": noisy[::10, 1],
        "z_noisy": noisy[::10, 2],
    })
    return df


# -----------------------------------------
# Parallel alternative worker
# -----------------------------------------

def alt_map_worker(args):
    """
    Worker to run one alternative (one tau / candidate cp index)
    for the Lorenz model.

    args = (
        ts_obs,
        X_obs,
        theta_null,     # (sigma, rho, beta)
        X_init_null,
        tau,
        k_idx,
        cp_idx,
        map_iters,
        map_patience,
        map_tol,
        map_lr_X,
        map_lr_theta
    )
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

    from python_magi.magi import MAGI

    (ts_obs,
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
     map_lr_theta) = args

    print(f"[alt_worker] starting k={k_idx}, tau={tau:.3f}", flush=True)

    # theta_null = (sigma, rho, beta)
    sigma0 = float(theta_null[0])
    rho0   = float(theta_null[1])
    beta0  = float(theta_null[2])

    tau_tf = tf.constant(tau, dtype=tf.float64)

    def f_vec_alt_rho_tauvar(t, X, theta):
        """
        Alternative Lorenz with piecewise rho(t):
        theta = (sigma, beta, rho_L, rho_R).
        """
        sigma = theta[0]
        beta  = theta[1]
        rho_L = theta[2]
        rho_R = theta[3]

        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        rho_t = tf.where(t[:, 0] < tau_tf, rho_L, rho_R)

        dx = sigma * (y - x)
        dy = x * (rho_t - z) - y
        dz = x * y - beta * z
        return tf.stack([dx, dy, dz], axis=1)

    magi_alt = MAGI(D_thetas=4, ts_obs=ts_obs, X_obs=X_obs,
                    bandsize=None, f_vec=f_vec_alt_rho_tauvar)
    magi_alt.initial_fit(discretization=0, verbose=False)

    if X_init_null is not None:
        X_init_alt = X_init_null
    else:
        X_init_alt = magi_alt.Xhat_init

    # Alt theta: (sigma, beta, rho_L, rho_R)
    theta_init_alt = np.array([sigma0, beta0, rho0, rho0], dtype=np.float64)
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


# -----------------------------------------
# Main Lorenz CP detection (Strategy B: jump + refill)
# -----------------------------------------

def run_lorenz_detection(df: pd.DataFrame,
                         seed: int = 20250720,
                         chi2_threshold: float = 20,
                         window: int = 60,
                         clean_every: int = 5,
                         prefix: str = "run_lorenz"):
    """
    Run CP detection on Lorenz using one long segment (no chunking),
    with MCMC-once + MAP-to-MAP + jump-and-refill.
    """
    import csv
    import tensorflow as tf

    # ---------- TF + env setup ----------
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(False)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    set_seeds(seed)
    tf.random.set_seed(seed)

    from python_magi.magi import MAGI

    SAVE_DIR = "cache"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Config
    WINDOW = int(window)
    SCAN = 10                      # last 10 points in window
    MIN_WINDOW_FOR_DETECTION = min(WINDOW, SCAN)
    D_null = 3                     # (sigma, rho, beta)
    D_alt = 4                      # (sigma, beta, rho_L, rho_R)

    def deep_cleanup(tag=""):
        print(f"[main deep-clean] {tag}", flush=True)
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

    def write_csv_row(csv_path: str, row_dict: dict):
        fieldnames = list(row_dict.keys())
        header_needed = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if header_needed:
                w.writeheader()
            w.writerow(row_dict)

    # Null Lorenz vector field
    def f_vec_const(t, X, theta):
        """
        Lorenz model with constant (sigma, rho, beta).
        theta = (sigma, rho, beta).
        """
        sigma = theta[0]
        rho   = theta[1]
        beta  = theta[2]
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return tf.stack([dx, dy, dz], axis=1)

    # Global state within this single run
    FIRST_FULL_WINDOW_DONE = False
    LAST_NULL_INIT = {"theta": None, "X": None}

    # Process pool for alternative windows
    num_workers = min(10, max(1, (os.cpu_count() or 2) // 2))
    ctx = mp.get_context("spawn")
    alt_pool = ctx.Pool(processes=num_workers, maxtasksperchild=5)

    batch_df = df.reset_index(drop=True)
    n = len(batch_df)
    print(f"[main] Lorenz detection on full slice 0:{n} (len={n})", flush=True)

    csv_path = "results_lorenz_jump_refill.csv"

    # --- index-based jump-and-refill loop ---
    window_start = 0
    i = 0
    num_windows = 0

    while i < n:
        current_len = i - window_start + 1

        # Enforce maximum window length
        if current_len > WINDOW:
            window_start = i - WINDOW + 1
            current_len = WINDOW

        # Gating logic:
        # - at start of stream (window_start == 0), require full WINDOW
        # - after jump (window_start > 0), allow refill with length >= MIN_WINDOW_FOR_DETECTION
        if window_start == 0:
            if current_len < WINDOW:
                i += 1
                continue
        else:
            if current_len < MIN_WINDOW_FOR_DETECTION:
                i += 1
                continue

        # Build window data
        win_df = batch_df.iloc[window_start : i + 1]
        ts_obs = win_df["t"].to_numpy(dtype=np.float64).reshape(-1, 1)
        X_obs  = win_df[["x_noisy", "y_noisy", "z_noisy"]].to_numpy(dtype=np.float64)

        current_t = float(batch_df["t"].iloc[i])

        # Null MAGI
        magi_null = MAGI(D_thetas=D_null, ts_obs=ts_obs, X_obs=X_obs,
                         bandsize=None, f_vec=f_vec_const)
        magi_null.initial_fit(discretization=0, verbose=False)

        # ---- Warm-start logic: MCMC-once + MAP-to-MAP ----
        if not FIRST_FULL_WINDOW_DONE:
            print(f"[main] MCMC warm start for initial window [{window_start},{i}]", flush=True)
            mcmc_res = magi_null.predict(
                num_results=1000,
                num_burnin_steps=1000,
                verbose=True,
            )
            X_init_null     = mcmc_res["X_samps"].mean(axis=0)
            theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
            theta_init_null = np.clip(theta_init_null, 1e-6, None)
            del mcmc_res
            FIRST_FULL_WINDOW_DONE = True
        else:
            X_prev     = LAST_NULL_INIT.get("X")
            theta_prev = LAST_NULL_INIT.get("theta")
            if X_prev is not None and theta_prev is not None:
                prev_len = X_prev.shape[0]
                if prev_len == current_len:
                    # Sliding warm start
                    print(f"[main] sliding warm start ({prev_len}→{current_len}).", flush=True)
                    X_init_null = X_prev
                    theta_init_null = theta_prev
                elif prev_len == current_len - 1:
                    # Refill warm start (add one time step)
                    print(f"[main] refill warm start ({prev_len}→{current_len}).", flush=True)
                    pad = X_prev[-1:, :]
                    X_init_null = np.concatenate([X_prev, pad], axis=0)
                    theta_init_null = theta_prev
                else:
                    # Shape mismatch → re-run MCMC
                    print(f"[main] WARN shape mismatch (prev_len={prev_len}, cur_len={current_len}), MCMC again.", flush=True)
                    mcmc_res = magi_null.predict(
                        num_results=1000,
                        num_burnin_steps=1000,
                        verbose=True,
                    )
                    X_init_null     = mcmc_res["X_samps"].mean(axis=0)
                    theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
                    theta_init_null = np.clip(theta_init_null, 1e-6, None)
                    del mcmc_res
            else:
                # No stored MAP → MCMC again
                print("[main] WARN LAST_NULL_INIT empty, MCMC again.", flush=True)
                mcmc_res = magi_null.predict(
                    num_results=1000,
                    num_burnin_steps=1000,
                    verbose=True,
                )
                X_init_null     = mcmc_res["X_samps"].mean(axis=0)
                theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
                theta_init_null = np.clip(theta_init_null, 1e-6, None)
                del mcmc_res

        # ---- Null MAP ----
        t0 = time.perf_counter()
        print(f"[main] starting null MAP at t={current_t:.3f}, len={current_len}, "
              f"win=[{window_start},{i}]", flush=True)

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
        print(f"[main] finished null MAP in {dt_s:.1f}s", flush=True)

        if null_res.get("X_hat") is not None and null_res.get("theta_hat") is not None:
            LAST_NULL_INIT["X"]     = null_res["X_hat"]
            LAST_NULL_INIT["theta"] = null_res["theta_hat"]

        null_small = shrink_result(null_res)
        del magi_null, null_res

        # Per-window cleanup
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

        # ---- Parallel alternatives on last SCAN points ----
        best_alt = {
            "loglik": -np.inf,
            "k_idx": None,
            "theta_hat": None,
            "k_time": None,
            "cp_idx": None,
        }

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
            # k_idx: 1 = last point in window, scan_len = earliest
            k_idx = i - idx_local + 1

            jobs.append((
                ts_obs,
                X_obs,
                theta_null,
                LAST_NULL_INIT.get("X"),
                tau,
                k_idx,
                idx_local,       # cp_idx
                100000,          # map_iters
                200,             # map_patience
                1e-12,           # map_tol
                1e-4,            # map_lr_X
                1e-4,            # map_lr_theta
            ))

        alt_results = alt_pool.map(alt_map_worker, jobs)

        for alt_small in alt_results:
            if alt_small.get("loglik", -np.inf) > best_alt.get("loglik", -np.inf):
                best_alt = alt_small

        # ---- GLRT ----
        T = -2 * (null_small.get("loglik", -np.inf) - best_alt.get("loglik", -np.inf))

        # Infer rho: null theta = (sigma, rho, beta)
        inferred_rho = np.nan
        null_rho = np.nan
        try:
            null_rho = float(LAST_NULL_INIT["theta"][1])
        except Exception:
            pass

        if (best_alt.get("theta_hat") is not None and
                best_alt.get("k_time") is not None and
                np.isfinite(T) and T > chi2_threshold):
            try:
                # alt theta = (sigma, beta, rho_L, rho_R)
                inferred_rho = float(best_alt["theta_hat"][3])  # rho_R
            except Exception:
                inferred_rho = null_rho
        else:
            inferred_rho = null_rho

        # ---- Save row ----
        row_out = {
            "t": float(current_t),
            "T": float(T),
            "alt_loglik": float(best_alt.get("loglik", -np.inf)),
            "null_loglik": float(null_small.get("loglik", -np.inf)),
            "k_idx": best_alt.get("k_idx"),
            "k_time": best_alt.get("k_time"),
            "first_alt_tau": float(first_alt_tau),
            "prefix": prefix,
            "null_sigma": float(LAST_NULL_INIT["theta"][0]) if LAST_NULL_INIT["theta"] is not None else np.nan,
            "null_rho": null_rho,
            "null_beta": float(LAST_NULL_INIT["theta"][2]) if LAST_NULL_INIT["theta"] is not None else np.nan,
            "inferred_rho": inferred_rho,
            "sec_per_window": dt_s,
            "cp_idx_local": best_alt.get("cp_idx"),
            "win_start_local": window_start,
            "win_end_local": i,
        }

        if best_alt.get("theta_hat") is not None:
            try:
                row_out["alt_sigma"] = float(best_alt["theta_hat"][0])
                row_out["alt_beta"]  = float(best_alt["theta_hat"][1])
                row_out["alt_rho_L"] = float(best_alt["theta_hat"][2])
                row_out["alt_rho_R"] = float(best_alt["theta_hat"][3])
            except Exception:
                pass

        write_csv_row(csv_path, row_out)

        print(f"[main window] t={current_t:7.3f}  T={T:8.3f}  "
              f"alt_k={best_alt.get('k_idx')}  "
              f"alt_ll={best_alt.get('loglik', -np.inf):.2f}  "
              f"null_ll={null_small.get('loglik', -np.inf):.2f}  "
              f"win=[{window_start},{i}]",
              flush=True)

        # Save pickle per window (optional)
        win_idx_global = i
        with open(os.path.join(SAVE_DIR, f"{prefix}_window_{win_idx_global:05d}.pkl"), "wb") as f:
            pickle.dump({"null": null_small, "alt": best_alt}, f)

        num_windows += 1
        if (num_windows % max(1, clean_every)) == 0:
            deep_cleanup(f"global_window_idx {win_idx_global}")

        # ---- Jump or slide ----
        if (np.isfinite(T) and T > chi2_threshold and
                best_alt.get("cp_idx") is not None):
            cp_idx_local = int(best_alt["cp_idx"])
            print(f"[main] DETECTION: T={T:.2f}, jumping to cp_idx_local={cp_idx_local}", flush=True)

            window_start = cp_idx_local
            new_i = cp_idx_local + MIN_WINDOW_FOR_DETECTION - 1
            new_i = min(new_i, n - 1)
            i = new_i
        else:
            i += 1

    # End while
    alt_pool.close()
    alt_pool.join()

    print("[main] Lorenz detection done.", flush=True)


# -----------------------------------------
# CLI Entrypoint
# -----------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Lorenz CP detection (single chunk, jump+refill).")
    ap.add_argument("--seed", type=int, default=20250720)
    ap.add_argument("--chi2", type=float, default=20,
                    help="GLRT chi-square threshold (df=1 -> 3.84).")
    ap.add_argument("--window", type=int, default=60,
                    help="window length (e.g., 40).")
    ap.add_argument("--clean_every", type=int, default=5,
                    help="how many windows between deep cleanups.")
    ap.add_argument("--prefix", type=str, default="run_lorenz")
    return ap.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    df = simulate_df()
    run_lorenz_detection(
        df=df,
        seed=args.seed,
        chi2_threshold=args.chi2,
        window=args.window,
        clean_every=args.clean_every,
        prefix=args.prefix,
    )
