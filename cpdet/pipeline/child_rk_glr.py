#!/usr/bin/env python3
"""
Deterministic RK-based GLR child benchmark using the same sliding-window layout as the MAGI child.
- Reads a pickled dataframe (same as child.py input) and emits <prefix>_glr_results.csv.
- Uses model.prepare_window, model.null_f_vec, model.alt_f_vec.
- Variance sigma2_hat estimated from the first 50 observed points (all dims).
- Fits null/alt via L-BFGS-B; scans trailing --scan points per window.
- No changes to existing pipeline or pickle schema.
"""

import argparse
import os
import pickle
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cpdet.models import get_model
from cpdet.utils.seeds import set_seeds
from cpdet.pipeline.alt_worker_rk import alt_map_worker_rk


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


def write_csv_row(csv_path: str, row_dict: dict):
    fieldnames = sorted(list(row_dict.keys()))
    header_needed = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=fieldnames)
        if header_needed:
            w.writeheader()
        w.writerow(row_dict)


def parse_args(argv=None):
    """Mirror child.py CLI so it can be swapped in easily."""
    ap = argparse.ArgumentParser(description="Deterministic RK GLR child benchmark (sliding window).")
    ap.add_argument("--child", action="store_true", help="unused (for parity with parent launch)")
    ap.add_argument("--data_path", type=str, required=True, help="Pickled dataframe (same as child input).")
    ap.add_argument("--start", type=int, default=0, help="inclusive start index")
    ap.add_argument("--stop", type=int, default=-1, help="exclusive stop index; -1 uses full length")
    ap.add_argument("--seed", type=int, default=20251209)
    ap.add_argument("--chi2", type=float, default=20.0, help="(not used for logic; recorded)")
    ap.add_argument("--prefix", type=str, default="rk_glr_run", help="output prefix (CSV)")
    ap.add_argument("--window", type=int, default=60, help="sliding window length")
    ap.add_argument("--scan", type=int, default=10, help="trailing points in window to scan for tau")
    ap.add_argument("--model", type=str, default="lotka_volterra", help="model registry key")
    # parity-only args (ignored in logic)
    ap.add_argument("--clean_every", type=int, default=100)
    ap.add_argument("--warm_in", type=str, default="")
    ap.add_argument("--warm_out", type=str, default="")
    ap.add_argument("--chunk", type=int, default=110)
    ap.add_argument("--stride", type=int, default=50)
    return ap.parse_args(argv)


def child_rk_glr_main(args):
    set_seeds(args.seed)

    with open(args.data_path, "rb") as f:
        df = pickle.load(f)

    stop = None if args.stop < 0 else args.stop
    batch_df = df.iloc[args.start:stop].reset_index(drop=True)
    n = len(batch_df)
    if n == 0:
        print("[rk_glr] empty slice after start/stop", file=sys.stderr)
        sys.exit(1)

    model = get_model(args.model)
    f_vec_null = model.null_f_vec()
    WINDOW = int(args.window)
    SCAN = int(args.scan)
    MIN_WINDOW_FOR_DETECTION = min(WINDOW, SCAN)
    EPS = 1e-8

    # shared variance from first 50 obs
    try:
        init_df = batch_df.iloc[: min(50, n)]
        _, X_init_obs = model.prepare_window(init_df, eps=EPS)
        sigma2_hat = float(np.mean((X_init_obs - X_init_obs.mean(axis=0)) ** 2))
    except Exception:
        sigma2_hat = 1.0
    sigma2_hat = max(sigma2_hat, 1e-6)

    csv_path = f"{args.prefix}_glr_results.csv"
    print(f"[rk_glr] slice {args.start}:{args.stop if args.stop>=0 else 'end'} (len={n}) prefix={args.prefix}")

    num_workers = min(10, max(1, (os.cpu_count() or 2) // 2))
    ctx = mp.get_context("spawn")
    alt_pool = ctx.Pool(processes=num_workers, maxtasksperchild=5)

    window_start = 0
    i = 0
    num_windows = 0
    theta_null_prev = None

    while i < n:
        current_len = i - window_start + 1
        if current_len > WINDOW:
            window_start = i - WINDOW + 1
            current_len = WINDOW

        if window_start == 0 and current_len < WINDOW:
            i += 1
            continue
        if window_start > 0 and current_len < MIN_WINDOW_FOR_DETECTION:
            i += 1
            continue

        win_df = batch_df.iloc[window_start : i + 1]
        ts_obs, X_obs = model.prepare_window(win_df, eps=EPS)
        current_t = float(batch_df["t"].iloc[i])

        theta0_null = theta_null_prev if theta_null_prev is not None else np.ones(model.param_dim_null, dtype=np.float64)

        def nll_null_fn(theta_vec):
            theta_c = np.maximum(theta_vec, 1e-8)
            rhs = tf_fvec_to_numpy(f_vec_null, theta_c)
            traj = rk4(ts_obs[:, 0], X_obs[0], theta_c, rhs)
            return gaussian_nll(X_obs, traj, sigma2_hat)

        theta_null_hat, nll_null = fit_theta(nll_null_fn, theta0_null)
        theta_null_prev = theta_null_hat

        scan_len = min(SCAN, current_len)
        candidate_start = i - scan_len + 1
        candidate_indices = list(range(candidate_start, i + 1))
        first_alt_tau = float(batch_df["t"].iloc[candidate_indices[0]]) if candidate_indices else np.nan

        jobs = []
        for idx_local in candidate_indices:
            tau = float(batch_df["t"].iloc[idx_local])
            k_idx = i - idx_local + 1
            cp_idx = idx_local
            jobs.append((model.name, ts_obs, X_obs, theta_null_hat, tau, k_idx, cp_idx, sigma2_hat))

        best = {"nll": np.inf, "theta_hat": None, "k_idx": None, "k_time": None, "cp_idx": None}
        alt_results = alt_pool.map(alt_map_worker_rk, jobs) if jobs else []
        for alt_small in alt_results:
            if alt_small.get("nll", np.inf) < best.get("nll", np.inf):
                best = alt_small

        T_rk = 2.0 * (nll_null - best["nll"]) if np.isfinite(best.get("nll", np.nan)) else np.nan

        row_out = {
            "t": current_t,
            "T": T_rk,
            "T_rk": T_rk,
            "null_nll_rk": nll_null,
            "alt_nll_rk": best.get("nll"),
            "sigma2_hat_rk": sigma2_hat,
            "k_idx": best.get("k_idx"),
            "k_time": best.get("k_time"),
            "first_alt_tau": first_alt_tau,
            "slice_start": args.start,
            "slice_stop": args.stop if args.stop >= 0 else args.start + n,
            "prefix": args.prefix,
            "model": model.name,
            "cp_idx_local": best.get("cp_idx"),
            "win_start_local": window_start,
            "win_end_local": i,
            "chi2": args.chi2,
        }

        for idx, val in enumerate(theta_null_hat.flatten()):
            row_out[f"null_theta_{idx}"] = float(val)
        if best.get("theta_hat") is not None:
            for idx, val in enumerate(best["theta_hat"].flatten()):
                row_out[f"alt_theta_{idx}"] = float(val)

        write_csv_row(csv_path, row_out)

        num_windows += 1
        if num_windows % 50 == 0:
            print(f"[rk_glr] processed {num_windows} windows; last t={current_t:.3f}", flush=True)

        i += 1

    alt_pool.close()
    alt_pool.join()

    print(f"[rk_glr] done. wrote {csv_path}")


def main():
    args = parse_args()
    child_rk_glr_main(args)


if __name__ == "__main__":
    main()
