from __future__ import annotations

import csv
import gc
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from cpdet.models import get_model
from cpdet.pipeline.alt_worker import alt_map_worker
from cpdet.utils.results import shrink_result
from cpdet.utils.seeds import set_seeds


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


def child_main(child_args):
    _configure_tf_single_thread()

    import tensorflow as tf

    set_seeds(child_args.seed)
    tf.random.set_seed(child_args.seed)

    try:
        from python_magi.magi import MAGI
    except ImportError:
        print("[child] ERROR: python_magi.magi module not found.", flush=True)
        sys.exit(1)

    model = get_model(child_args.model)

    SAVE_DIR = "cache"
    os.makedirs(SAVE_DIR, exist_ok=True)

    WINDOW = int(child_args.window)
    SCAN = 10
    DETECTION_THRESHOLD = child_args.chi2
    MIN_WINDOW_FOR_DETECTION = min(WINDOW, SCAN)

    D_null = model.param_dim_null
    EPS = 1e-8

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
    prev_win_start_global = warm.get("LAST_WIN_START_GLOBAL", None)
    prev_win_end_global = warm.get("LAST_WIN_END_GLOBAL", None)

    num_workers = min(10, max(1, (os.cpu_count() or 2) // 2))
    ctx = mp.get_context("spawn")
    alt_pool = ctx.Pool(processes=num_workers, maxtasksperchild=5)

    batch_df = df.iloc[child_args.start : child_args.stop].reset_index(drop=True)
    n = len(batch_df)
    print(f"[child] slice {child_args.start}:{child_args.stop} (len={n})", flush=True)

    f_vec_null = model.null_f_vec()

    csv_path = f"{child_args.prefix}_results.csv"

    window_start = 0
    i = 0

    if prev_win_start_global is not None and prev_win_end_global is not None:
        local_start = prev_win_start_global - child_args.start
        local_end = prev_win_end_global - child_args.start

        if local_end >= 0 and local_start < n:
            window_start = max(0, local_start)
            i = max(window_start, min(local_end, n - 1))
            print(
                f"[child] continuing from previous window global [{prev_win_start_global},{prev_win_end_global}] -> "
                f"local [{window_start},{i}]",
                flush=True,
            )
        else:
            window_start = 0
            i = 0

    num_windows = 0
    last_win_start_global = None
    last_win_end_global = None

    while i < n:
        current_len = i - window_start + 1

        if current_len > WINDOW:
            window_start = i - WINDOW + 1
            current_len = WINDOW

        if window_start == 0:
            if current_len < WINDOW:
                i += 1
                continue
        else:
            if current_len < MIN_WINDOW_FOR_DETECTION:
                i += 1
                continue

        win_df = batch_df.iloc[window_start : i + 1]
        ts_obs, X_obs = model.prepare_window(win_df, eps=EPS)

        current_t = float(batch_df["t"].iloc[i])

        magi_null = MAGI(D_thetas=D_null, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec_null)
        magi_null.initial_fit(discretization=0, verbose=False)

        X_init_null = None
        theta_init_null = None

        if not FIRST_FULL_WINDOW_DONE:
            print(
                f"[child] calling magi_null.predict (MCMC) for initial warm start. Window indices {window_start}-{i}",
                flush=True,
            )
            mcmc_res = magi_null.predict(num_results=1000, num_burnin_steps=1000, verbose=True)
            X_init_null = mcmc_res["X_samps"].mean(axis=0)
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
                    print(
                        f"[child] Using previous MAP values as init for sliding window ({prev_len}→{current_len}).",
                        flush=True,
                    )
                    X_init_null = X_prev
                    theta_init_null = theta_prev

                elif prev_len == current_len - 1:
                    print(f"[child] Using refill warm start ({prev_len}→{current_len}).", flush=True)
                    pad = X_prev[-1:, :]
                    X_init_null = np.concatenate([X_prev, pad], axis=0)
                    theta_init_null = theta_prev

                else:
                    print(
                        f"[child] WARN: shape mismatch in warm start (prev_len={prev_len}, cur_len={current_len}). "
                        f"Falling back to MCMC.",
                        flush=True,
                    )
                    mcmc_res = magi_null.predict(num_results=1000, num_burnin_steps=1000, verbose=True)
                    X_init_null = mcmc_res["X_samps"].mean(axis=0)
                    theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
                    theta_init_null = np.clip(theta_init_null, 1e-6, None)
                    del mcmc_res
            else:
                print("[child] WARN: LAST_NULL_INIT empty; running MCMC.", flush=True)
                mcmc_res = magi_null.predict(num_results=1000, num_burnin_steps=1000, verbose=True)
                X_init_null = mcmc_res["X_samps"].mean(axis=0)
                theta_init_null = mcmc_res["thetas_samps"].mean(axis=0)
                theta_init_null = np.clip(theta_init_null, 1e-6, None)
                del mcmc_res

        t0 = time.perf_counter()
        print(
            f"[child] starting null MAP for window ending at t={current_t:.3f} (len={current_len}, start_idx={window_start})",
            flush=True,
        )

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

        best_alt = {"loglik": -np.inf, "k_idx": None, "theta_hat": None, "k_time": None, "cp_idx": None}
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

            jobs.append(
                (
                    model.name,
                    ts_obs,
                    X_obs,
                    theta_null,
                    LAST_NULL_INIT.get("X"),
                    tau,
                    k_idx,
                    idx_local,
                    100000,
                    200,
                    1e-12,
                    1e-4,
                    1e-4,
                )
            )

        alt_results = alt_pool.map(alt_map_worker, jobs)

        for alt_small in alt_results:
            if alt_small.get("loglik", -np.inf) > best_alt.get("loglik", -np.inf):
                best_alt = alt_small

        T_stat = -2 * (null_small.get("loglik", -np.inf) - best_alt.get("loglik", -np.inf))

        inferred_change = model.infer_change_value(LAST_NULL_INIT.get("theta"), best_alt, T_stat, DETECTION_THRESHOLD)

        row_out = {
            "t": float(current_t),
            "T": float(T_stat),
            "alt_loglik": float(best_alt.get("loglik", -np.inf)),
            "null_loglik": float(null_small.get("loglik", -np.inf)),
            "k_idx": best_alt.get("k_idx"),
            "k_time": best_alt.get("k_time"),
            "first_alt_tau": float(first_alt_tau),
            "slice_start": child_args.start,
            "slice_stop": child_args.stop,
            "prefix": child_args.prefix,
            "model": model.name,
            "inferred_change": inferred_change,
            "sec_per_window": dt_s,
            "cp_idx_local": best_alt.get("cp_idx"),
            "win_start_local": window_start,
            "win_end_local": i,
        }

        if LAST_NULL_INIT.get("theta") is not None:
            try:
                for idx, val in enumerate(np.asarray(LAST_NULL_INIT["theta"]).flatten()):
                    row_out[f"null_theta_{idx}"] = float(val)
            except Exception:
                pass

        if best_alt.get("theta_hat") is not None:
            try:
                for idx, val in enumerate(np.asarray(best_alt["theta_hat"]).flatten()):
                    row_out[f"alt_theta_{idx}"] = float(val)
            except Exception:
                pass

        write_csv_row(csv_path, row_out)

        print(
            f"[child window] t={current_t:7.3f}  T={T_stat:8.3f}  alt_k={best_alt.get('k_idx')}  "
            f"alt_ll={best_alt.get('loglik', -np.inf):.2f}  null_ll={null_small.get('loglik', -np.inf):.2f}  "
            f"win=[{window_start},{i}]",
            flush=True,
        )

        win_idx_global = child_args.start + i
        with open(os.path.join(SAVE_DIR, f"{child_args.prefix}_window_{win_idx_global:05d}.pkl"), "wb") as f:
            pickle.dump({"null": null_small, "alt": best_alt}, f)

        last_win_start_global = child_args.start + window_start
        last_win_end_global = child_args.start + i

        num_windows += 1
        if (num_windows % max(1, child_args.clean_every)) == 0:
            deep_cleanup(f"global_window_idx {win_idx_global}")

        if np.isfinite(T_stat) and T_stat > DETECTION_THRESHOLD and best_alt.get("cp_idx") is not None:
            cp_idx_local = int(best_alt["cp_idx"])
            print(f"[child] DETECTION: T={T_stat:.2f}, jumping to cp_idx_local={cp_idx_local}", flush=True)

            window_start = cp_idx_local
            new_i = cp_idx_local + MIN_WINDOW_FOR_DETECTION - 1
            new_i = min(new_i, n - 1)
            i = new_i
        else:
            i += 1

    alt_pool.close()
    alt_pool.join()

    with open(child_args.warm_out, "wb") as f:
        pickle.dump(
            {
                "FIRST_FULL_WINDOW_DONE": FIRST_FULL_WINDOW_DONE,
                "LAST_NULL_INIT": LAST_NULL_INIT,
                "LAST_WIN_START_GLOBAL": last_win_start_global,
                "LAST_WIN_END_GLOBAL": last_win_end_global,
            },
            f,
        )

    print("[child] done.", flush=True)
