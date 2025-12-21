from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from cpdet.models import get_model
from cpdet.utils.seeds import set_seeds


def parent_main(args):
    t_start_parent = time.perf_counter()

    set_seeds(args.seed)
    np.random.seed(args.seed)

    model = get_model(args.model)

    tmpdir = tempfile.mkdtemp(prefix="cpdet_lv_")
    data_path = os.path.join(tmpdir, "data.pkl")
    warm_in = os.path.join(tmpdir, "warm_in.pkl")
    warm_out = os.path.join(tmpdir, "warm_out.pkl")

    print(f"[parent] temp dir: {tmpdir}", flush=True)

    df, true_cp_idx, true_cp_times = model.simulate(seed=args.seed)

    with open(data_path, "wb") as f:
        pickle.dump(df, f)

    gt_path = f"{args.prefix}_true_cps.npz"
    np.savez(gt_path, cp_idx=true_cp_idx, cp_time=true_cp_times, n_obs=len(df))
    print(f"[parent] saved ground truth CP info to {gt_path}", flush=True)

    first_full_done = False
    last_theta = None
    last_X = None
    last_win_start_global = None
    last_win_end_global = None

    start_idx = args.start
    stop_idx = args.stop
    total = stop_idx - start_idx
    if total <= 0:
        print("[parent] empty slice.", flush=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)

    if args.chunk < args.window:
        print(f"[parent] ERROR: --chunk ({args.chunk}) must be >= --window ({args.window}).", flush=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(2)

    # Select which child implementation to spawn; always use packaged pipeline scripts
    if getattr(args, "child_impl", "magi") == "rk_glr":
        try:
            import cpdet.pipeline.child_rk_glr as mod

            script_path = os.path.abspath(mod.__file__)
        except Exception as e:
            print(f"[parent] ERROR: could not locate child_rk_glr: {e}", flush=True)
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(2)
    else:
        try:
            import cpdet.pipeline.child as mod

            script_path = os.path.abspath(mod.__file__)
        except Exception as e:
            print(f"[parent] ERROR: could not locate child.py: {e}", flush=True)
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(2)

    S = start_idx
    while S < stop_idx:
        E = min(S + args.chunk, stop_idx)

        with open(warm_in, "wb") as f:
            pickle.dump(
                {
                    "FIRST_FULL_WINDOW_DONE": first_full_done,
                    "LAST_NULL_INIT": {"theta": last_theta, "X": last_X},
                    "LAST_WIN_START_GLOBAL": last_win_start_global,
                    "LAST_WIN_END_GLOBAL": last_win_end_global,
                    "seed": args.seed,
                },
                f,
            )

        cmd = [
            sys.executable,
            script_path,
            "--child",
            "--data_path",
            data_path,
            "--start",
            str(S),
            "--stop",
            str(E),
            "--seed",
            str(args.seed),
            "--chi2",
            str(args.chi2),
            "--prefix",
            args.prefix,
            "--clean_every",
            str(args.clean_every),
            "--window",
            str(args.window),
            "--warm_in",
            warm_in,
            "--warm_out",
            warm_out,
            "--model",
            args.model,
        ]

        # Only RK GLR child understands --scan; harmless to add when selected
        if getattr(args, "child_impl", "magi") == "rk_glr":
            cmd.extend(["--scan", str(getattr(args, "scan", 10))])
        print(f"[parent] spawn child for [{S}:{E})", flush=True)
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[parent] child failed for [{S}:{E})", flush=True)
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(ret.returncode)

        if os.path.exists(warm_out):
            try:
                with open(warm_out, "rb") as f:
                    w = pickle.load(f)
                first_full_done = bool(w.get("FIRST_FULL_WINDOW_DONE", first_full_done))
                ni = w.get("LAST_NULL_INIT", {})
                last_theta = ni.get("theta", last_theta)
                last_X = ni.get("X", last_X)
                last_win_start_global = w.get("LAST_WIN_START_GLOBAL", last_win_start_global)
                last_win_end_global = w.get("LAST_WIN_END_GLOBAL", last_win_end_global)
            except Exception as e:
                print(f"[parent] warning: could not read warm_out: {e}", flush=True)

        if S + args.stride >= E and E < stop_idx:
            S = E
        else:
            S += args.stride

    print("[parent] all chunks complete.", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    t_end_parent = time.perf_counter()
    print(f"\n[parent] TOTAL SCRIPT DURATION: {t_end_parent - t_start_parent:.2f} seconds\n", flush=True)
