#!/usr/bin/env python3
"""
RK-GLR runner that reuses the shared pipeline parent, forcing child_impl="rk_glr".
Parent simulates data, chunks the series, and spawns the RK GLR child via parent.py.
Child writes <prefix>_glr_results.csv; ground truth saved to <prefix>_true_cps.npz.
"""

import argparse
import multiprocessing as mp
import sys

from cpdet.metrics import eval_one_run
from cpdet.pipeline import parent_main
from cpdet.pipeline.child_rk_glr import child_rk_glr_main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="RK GLR change-point detection runner.")
    ap.add_argument("--child", action="store_true", help="Internal flag: run child mode.")
    ap.add_argument("--data_path", type=str, default="", help="(child) path to pickled data df.")
    ap.add_argument("--start", type=int, default=0, help="inclusive start index of observation slice.")
    ap.add_argument("--stop", type=int, default=1000, help="exclusive stop index of observation slice.")
    ap.add_argument("--seed", type=int, default=20250720, help="random seed for simulation.")
    ap.add_argument("--chi2", type=float, default=20.0, help="recorded threshold (not used in GLR logic).")
    ap.add_argument("--prefix", type=str, default="rk_glr_run", help="prefix for output files.")
    ap.add_argument("--clean_every", type=int, default=100, help="unused here (parity).")
    ap.add_argument("--window", type=int, default=60, help="sliding window length.")
    ap.add_argument("--warm_in", type=str, default="", help="unused (parity).")
    ap.add_argument("--warm_out", type=str, default="", help="unused (parity).")
    ap.add_argument("--chunk", type=int, default=110, help="points per child (span).")
    ap.add_argument("--stride", type=int, default=50, help="child start step.")
    ap.add_argument("--model", type=str, default="lotka_volterra", help="Model to use (registry key).")
    ap.add_argument("--scan", type=int, default=10, help="Trailing points per window to scan (child).")
    return ap.parse_args(argv)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args(sys.argv[1:])
    if args.child:
        child_rk_glr_main(args)
    else:
        # Force the pipeline parent to launch the RK GLR child
        args.child_impl = "rk_glr"
        parent_main(args)
        try:
            metrics = eval_one_run(args.prefix, threshold=args.chi2, tol_idx=10)
            print("\n[eval] Metrics for this run:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"[eval] WARNING: could not compute metrics: {e}")
