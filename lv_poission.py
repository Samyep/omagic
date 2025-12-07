#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import sys

from cpdet.metrics import eval_one_run
from cpdet.pipeline import child_main, parent_main


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Change-point detection runner (Lotka-Volterra Poisson by default).")
    ap.add_argument("--child", action="store_true", help="Internal flag: run in child mode.")
    ap.add_argument("--data_path", type=str, default="", help="(child) path to pickled data df.")
    ap.add_argument("--start", type=int, default=0, help="inclusive start index of observation slice.")
    ap.add_argument("--stop", type=int, default=1000, help="exclusive stop index of observation slice.")
    ap.add_argument("--seed", type=int, default=20250720, help="random seed for simulation / MAGI.")
    ap.add_argument("--chi2", type=float, default=20.0, help="Detection T-statistic threshold.")
    ap.add_argument("--prefix", type=str, default="lv_poisson_run", help="prefix for output files.")
    ap.add_argument("--clean_every", type=int, default=100, help="child deep-clean interval (windows).")
    ap.add_argument("--window", type=int, default=60, help="sliding window length.")
    ap.add_argument("--warm_in", type=str, default="", help="(child) path to warm-in pickle.")
    ap.add_argument("--warm_out", type=str, default="", help="(child) path to warm-out pickle.")
    ap.add_argument("--chunk", type=int, default=110, help="points per child (span).")
    ap.add_argument("--stride", type=int, default=50, help="child start step.")
    ap.add_argument("--model", type=str, default="lotka_volterra", help="Model to use (e.g., lotka_volterra, lorenz, seird).")
    return ap.parse_args(argv)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args(sys.argv[1:])
    if args.child:
        child_main(args)
    else:
        parent_main(args)
        try:
            metrics = eval_one_run(args.prefix, threshold=args.chi2, tol_idx=5)
            print("\n[eval] Metrics for this run:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"[eval] WARNING: could not compute metrics: {e}")
