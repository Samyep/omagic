from __future__ import annotations

import numpy as np
import pandas as pd


def _build_segments(cp_idx, T):
    cp_idx = sorted(int(i) for i in cp_idx if 0 < i < T)
    segments = []
    start = 0
    for c in cp_idx:
        segments.append((start, c))
        start = c
    segments.append((start, T))
    return segments


def _jaccard_segment(a, b):
    s1, e1 = a
    s2, e2 = b
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    inter = max(0, inter_end - inter_start)
    union = (e1 - s1) + (e2 - s2) - inter
    if union == 0:
        return 0.0
    return inter / union


def covering_metric(true_cp_idx, pred_cp_idx, T):
    G = _build_segments(true_cp_idx, T)
    S = _build_segments(pred_cp_idx, T)
    if not G or not S:
        return 0.0
    total = 0.0
    for A in G:
        best = 0.0
        for B in S:
            j = _jaccard_segment(A, B)
            if j > best:
                best = j
        total += (A[1] - A[0]) * best
    return total / T


def _assign_detections(true_cp_idx, det_idx, tol_idx=5):
    true_cp_idx = np.array(sorted(true_cp_idx), dtype=int)
    det_idx = np.array(det_idx, dtype=int)
    n_true = len(true_cp_idx)
    assigned = np.full(len(det_idx), -1, dtype=int)
    if n_true == 0 or len(det_idx) == 0:
        return assigned
    for k, idx in enumerate(det_idx):
        pos = np.searchsorted(true_cp_idx, idx)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < n_true:
            candidates.append(pos)
        best_true = -1
        best_dist = None
        for c in candidates:
            d = abs(idx - true_cp_idx[c])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_true = c
        if best_dist is not None and best_dist <= tol_idx:
            assigned[k] = best_true
    return assigned


def compute_metrics(true_cp_idx, true_cp_time, T, det_idx, det_time, est_idx, est_time, tol_idx=5):
    true_cp_idx = np.array(sorted(true_cp_idx), dtype=int)
    true_cp_time = np.array(true_cp_time, dtype=float)
    det_idx = np.array(det_idx, dtype=int)
    det_time = np.array(det_time, dtype=float)
    est_idx = np.array(est_idx, dtype=int)
    est_time = np.array(est_time, dtype=float)

    assert det_idx.shape == det_time.shape == est_idx.shape == est_time.shape
    assert true_cp_idx.shape == true_cp_time.shape

    n_true = len(true_cp_idx)
    assigned = _assign_detections(true_cp_idx, det_idx, tol_idx=tol_idx)

    fp = int(np.sum(assigned == -1))
    detected_true_mask = np.zeros(n_true, dtype=bool)
    for a in assigned:
        if a >= 0:
            detected_true_mask[a] = True
    tp = int(np.sum(detected_true_mask))
    fn = n_true - tp

    denom = fp + (T - n_true)
    FAR = fp / denom if denom > 0 else 0.0
    MAR = fn / n_true if n_true > 0 else 0.0

    delays, mae_vals = [], []
    for j in range(n_true):
        idx_dets = np.where(assigned == j)[0]
        if idx_dets.size == 0:
            continue
        k = idx_dets[np.argmin(det_time[idx_dets])]
        C = true_cp_time[j]
        Cdet = det_time[k]
        Cest = est_time[k]
        delays.append(Cdet - C)
        mae_vals.append(abs(Cest - C))

    EDD = float(np.mean(delays)) if delays else float("nan")
    MAE = float(np.mean(mae_vals)) if mae_vals else float("nan")

    pred_cp_idx = sorted(set(int(i) for i in est_idx))
    Cover = covering_metric(true_cp_idx, pred_cp_idx, T)

    return {
        "FAR": FAR,
        "MAR": MAR,
        "EDD": EDD,
        "MAE": MAE,
        "Cover": Cover,
        "FP": fp,
        "TP": tp,
        "FN": fn,
        "n_true": n_true,
        "n_det": len(det_idx),
    }


def extract_detections_from_results(csv_path, threshold=20.0):
    df_res = pd.read_csv(csv_path)
    mask = df_res["cp_idx_local"].notna()
    if threshold is not None:
        mask &= df_res["T"] > threshold
    df_det = df_res.loc[mask].copy()
    if df_det.empty:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    det_idx_global = (df_det["slice_start"] + df_det["win_end_local"]).to_numpy(dtype=int)
    det_time = df_det["t"].to_numpy(dtype=float)
    est_idx_global = (df_det["slice_start"] + df_det["cp_idx_local"]).to_numpy(dtype=int)
    est_time = df_det["k_time"].to_numpy(dtype=float)

    order = np.argsort(det_time)
    return det_idx_global[order], det_time[order], est_idx_global[order], est_time[order]


def eval_one_run(prefix, threshold=20.0, tol_idx=5):
    gt = np.load(f"{prefix}_true_cps.npz")
    true_cp_idx = gt["cp_idx"]
    true_cp_time = gt["cp_time"]
    T = int(gt["n_obs"])

    det_idx, det_time, est_idx, est_time = extract_detections_from_results(
        f"{prefix}_results.csv", threshold=threshold
    )
    return compute_metrics(true_cp_idx, true_cp_time, T, det_idx, det_time, est_idx, est_time, tol_idx=tol_idx)
