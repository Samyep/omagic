from __future__ import annotations

from typing import Dict


def shrink_result(res: dict | None) -> Dict:
    """Drop heavy sampler fields from MAGI outputs to ease IPC and storage."""
    if res is None:
        return {}
    small, drop = {}, {"X_samps", "sigma_sqs_samps", "thetas_samps", "kernel_results", "sample_results", "X_hat"}
    for k, v in res.items():
        if k in drop:
            continue
        small[k] = v
    return small
