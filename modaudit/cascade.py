"""Cascade: a cheap model decides the easy comments and sends only the uncertain ones to Claude."""
from __future__ import annotations

import numpy as np

from . import metrics


def route_mask(cheap, cheap_thr: float, frac: float) -> np.ndarray:
    """Pick the `frac` share of comments whose cheap score sits closest to the cheap model's threshold."""
    cheap = np.asarray(cheap, dtype=float)
    k = int(round(frac * len(cheap)))
    mask = np.zeros(len(cheap), dtype=bool)
    if k > 0:
        mask[np.argsort(np.abs(cheap - cheap_thr), kind="stable")[:k]] = True
    return mask


def decide(cheap, llm, cheap_thr: float, llm_thr: float, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Final flag per comment: Claude's call for routed comments, the cheap model's call for the rest."""
    routed = route_mask(cheap, cheap_thr, frac)
    flag = np.where(routed, np.asarray(llm) >= llm_thr, np.asarray(cheap) >= cheap_thr)
    return flag, routed


def evaluate(y, ident, cheap, llm, cheap_thr: float, llm_thr: float, frac: float,
             cost_per_call: float | None = None) -> dict:
    y = np.asarray(y)
    ident = np.asarray(ident, dtype=bool)
    flag, routed = decide(cheap, llm, cheap_thr, llm_thr, frac)
    st = metrics.at_threshold(y, flag.astype(float), 0.5)
    neg_id, neg_bg = (y == 0) & ident, (y == 0) & ~ident
    out = {
        "routed_share": float(routed.mean()),
        **st,
        "fpr_identity": float(flag[neg_id].mean()) if neg_id.any() else float("nan"),
        "fpr_background": float(flag[neg_bg].mean()) if neg_bg.any() else float("nan"),
        "claude_calls_per_1k": 1000 * float(routed.mean()),
    }
    out["cost_per_1k"] = out["claude_calls_per_1k"] * cost_per_call if cost_per_call is not None else float("nan")
    return out
