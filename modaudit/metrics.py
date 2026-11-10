"""Moderation metrics: AUC, review load at a target recall, and Jigsaw's bias metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import IDENTITIES


def auc(y, s) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def threshold_at_recall(y, s, recall: float = 0.90) -> float:
    """Highest threshold that still catches at least `recall` of the toxic comments."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos = np.sort(s[y == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("No positive examples to set a threshold.")
    k = max(1, math.ceil(recall * len(pos)))
    return float(pos[k - 1])


def at_threshold(y, s, thr: float) -> dict:
    """Recall, precision, false-positive rate and review load (share of all comments flagged)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    flag = s >= thr
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {
        "recall": tp / pos if pos else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "fpr": fp / neg if neg else float("nan"),
        "review_rate": float(flag.mean()),
    }


def subgroup_aucs(df: pd.DataFrame, s, label: str = "toxic", groups=IDENTITIES, min_size: int = 20) -> pd.DataFrame:
    """Jigsaw's three per-group AUCs.

    subgroup AUC: toxic vs non-toxic comments inside the group.
    BPSN (background positive, subgroup negative): low means non-toxic comments that mention
        the group score higher than toxic comments that do not, i.e. false positives on the group.
    BNSP (background negative, subgroup positive): low means toxic comments that mention the
        group score lower than non-toxic ones that do not, i.e. missed abuse aimed at the group.
    """
    y = df[label].to_numpy()
    s = np.asarray(s, dtype=float)
    rows = []
    for g in groups:
        sub = df[g].to_numpy().astype(bool)
        if sub.sum() < min_size:
            continue
        bpsn = (~sub & (y == 1)) | (sub & (y == 0))
        bnsp = (~sub & (y == 0)) | (sub & (y == 1))
        rows.append({
            "group": g,
            "n": int(sub.sum()),
            "subgroup_auc": auc(y[sub], s[sub]),
            "bpsn_auc": auc(y[bpsn], s[bpsn]),
            "bnsp_auc": auc(y[bnsp], s[bnsp]),
        })
    return pd.DataFrame(rows)


def power_mean(values, p: float = -5.0) -> float:
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan")
    return float(np.power(np.mean(np.power(v, p)), 1.0 / p))


def jigsaw_final_score(overall_auc: float, table: pd.DataFrame, p: float = -5.0, w: float = 0.25) -> float:
    """Jigsaw's competition score: overall AUC blended with power means of the three bias AUCs.

    The p = -5 power mean is pulled toward the worst group, so one badly treated group drags it down.
    """
    if table.empty:
        return float("nan")
    parts = [power_mean(table[c], p) for c in ("subgroup_auc", "bpsn_auc", "bnsp_auc")]
    return float(w * overall_auc + w * sum(parts))


def identity_fpr(df: pd.DataFrame, s, thr: float, label: str = "toxic") -> dict:
    """False-positive rate on non-toxic comments that mention an identity vs those that do not."""
    y = df[label].to_numpy()
    flag = np.asarray(s, dtype=float) >= thr
    ident = df["any_identity"].to_numpy().astype(bool)
    neg_id = (y == 0) & ident
    neg_bg = (y == 0) & ~ident
    fpr_id = float(flag[neg_id].mean()) if neg_id.any() else float("nan")
    fpr_bg = float(flag[neg_bg].mean()) if neg_bg.any() else float("nan")
    return {"fpr_identity": fpr_id, "fpr_background": fpr_bg, "ratio": fpr_id / fpr_bg if fpr_bg else float("nan"),
            "n_identity_nontoxic": int(neg_id.sum())}


def bootstrap(stat, n: int, b: int = 1000, seed: int = 0, alpha: float = 0.10) -> tuple[float, float]:
    """Percentile bootstrap interval (default 90%) for stat(idx) over n resampled rows."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        v = stat(rng.integers(0, n, n))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def fpr_reduction_ci(flag_before, flag_after, b: int = 1000, seed: int = 0) -> dict:
    """Relative drop in false-positive rate on the same non-toxic rows, with a paired bootstrap interval."""
    before = np.asarray(flag_before, dtype=float)
    after = np.asarray(flag_after, dtype=float)

    def rel(idx):
        base = before[idx].mean()
        return (base - after[idx].mean()) / base if base > 0 else float("nan")

    point = rel(np.arange(len(before)))
    lo, hi = bootstrap(rel, len(before), b=b, seed=seed)
    return {"reduction": point, "lo": lo, "hi": hi}
