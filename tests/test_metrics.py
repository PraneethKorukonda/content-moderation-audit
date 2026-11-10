import numpy as np
import pandas as pd
import pytest

from modaudit import metrics
from modaudit.data import IDENTITIES


def test_threshold_hits_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = y * 0.5 + rng.random(5000)
    for r in (0.5, 0.8, 0.9, 0.99):
        t = metrics.threshold_at_recall(y, s, r)
        got = metrics.at_threshold(y, s, t)["recall"]
        assert got >= r
        assert got < r + 0.01  # tightest threshold that reaches the target


def test_at_threshold_counts():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = metrics.at_threshold(y, s, 0.5)
    assert m["recall"] == 0.5 and m["precision"] == 0.5
    assert m["fpr"] == pytest.approx(1 / 3) and m["review_rate"] == pytest.approx(0.4)


def _frame(y, group):
    df = pd.DataFrame({"toxic": y})
    for g in IDENTITIES:
        df[g] = False
    df["muslim"] = group
    df["any_identity"] = group
    return df


def test_bpsn_catches_identity_false_positives():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.integers(0, 2, n)
    group = rng.random(n) < 0.3
    fair = y + rng.normal(0, 0.6, n)
    biased = fair + 1.0 * group  # every comment naming the group scores higher
    tf = metrics.subgroup_aucs(_frame(y, group), fair).set_index("group")
    tb = metrics.subgroup_aucs(_frame(y, group), biased).set_index("group")
    assert tb.loc["muslim", "bpsn_auc"] < tf.loc["muslim", "bpsn_auc"] - 0.1
    assert tb.loc["muslim", "bnsp_auc"] > tf.loc["muslim", "bnsp_auc"]
    assert tb.loc["muslim", "subgroup_auc"] == pytest.approx(tf.loc["muslim", "subgroup_auc"], abs=1e-9)


def test_identity_fpr_and_reduction_ci():
    df = _frame(np.zeros(1000, dtype=int), np.r_[np.ones(500, bool), np.zeros(500, bool)])
    s = np.r_[np.full(100, 0.9), np.full(400, 0.1), np.full(25, 0.9), np.full(475, 0.1)]
    f = metrics.identity_fpr(df, s, 0.5)
    assert f["fpr_identity"] == pytest.approx(0.2) and f["fpr_background"] == pytest.approx(0.05)
    before = np.r_[np.ones(100), np.zeros(400)].astype(bool)
    after = np.r_[np.ones(50), np.zeros(450)].astype(bool)
    r = metrics.fpr_reduction_ci(before, after, b=300)
    assert r["reduction"] == pytest.approx(0.5)
    assert r["lo"] < 0.5 < r["hi"]


def test_power_mean_pulled_to_worst():
    assert metrics.power_mean([0.9, 0.9, 0.5]) < np.mean([0.9, 0.9, 0.5])
