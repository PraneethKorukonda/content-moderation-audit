import numpy as np

import fake_data
from modaudit import baseline, data, mitigation, robustness


def test_load_and_split(tmp_path):
    p = tmp_path / "train.csv"
    fake_data.make(1000).to_csv(p, index=False)
    df = data.load_jigsaw(p)
    assert set(df.toxic.unique()) <= {0, 1}
    assert df.any_identity.dtype == bool
    parts = data.split(df, seed=0)
    ids = [set(v.id) for v in parts.values()]
    assert sum(len(s) for s in ids) == len(df)
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_bias_weights_mean_one_and_order():
    import pandas as pd

    df = pd.DataFrame({"toxic": [0, 0, 1, 1], "any_identity": [False, True, False, True]})
    w = mitigation.bias_weights(df)
    assert np.isclose(w.mean(), 1.0)
    assert w[1] == w.max()  # non-toxic identity comment gets the most weight
    assert w[0] == w.min()


def test_neutral_sentences():
    aug = mitigation.neutral_identity_sentences(repeats=5)
    assert (aug.toxic == 0).all() and aug.any_identity.all()
    assert len(aug) == 5 * len(mitigation.IDENTITY_TERMS)


def test_perturbations_change_only_targets():
    t, n = robustness.perturb("You are an Idiot and a nice person", {"idiot"}, "leetspeak")
    assert n == 1 and "1d10t" in t and "nice person" in t
    assert robustness.dotted("dumb") == "d.u.m.b"
    assert robustness.stretched("idiot") == "iiidiot"
    assert robustness.swapped("stupid") == "stpuid"


def test_baseline_learns_and_finds_insults(tmp_path):
    p = tmp_path / "train.csv"
    fake_data.make(3000).to_csv(p, index=False)
    df = data.load_jigsaw(p)
    m = baseline.fit(baseline.make_model(), df.comment_text, df.toxic)
    top = baseline.top_toxic_words(m, k=6)
    assert len(set(top) & set(fake_data.INSULTS)) >= 4
    assert baseline.score(m, ["you idiot stupid moron"])[0] > baseline.score(m, ["the council voted on the budget"])[0]


def test_loads_all_data_csv_layout(tmp_path):
    p = tmp_path / "all_data.csv"
    fake_data.make(500).rename(columns={"target": "toxicity"}).assign(split="train").to_csv(p, index=False)
    df = data.load_jigsaw(p)
    assert len(df) == 500 and "target" in df and set(df.toxic.unique()) <= {0, 1}
