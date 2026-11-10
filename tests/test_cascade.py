import numpy as np

from modaudit import cascade


def test_route_picks_closest_to_threshold():
    m = cascade.route_mask([0.1, 0.48, 0.9, 0.55, 0.3], 0.5, 0.4)
    assert m.tolist() == [False, True, False, True, False]


def test_endpoints_match_single_models():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    cheap = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    llm = np.clip(y * 0.6 + rng.random(500) * 0.4, 0, 1)
    ident = rng.random(500) < 0.3
    r0 = cascade.evaluate(y, ident, cheap, llm, 0.5, 0.5, 0.0)
    r1 = cascade.evaluate(y, ident, cheap, llm, 0.5, 0.5, 1.0, cost_per_call=0.001)
    assert r0["recall"] == ((cheap >= 0.5) & (y == 1)).sum() / (y == 1).sum()
    assert r1["recall"] == ((llm >= 0.5) & (y == 1)).sum() / (y == 1).sum()
    assert r0["claude_calls_per_1k"] == 0 and r1["claude_calls_per_1k"] == 1000
    assert abs(r1["cost_per_1k"] - 1.0) < 1e-9


def test_better_llm_helps_on_uncertain_band():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    cheap = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    llm = y * 0.9 + rng.random(2000) * 0.1  # near-perfect second opinion
    ident = np.zeros(2000, bool)
    acc = lambda f: 1 - abs(cascade.decide(cheap, llm, 0.5, 0.5, f)[0].astype(int) - y).mean()
    assert acc(0.3) > acc(0.0)
