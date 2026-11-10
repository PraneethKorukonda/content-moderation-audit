import numpy as np

from modaudit import llm


def test_parsers():
    assert llm.parse_score("87") == 0.87
    assert llm.parse_score("Score: 5") == 0.05
    assert llm.parse_score("250") == 1.0
    assert np.isnan(llm.parse_score("no idea"))
    assert np.isnan(llm.parse_score("")) and np.isnan(llm.parse_score("REFUSED"))
    assert llm.parse_score("On a 0-100 scale this is 85") == 0.85
    assert llm.parse_category("Obfuscation") == "obfuscation"
    assert llm.parse_category("label_noise.") == "label_noise"
    assert llm.parse_category("weird") == "other"


def test_moderate_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache.jsonl")
    calls = []

    def fake_call(client, model, prompt, max_tokens, kind="mod"):
        calls.append(prompt)
        return "90" if "idiot" in prompt else "3"

    monkeypatch.setattr(llm, "_client", lambda: object())
    monkeypatch.setattr(llm, "_call", fake_call)
    s = llm.moderate([1, 2], ["you idiot", "nice weather"], workers=2)
    assert s == [0.9, 0.03]
    s2 = llm.moderate([1, 2], ["you idiot", "nice weather"], workers=2)
    assert s2 == s and len(calls) == 2  # second run reads the cache

    cats = llm.label_failures([{"id": 1, "comment_text": "y0u 1d10t", "toxic": 1, "score": 0.1, "threshold": 0.5}])
    assert cats == ["other"]
