"""Disguise insult words in 1,000 toxic eval comments and measure how much recall each model loses."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _setup  # noqa: F401
from modaudit import baseline, data, metrics, robustness

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1000)
p.add_argument("--words", type=int, default=200, help="how many top toxic words to disguise")
p.add_argument("--llm-n", type=int, default=0, help="also test Claude on this many comments per disguise")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

ev = data.load("eval")
plain = joblib.load("runs/tfidf.joblib")
words = baseline.top_toxic_words(plain, k=a.words)
pool, variants = robustness.build_attack_set(ev, words, n=a.n, seed=a.seed)
print(f"Attack set: {len(pool):,} toxic comments containing one of the top {len(words)} toxic words.")
print("  example:", variants["clean"][0][:120].replace("\n", " "))
print("  leet:   ", variants["leetspeak"][0][:120].replace("\n", " "))

scorers = {}
for name in ("tfidf", "tfidf_mitigated"):
    if Path(f"runs/{name}.joblib").exists():
        m = joblib.load(f"runs/{name}.joblib")
        scorers[name] = lambda texts, m=m: baseline.score(m, texts)
for name in ("distilbert", "distilbert_mitigated"):
    if Path(f"runs/{name}").exists():
        from modaudit import distilbert
        mdl, tok = distilbert.load(f"runs/{name}")
        scorers[name] = lambda texts, mdl=mdl, tok=tok: distilbert.score(mdl, tok, texts)

rows = []
for name, fn in scorers.items():
    val_pred = data.load_preds(name, "val")
    val = data.load("val").merge(val_pred, on="id")
    thr = metrics.threshold_at_recall(val["toxic"], val["score"], 0.90)
    for kind, texts in variants.items():
        rows.append({"model": name, "variant": kind, "n": len(texts),
                     "recall": robustness.recall(fn(texts), thr)})
    print(f"  {name} done")

if a.llm_n:
    from modaudit import llm
    llm_eval = data.load_preds("claude_haiku", "eval")
    if llm_eval is None:
        raise SystemExit("Run scripts/04_llm.py first so Claude has a threshold.")
    j = ev.merge(llm_eval, on="id")
    thr = metrics.threshold_at_recall(j["toxic"], j["score"], 0.90)
    ids = pool["id"].tolist()[: a.llm_n]
    for kind, texts in variants.items():
        s = llm.moderate(ids, texts[: a.llm_n], tag=kind)
        rows.append({"model": "claude_haiku", "variant": kind, "n": len(ids), "recall": robustness.recall(s, thr)})
    print("  claude_haiku done")

out = pd.DataFrame(rows)
out.to_csv("data/robustness.csv", index=False)
pd.DataFrame({k: v[:20] for k, v in variants.items()}).to_csv("data/robustness_examples.csv", index=False)
print(out.pivot(index="model", columns="variant", values="recall").round(3).to_string())
