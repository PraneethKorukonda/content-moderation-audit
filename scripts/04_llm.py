"""Score a random sample of the eval set with Claude Haiku as a zero-shot moderator."""
import argparse

import numpy as np

import _setup  # noqa: F401
from modaudit import data, llm, metrics

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=2000)
p.add_argument("--model", default=llm.DEFAULT_MODEL)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

ev = data.sample(data.load("eval"), a.n, a.seed)
print(f"Scoring {len(ev):,} comments with {a.model} ...", flush=True)
scores = np.array(llm.moderate(ev["id"].tolist(), ev["comment_text"].tolist(), model=a.model, workers=a.workers))
bad = np.isnan(scores)
print(f"  unparseable, refused or failed: {bad.sum()} of {len(scores)}")
if bad.mean() > 0.05:
    print("  More than 5% missing. Check data/llm_cache.jsonl for what Claude returned.")
data.save_preds("claude_haiku", "eval", ev["id"][~bad], scores[~bad])
print(f"  AUC on this sample {metrics.auc(ev['toxic'][~bad], scores[~bad]):.3f}")
