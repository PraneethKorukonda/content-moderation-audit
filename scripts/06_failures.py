"""Ask Claude why the best model's most confident mistakes happened, then count the patterns."""
import argparse

import pandas as pd

import _setup  # noqa: F401
from modaudit import data, llm, metrics

p = argparse.ArgumentParser()
p.add_argument("--model", default=None, help="which model's mistakes to review (default: distilbert if trained)")
p.add_argument("--n", type=int, default=150, help="mistakes of each kind (missed toxic, false flags)")
a = p.parse_args()

name = a.model or ("distilbert" if data.load_preds("distilbert", "eval") is not None else "tfidf")
val = data.load("val").merge(data.load_preds(name, "val"), on="id")
thr = metrics.threshold_at_recall(val["toxic"], val["score"], 0.90)
ev = data.load("eval").merge(data.load_preds(name, "eval"), on="id")

missed = ev[(ev.toxic == 1) & (ev.score < thr)].nsmallest(a.n, "score")
false_flags = ev[(ev.toxic == 0) & (ev.score >= thr)].nlargest(a.n, "score")
rows = pd.concat([missed.assign(error="missed_toxic"), false_flags.assign(error="false_flag")], ignore_index=True)
rows["threshold"] = thr
print(f"Reviewing {len(rows)} confident mistakes from {name} with Claude ...", flush=True)
rows["category"] = llm.label_failures(rows.to_dict("records"))
rows[["id", "error", "category", "score", "toxic", "any_identity", "comment_text"]].to_csv("data/failures.csv", index=False)
print(pd.crosstab(rows["category"], rows["error"], margins=True).to_string())
