"""Send only the most uncertain comments to Claude and measure recall, review load, bias and cost."""
import argparse

import pandas as pd

import _setup  # noqa: F401
from modaudit import cascade, data, llm, metrics

p = argparse.ArgumentParser()
p.add_argument("--cheap", default=None, help="first-pass model (default: distilbert if trained, else tfidf)")
p.add_argument("--fracs", default="0,0.05,0.1,0.2,0.3,0.5,1.0", help="shares of comments to send to Claude")
p.add_argument("--input-price", type=float, default=None, help="USD per million input tokens (from Anthropic's pricing page)")
p.add_argument("--output-price", type=float, default=None, help="USD per million output tokens")
a = p.parse_args()

cheap = a.cheap or ("distilbert" if data.load_preds("distilbert", "eval") is not None else "tfidf")
llm_preds = data.load_preds("claude_haiku", "eval")
if llm_preds is None:
    raise SystemExit("Run scripts/04_llm.py first.")

val = data.load("val").merge(data.load_preds(cheap, "val"), on="id")
cheap_thr = metrics.threshold_at_recall(val["toxic"], val["score"], 0.90)
j = (data.load("eval")[["id", "toxic", "any_identity"]]
     .merge(data.load_preds(cheap, "eval").rename(columns={"score": "cheap"}), on="id")
     .merge(llm_preds.rename(columns={"score": "llm"}), on="id"))
llm_thr = metrics.threshold_at_recall(j["toxic"], j["llm"], 0.90)

cost = None
usage = llm.mean_usage()
if usage and a.input_price is not None and a.output_price is not None:
    cost = (usage["in"] * a.input_price + usage["out"] * a.output_price) / 1e6
    print(f"Average Claude call: {usage['in']:.0f} input + {usage['out']:.1f} output tokens = ${cost:.6f}")
elif usage:
    print(f"Average Claude call: {usage['in']:.0f} input + {usage['out']:.1f} output tokens. "
          "Pass --input-price and --output-price to get cost.")

rows = []
for f in [float(x) for x in a.fracs.split(",")]:
    r = cascade.evaluate(j["toxic"], j["any_identity"], j["cheap"], j["llm"], cheap_thr, llm_thr, f, cost)
    rows.append({"cheap_model": cheap, **r})
out = pd.DataFrame(rows)
out.to_csv("data/cascade.csv", index=False)
print(f"Cascade on {len(j):,} comments. {cheap} threshold from validation, Claude threshold at 90% recall on this sample.")
cols = ["routed_share", "recall", "precision", "review_rate", "fpr_identity", "fpr_background", "claude_calls_per_1k", "cost_per_1k"]
print(out[cols].round(4).to_string(index=False))
