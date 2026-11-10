"""Fine-tune DistilBERT on a sample of the training data. Add --mitigate for the bias-fixed version."""
import argparse
import time

import pandas as pd

import _setup  # noqa: F401
from modaudit import data, distilbert, metrics, mitigation

p = argparse.ArgumentParser()
p.add_argument("--train-n", type=int, default=150_000)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--grad-accum", type=int, default=2, help="batches per optimizer step (effective batch = batch-size x this)")
p.add_argument("--max-len", type=int, default=128)
p.add_argument("--mitigate", action="store_true", help="train with reweighting + neutral identity sentences")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

name = "distilbert_mitigated" if a.mitigate else "distilbert"
train = data.sample(data.load("train"), a.train_n, a.seed)
weights = None
if a.mitigate:
    train = pd.concat([train, mitigation.neutral_identity_sentences(seed=a.seed)], ignore_index=True)
    weights = mitigation.bias_weights(train)
val, ev = data.load("val"), data.load("eval")

print(f"Device: {distilbert.device()}. Training {name} on {len(train):,} rows ...", flush=True)
t0 = time.time()
model, tok = distilbert.train(train["comment_text"], train["toxic"], weights, max_len=a.max_len,
                              epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
                              grad_accum=a.grad_accum)
print(f"  trained in {(time.time() - t0) / 60:.1f} min. Scoring val and eval ...", flush=True)
distilbert.save(model, tok, f"runs/{name}")
sv = distilbert.score(model, tok, val["comment_text"], max_len=a.max_len)
se = distilbert.score(model, tok, ev["comment_text"], max_len=a.max_len)
data.save_preds(name, "val", val["id"], sv)
data.save_preds(name, "eval", ev["id"], se)
thr = metrics.threshold_at_recall(val["toxic"], sv, 0.90)
m = metrics.at_threshold(ev["toxic"], se, thr)
f = metrics.identity_fpr(ev, se, thr)
print(f"  AUC {metrics.auc(ev['toxic'], se):.3f}  recall {m['recall']:.1%}  review rate {m['review_rate']:.1%}  "
      f"FPR identity {f['fpr_identity']:.1%} vs other {f['fpr_background']:.1%}  total {(time.time() - t0) / 60:.1f} min")
