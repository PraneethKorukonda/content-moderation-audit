"""Train the TF-IDF baseline twice: as is, and with the bias fixes (reweighting + neutral identity sentences)."""
import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

import _setup  # noqa: F401
from modaudit import baseline, data, metrics, mitigation

p = argparse.ArgumentParser()
p.add_argument("--max-train", type=int, default=None, help="cap training rows (default: all)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

train = data.sample(data.load("train"), a.max_train, a.seed)
val, ev = data.load("val"), data.load("eval")
Path("runs").mkdir(exist_ok=True)

aug = mitigation.neutral_identity_sentences(seed=a.seed)
train_mit = pd.concat([train, aug], ignore_index=True)
configs = {
    "tfidf": (train, None),
    "tfidf_mitigated": (train_mit, mitigation.bias_weights(train_mit)),
}
for name, (tr, w) in configs.items():
    t0 = time.time()
    print(f"Training {name} on {len(tr):,} rows ...", flush=True)
    model = baseline.fit(baseline.make_model(), tr["comment_text"], tr["toxic"], sample_weight=w)
    joblib.dump(model, f"runs/{name}.joblib")
    sv = baseline.score(model, val["comment_text"])
    se = baseline.score(model, ev["comment_text"])
    data.save_preds(name, "val", val["id"], sv)
    data.save_preds(name, "eval", ev["id"], se)
    thr = metrics.threshold_at_recall(val["toxic"], sv, 0.90)
    m = metrics.at_threshold(ev["toxic"], se, thr)
    f = metrics.identity_fpr(ev, se, thr)
    print(f"  done in {time.time() - t0:.0f}s  AUC {metrics.auc(ev['toxic'], se):.3f}  recall {m['recall']:.1%}  "
          f"review rate {m['review_rate']:.1%}  FPR identity {f['fpr_identity']:.1%} vs other {f['fpr_background']:.1%}")
