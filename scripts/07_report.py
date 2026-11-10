"""Collect every model's predictions and write results.md (plus results.json)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _setup  # noqa: F401
from modaudit import data, metrics

RECALL = 0.90
MODELS = ["tfidf", "tfidf_mitigated", "distilbert", "distilbert_mitigated"]
NICE = {"tfidf": "TF-IDF + LR", "tfidf_mitigated": "TF-IDF + LR, bias-fixed", "distilbert": "DistilBERT",
        "distilbert_mitigated": "DistilBERT, bias-fixed", "claude_haiku": "Claude Haiku (zero-shot)"}

ev, val = data.load("eval"), data.load("val")
present = [m for m in MODELS if data.load_preds(m, "eval") is not None]
if not present:
    raise SystemExit("No predictions found. Run scripts 02 and 03 first.")


def scored(model, frame, split_name):
    return frame.merge(data.load_preds(model, split_name), on="id", how="inner")


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


out, res = [], {"n_eval": len(ev), "toxic_rate": float(ev.toxic.mean()), "models": {}}
out.append("# Content Moderation Audit: Results\n")
out.append(f"Eval set: {len(ev):,} held-out comments, {pct(ev.toxic.mean())} toxic, "
           f"{pct(ev.any_identity.mean())} mention an identity group. Thresholds are set on a separate "
           f"validation set so each model catches {RECALL:.0%} of toxic comments.\n")

# 1. Accuracy and review load
out.append("## 1. Accuracy and review load at 90% recall\n")
out.append("| Model | ROC-AUC | Recall | Precision | Comments sent to review | Review load vs TF-IDF |")
out.append("|---|---|---|---|---|---|")
thr, eval_scored = {}, {}
base_rr = None
for m in present:
    v = scored(m, val, "val")
    e = scored(m, ev, "eval")
    thr[m] = metrics.threshold_at_recall(v["toxic"], v["score"], RECALL)
    eval_scored[m] = e
    st = metrics.at_threshold(e["toxic"], e["score"], thr[m])
    a = metrics.auc(e["toxic"], e["score"])
    if m == "tfidf":
        base_rr = st["review_rate"]
    rel = (base_rr - st["review_rate"]) / base_rr if base_rr else float("nan")
    out.append(f"| {NICE[m]} | {a:.3f} | {pct(st['recall'])} | {pct(st['precision'])} | {pct(st['review_rate'])} | "
               f"{'baseline' if m == 'tfidf' else f'{-100 * rel:+.1f}%'} |")
    res["models"][m] = {"auc": a, "threshold": thr[m], **st, "review_load_change_vs_tfidf": -rel if m != "tfidf" else 0.0}

# 2. Identity bias
out.append("\n## 2. Identity bias\n")
out.append("False-positive rate = share of non-toxic comments wrongly flagged. Jigsaw score blends overall AUC with "
           "the worst-group-weighted subgroup, BPSN and BNSP AUCs (higher is better).\n")
out.append("| Model | Jigsaw bias score | Mean BPSN AUC | FPR, mentions identity | FPR, no identity | Ratio |")
out.append("|---|---|---|---|---|---|")
group_tables = {}
for m in present:
    e = eval_scored[m]
    tab = metrics.subgroup_aucs(e, e["score"])
    group_tables[m] = tab
    final = metrics.jigsaw_final_score(metrics.auc(e["toxic"], e["score"]), tab)
    f = metrics.identity_fpr(e, e["score"], thr[m])
    ratio = "n/a" if np.isnan(f["ratio"]) else f"{f['ratio']:.2f}x"
    out.append(f"| {NICE[m]} | {final:.4f} | {tab['bpsn_auc'].mean():.3f} | {pct(f['fpr_identity'])} | "
               f"{pct(f['fpr_background'])} | {ratio} |")
    res["models"][m].update({"jigsaw_score": final, **f, "groups": tab.to_dict("records")})

worst_model = "distilbert" if "distilbert" in group_tables else "tfidf"
tab = group_tables[worst_model].sort_values("bpsn_auc")
out.append(f"\nPer-group AUCs for {NICE[worst_model]} (lowest BPSN first):\n")
out.append("| Group | Comments | Subgroup AUC | BPSN AUC | BNSP AUC |")
out.append("|---|---|---|---|---|")
for r in tab.itertuples():
    out.append(f"| {r.group} | {r.n:,} | {r.subgroup_auc:.3f} | {r.bpsn_auc:.3f} | {r.bnsp_auc:.3f} |")

# 3. Bias fix
out.append("\n## 3. Effect of the bias fix (reweighting + neutral identity sentences)\n")
out.append("Each model uses its own 90%-recall threshold. The interval is a 90% paired bootstrap over the "
           "non-toxic identity comments.\n")
out.append("| Pair | Identity FPR before | after | Relative drop (90% CI) | Recall before | after | AUC before | after |")
out.append("|---|---|---|---|---|---|---|---|")
res["mitigation"] = {}
for plain, fixed in (("tfidf", "tfidf_mitigated"), ("distilbert", "distilbert_mitigated")):
    if plain in eval_scored and fixed in eval_scored:
        j = eval_scored[plain][["id", "toxic", "any_identity", "score"]].merge(
            eval_scored[fixed][["id", "score"]], on="id", suffixes=("_a", "_b"))
        neg = (j.toxic == 0) & j.any_identity
        r = metrics.fpr_reduction_ci(j.loc[neg, "score_a"] >= thr[plain], j.loc[neg, "score_b"] >= thr[fixed])
        a, b = res["models"][plain], res["models"][fixed]
        out.append(f"| {NICE[plain]} → bias-fixed | {pct(a['fpr_identity'])} | {pct(b['fpr_identity'])} | "
                   f"{pct(r['reduction'])} ({pct(r['lo'])} to {pct(r['hi'])}) | {pct(a['recall'])} | {pct(b['recall'])} | "
                   f"{a['auc']:.3f} | {b['auc']:.3f} |")
        res["mitigation"][plain] = r

# 4. Claude
llm_preds = data.load_preds("claude_haiku", "eval")
if llm_preds is not None:
    sub = ev.merge(llm_preds, on="id")
    out.append(f"\n## 4. Claude Haiku as a zero-shot moderator ({len(sub):,}-comment sample)\n")
    out.append("All models are compared on the same sample. Thresholds are set on this sample at 90% recall "
               "for every model, since Claude has no validation scores.\n")
    out.append("| Model | ROC-AUC | Comments sent to review | FPR, mentions identity | FPR, no identity |")
    out.append("|---|---|---|---|---|")
    res["llm_sample"] = {}
    frames = {"claude_haiku": sub}
    for m in present:
        frames[m] = sub[["id", "toxic", "any_identity"]].merge(eval_scored[m][["id", "score"]], on="id")
    for m, f in frames.items():
        t = metrics.threshold_at_recall(f["toxic"], f["score"], RECALL)
        st = metrics.at_threshold(f["toxic"], f["score"], t)
        fi = metrics.identity_fpr(f, f["score"], t)
        a = metrics.auc(f["toxic"], f["score"])
        out.append(f"| {NICE[m]} | {a:.3f} | {pct(st['review_rate'])} | {pct(fi['fpr_identity'])} | {pct(fi['fpr_background'])} |")
        res["llm_sample"][m] = {"auc": a, **st, **fi}

# 5. Robustness
rob = Path("data/robustness.csv")
if rob.exists():
    r = pd.read_csv(rob)
    piv = r.pivot(index="model", columns="variant", values="recall")
    cols = ["clean"] + [c for c in piv.columns if c != "clean"]
    n = int(r["n"].max())
    out.append(f"\n## 5. Robustness: recall on {n:,} toxic comments with disguised insult words\n")
    out.append("| Model | " + " | ".join(cols) + " |")
    out.append("|" + "---|" * (len(cols) + 1))
    for m, row in piv[cols].iterrows():
        out.append(f"| {NICE.get(m, m)} | " + " | ".join(pct(row[c]) for c in cols) + " |")
    res["robustness"] = piv.to_dict("index")

# 6. Failure modes
fails = Path("data/failures.csv")
if fails.exists():
    fdf = pd.read_csv(fails)
    ct = pd.crosstab(fdf["category"], fdf["error"])
    out.append(f"\n## 6. Why the model's most confident mistakes happened (Claude-labeled, {len(fdf)} mistakes)\n")
    out.append("| Category | " + " | ".join(ct.columns) + " | Total |")
    out.append("|" + "---|" * (len(ct.columns) + 2))
    for cat, row in ct.assign(total=ct.sum(axis=1)).sort_values("total", ascending=False).iterrows():
        out.append(f"| {cat} | " + " | ".join(str(int(row[c])) for c in ct.columns) + f" | {int(row['total'])} |")
    res["failures"] = ct.to_dict()

# 7. Cascade
casc = Path("data/cascade.csv")
if casc.exists():
    c = pd.read_csv(casc)
    out.append(f"\n## 7. Cascade: {NICE.get(c.cheap_model[0], c.cheap_model[0])} first, Claude only for uncertain comments\n")
    out.append("Comments whose first-pass score is closest to its threshold go to Claude. Same sample as section 4.\n")
    out.append("| Sent to Claude | Recall | Precision | Sent to review | FPR, mentions identity | FPR, no identity | Claude calls per 1K | Cost per 1K |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in c.itertuples():
        cost = "n/a" if np.isnan(r.cost_per_1k) else f"${r.cost_per_1k:.3f}"
        out.append(f"| {pct(r.routed_share)} | {pct(r.recall)} | {pct(r.precision)} | {pct(r.review_rate)} | "
                   f"{pct(r.fpr_identity)} | {pct(r.fpr_background)} | {r.claude_calls_per_1k:.0f} | {cost} |")
    res["cascade"] = c.to_dict("records")

Path("results.md").write_text("\n".join(out) + "\n")
Path("results.json").write_text(json.dumps(res, indent=2, default=float))
print("\n".join(out))
print("\nWrote results.md and results.json")
