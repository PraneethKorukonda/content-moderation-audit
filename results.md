# Content Moderation Audit: Results

Eval set: 60,000 held-out comments, 8.0% toxic, 9.0% mention an identity group. Thresholds are set on a separate validation set so each model catches 90% of toxic comments.

## 1. Accuracy and review load at 90% recall

| Model | ROC-AUC | Recall | Precision | Comments sent to review | Review load vs TF-IDF |
|---|---|---|---|---|---|
| TF-IDF + LR | 0.943 | 87.2% | 36.9% | 18.8% | baseline |
| TF-IDF + LR, bias-fixed | 0.935 | 87.7% | 33.2% | 21.0% | +11.7% |
| DistilBERT | 0.951 | 89.3% | 37.1% | 19.1% | +1.9% |
| DistilBERT, bias-fixed | 0.948 | 89.7% | 34.9% | 20.4% | +8.9% |

## 2. Identity bias

False-positive rate = share of non-toxic comments wrongly flagged. Jigsaw score blends overall AUC with the worst-group-weighted subgroup, BPSN and BNSP AUCs (higher is better).

| Model | Jigsaw bias score | Mean BPSN AUC | FPR, mentions identity | FPR, no identity | Ratio |
|---|---|---|---|---|---|
| TF-IDF + LR | 0.8908 | 0.836 | 32.9% | 11.1% | 2.97x |
| TF-IDF + LR, bias-fixed | 0.8911 | 0.890 | 23.4% | 14.5% | 1.62x |
| DistilBERT | 0.9041 | 0.854 | 36.6% | 11.0% | 3.33x |
| DistilBERT, bias-fixed | 0.9065 | 0.906 | 25.2% | 13.5% | 1.87x |

Per-group AUCs for DistilBERT (lowest BPSN first):

| Group | Comments | Subgroup AUC | BPSN AUC | BNSP AUC |
|---|---|---|---|---|
| black | 497 | 0.804 | 0.768 | 0.967 |
| white | 807 | 0.807 | 0.793 | 0.965 |
| homosexual_gay_or_lesbian | 357 | 0.842 | 0.800 | 0.970 |
| muslim | 732 | 0.849 | 0.828 | 0.962 |
| jewish | 282 | 0.886 | 0.863 | 0.962 |
| male | 1,455 | 0.897 | 0.899 | 0.955 |
| psychiatric_or_mental_illness | 136 | 0.908 | 0.904 | 0.954 |
| female | 1,714 | 0.909 | 0.904 | 0.956 |
| christian | 1,337 | 0.916 | 0.925 | 0.946 |

## 3. Effect of the bias fix (reweighting + neutral identity sentences)

Each model uses its own 90%-recall threshold. The interval is a 90% paired bootstrap over the non-toxic identity comments.

| Pair | Identity FPR before | after | Relative drop (90% CI) | Recall before | after | AUC before | after |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR → bias-fixed | 32.9% | 23.4% | 29.0% (26.8% to 31.1%) | 87.2% | 87.7% | 0.943 | 0.935 |
| DistilBERT → bias-fixed | 36.6% | 25.2% | 31.1% (28.9% to 33.0%) | 89.3% | 89.7% | 0.951 | 0.948 |

## 4. Claude Haiku as a zero-shot moderator (1,998-comment sample)

All models are compared on the same sample. Thresholds are set on this sample at 90% recall for every model, since Claude has no validation scores.

| Model | ROC-AUC | Comments sent to review | FPR, mentions identity | FPR, no identity |
|---|---|---|---|---|
| Claude Haiku (zero-shot) | 0.832 | 49.9% | 65.6% | 44.3% |
| TF-IDF + LR | 0.946 | 20.8% | 38.2% | 12.7% |
| TF-IDF + LR, bias-fixed | 0.945 | 22.7% | 28.0% | 16.0% |
| DistilBERT | 0.938 | 25.4% | 52.9% | 16.9% |
| DistilBERT, bias-fixed | 0.945 | 20.2% | 33.1% | 12.5% |

## 5. Robustness: recall on 1,000 toxic comments with disguised insult words

| Model | clean | dotted | leetspeak | stretched | swapped |
|---|---|---|---|---|---|
| Claude Haiku (zero-shot) | 89.3% | 96.7% | 97.3% | 93.3% | 91.0% |
| DistilBERT | 96.3% | 27.9% | 33.2% | 42.9% | 47.9% |
| DistilBERT, bias-fixed | 94.1% | 29.6% | 40.1% | 50.8% | 55.5% |
| TF-IDF + LR | 93.9% | 33.4% | 40.2% | 33.9% | 41.9% |
| TF-IDF + LR, bias-fixed | 93.3% | 35.8% | 42.9% | 36.9% | 43.8% |

## 6. Why the model's most confident mistakes happened (Claude-labeled, 300 mistakes)

| Category | false_flag | missed_toxic | Total |
|---|---|---|---|
| sarcasm_or_implicit | 34 | 130 | 164 |
| other | 35 | 0 | 35 |
| label_noise | 21 | 10 | 31 |
| identity_mention | 24 | 0 | 24 |
| quoted_or_reported | 18 | 3 | 21 |
| profanity_not_abusive | 18 | 0 | 18 |
| obfuscation | 0 | 7 | 7 |

## 7. Cascade: DistilBERT first, Claude only for uncertain comments

Comments whose first-pass score is closest to its threshold go to Claude. Same sample as section 4.

| Sent to Claude | Recall | Precision | Sent to review | FPR, mentions identity | FPR, no identity | Claude calls per 1K | Cost per 1K |
|---|---|---|---|---|---|---|---|
| 0.0% | 87.2% | 35.5% | 19.2% | 41.4% | 10.8% | 0 | $0.000 |
| 5.0% | 89.1% | 34.5% | 20.2% | 42.7% | 11.7% | 50 | $0.004 |
| 10.0% | 91.0% | 31.5% | 22.6% | 48.4% | 13.8% | 100 | $0.009 |
| 20.0% | 94.2% | 26.4% | 27.8% | 56.7% | 19.0% | 200 | $0.018 |
| 30.0% | 94.9% | 22.5% | 33.0% | 65.0% | 24.3% | 300 | $0.027 |
| 50.0% | 97.4% | 18.2% | 41.7% | 71.3% | 33.8% | 500 | $0.045 |
| 100.0% | 95.5% | 14.9% | 49.9% | 65.6% | 44.3% | 1000 | $0.089 |
