# Content Moderation Audit: Accuracy, Review Load, Identity Bias and Robustness

Toxicity classifiers decide which comments get hidden or sent to human reviewers. This project measures three
moderators on Jigsaw's Civil Comments data, about 1.8 million real news comments labeled by human raters, and asks
four questions a Trust and Safety team cares about:

1. **How much review work does each model create?** At a fixed 90% recall, what share of all comments does it flag?
2. **Does it treat identity groups fairly?** Are harmless comments that mention a group, such as "I'm a gay Muslim
   woman", flagged more often than other harmless comments?
3. **Can the bias be fixed without losing recall?** Reweighting plus neutral identity sentences, measured before and
   after with a bootstrap interval.
4. **Where does it break?** Recall on toxic comments whose insult words are disguised (leetspeak, dots, stretched
   letters, swapped letters), plus Claude-labeled categories for the model's most confident mistakes.
5. **Is an LLM worth its cost?** A cascade lets the cheap model decide easy comments and sends only the most
   uncertain ones to Claude, then measures recall, false flags and cost per 1,000 comments at each routing share.

## Models

| Model | Trained on | Notes |
|---|---|---|
| TF-IDF + logistic regression | all ~1.44M training comments | word 1–2 grams |
| DistilBERT | 150K-comment sample | fine-tuned 1 epoch, max 128 tokens |
| Claude Haiku | none (zero-shot) | scored on a 2,000-comment sample |

Each trained model also gets a **bias-fixed** version: training rows are reweighted so non-toxic comments that mention
an identity count more, and a few hundred neutral sentences such as "My neighbor is muslim and we get along well" are
added (the approach in Dixon et al., 2018).

## Metrics

- **ROC-AUC** overall.
- **Review load**: the threshold is set on a validation set so the model catches 90% of toxic comments. Review load
  is the share of all eval comments flagged at that threshold. Fewer flags at the same recall means less human work.
- **Jigsaw's bias metrics** for nine identity groups: subgroup AUC, BPSN AUC (low = harmless comments that mention
  the group get flagged) and BNSP AUC (low = abuse aimed at the group gets missed), combined into Jigsaw's final score.
- **Identity false-positive rate**: share of non-toxic comments that mention an identity and get flagged, compared
  with non-toxic comments that do not.

## Run it

### 1. Set up

```bash
conda create -n modaudit python=3.11 -y
conda activate modaudit
pip install -r requirements.txt
python -m pytest -q          # 15 tests should pass
```

### 2. Get the data (free Kaggle account)

1. Open https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules and accept the rules.
2. Download `train.csv` from the Data tab into this repo's `data/` folder (unzip it if it arrives zipped).

Or with the Kaggle CLI (needs your `~/.kaggle/kaggle.json` token):

```bash
pip install kaggle
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -f train.csv -p data
cd data && unzip train.csv.zip && cd ..
```

### 3. Run the steps

```bash
python scripts/01_prepare.py                      # ~2 min. Splits and labels the data
python scripts/02_baseline.py                     # ~20-30 min. TF-IDF, plain and bias-fixed
python scripts/03_distilbert.py                   # ~1-2 h on an Apple Silicon Mac
python scripts/03_distilbert.py --mitigate        # optional, another ~1-2 h
export ANTHROPIC_API_KEY=...                      # set it in your terminal, never paste it in chat
python scripts/04_llm.py --n 2000                 # ~10 min, about $1-2
python scripts/05_robustness.py --llm-n 300       # ~10 min
python scripts/06_failures.py                     # ~2 min
python scripts/08_cascade.py --input-price X --output-price Y   # USD per million tokens for your Claude model
python scripts/07_report.py                       # writes results.md
```

If DistilBERT is too slow, use `--train-n 80000`. If memory runs short in step 2, use `--max-train 600000`.
To check the whole pipeline without spending API credits, set `MODAUDIT_FAKE_LLM=1` (the Claude numbers will be
meaningless).

## Results

Full tables are in [`results.md`](results.md). Eval set: 60,000 held-out comments (8.0% toxic).

- **Accuracy:** DistilBERT had the best ROC-AUC (0.951), ahead of TF-IDF (0.943). Zero-shot Claude Haiku reached
  0.832 on a 2,000-comment sample and had to flag about half of all comments to catch 90% of toxic ones.
- **Identity bias:** at a 90%-recall threshold, DistilBERT wrongly flagged harmless comments that mention an identity
  group 36.6% of the time, against 11.0% for other harmless comments (3.3x). The worst groups by BPSN AUC were
  black (0.768), white (0.793) and gay or lesbian (0.800).
- **Bias fix:** reweighting plus neutral identity sentences cut identity false flags by 31% for DistilBERT (90% CI
  29–33%) and 29% for TF-IDF, with recall unchanged and AUC down by 0.003–0.008. The trade-off: false flags on other
  comments rose (11.0% to 13.5% for DistilBERT), so total review load went up 1–2 points.
- **Robustness:** disguising insult words (dots, leetspeak, stretched or swapped letters) cut DistilBERT's recall on
  1,000 toxic comments from 96% to 28–48% and TF-IDF's from 94% to 33–42%. Claude held at 91–97% on a 300-comment subset.
- **Failure modes:** in a Claude-labeled review of DistilBERT's 300 most confident errors, 130 of 150 missed toxic
  comments were sarcasm or implicit hostility. False flags were spread across identity mentions, quoted toxicity,
  non-abusive profanity and likely label noise.
- **Cascade:** sending DistilBERT's most uncertain comments to Claude raised recall from 87% to 94% when 20% were
  routed, at about $0.02 per 1,000 comments, but review load rose from 19% to 28% because Claude's scores are poorly
  calibrated to these human labels. Claude looks more useful as a second check for disguised abuse than as a judge
  for every borderline comment.

## Engineering notes: problems hit and trade-offs made

Everything ran on a MacBook Air (Apple M2, no fan, shared CPU/GPU memory). These are the issues that came up and
how each was handled.

| Problem | Cause | Fix | Trade-off |
|---|---|---|---|
| PyTorch stuck at 2.2 and NumPy errors | Anaconda was the Intel (x86_64) build running under Rosetta, and PyTorch stopped shipping Intel Mac builds after 2.2 | Built a native arm64 environment with Miniforge | One more environment to manage, but training ran on the M2 GPU (MPS) several times faster |
| GPU out of memory at step 1,600 of DistilBERT training | Batch size 32 on shared memory, and PyTorch kept cached GPU memory between steps | Batch size 16 with gradient accumulation over 2 steps (same effective batch of 32), and the MPS cache is cleared every 50 steps | Slightly slower per epoch, same optimization behavior |
| DistilBERT trained on 100K comments, not all 1.6M | One epoch on the full set would take most of a day on a fanless laptop | Uniform 100K sample, 1 epoch, 128-token limit | Not the best possible transformer score. Its 0.951 AUC is a floor, not a ceiling |
| 87% of Claude's answers came back empty | Haiku 5.5 reasons before answering, and the 10-token output limit ran out before the number | Raised the limit to 1,024 tokens, stopped caching empty answers, and logged refusals separately | Each call uses about 126 output tokens instead of 10. Cost is still about $0.09 per 1,000 comments |
| Every Claude call failed with "invalid x-api-key" | A placeholder value overwrote the real key in the terminal | Tested the key directly against the API, then made the code stop on the first authentication error instead of logging 2,000 failures | None |
| Claude has no validation scores | It is zero-shot, so there is no separate set to pick its threshold on | Every model gets its threshold on the same 2,000-comment sample in the Claude comparison | Those thresholds are in-sample, so section 4 is a fair comparison between models but slightly optimistic for all of them |
| Bias-fixed DistilBERT took 10 hours instead of 50 minutes | The Mac throttled overnight | None needed. Only the run time changed, not the results | Keep the lid open and the charger in for long runs |

## Layout

```
modaudit/
  data.py        load, label and split the Kaggle file
  metrics.py     AUC, threshold at target recall, review load, Jigsaw bias metrics, bootstrap
  mitigation.py  bias reweighting and neutral identity sentences
  baseline.py    TF-IDF + logistic regression
  distilbert.py  DistilBERT fine-tuning (uses Apple's GPU when available)
  llm.py         Claude as a moderator and as a failure-mode labeler, with a response cache
  robustness.py  disguise insult words and measure recall
  cascade.py     cheap model first, Claude only for uncertain comments
scripts/         numbered steps 01-08
tests/           unit tests on a small fake dataset
```

## Limitations

- DistilBERT is trained on a sample, so it is not the best possible transformer model on this data.
- Claude is scored on 2,000 comments, so its numbers have wider uncertainty than the other models.
- The disguised words are the ones the TF-IDF model weights most, so the attack is aimed at word-based models.
- Jigsaw's labels come from crowd raters, and some are wrong. The failure review counts how often that happens.

## Data and references

- Jigsaw Unintended Bias in Toxicity Classification (Kaggle, 2019). Civil Comments data, CC0.
- Borkan et al., 2019. Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.
- Dixon et al., 2018. Measuring and Mitigating Unintended Bias in Text Classification.
