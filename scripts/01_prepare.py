"""Read Kaggle's train.csv, label it and save fixed train / val / eval splits."""
import argparse

import _setup  # noqa: F401
from modaudit import data

p = argparse.ArgumentParser()
p.add_argument("--csv", default=None, help="Kaggle train.csv or all_data.csv (default: whichever is in data/)")
p.add_argument("--nrows", type=int, default=None, help="read only the first N rows (for a quick test)")
p.add_argument("--val-n", type=int, default=20_000, help="validation rows used to pick thresholds")
p.add_argument("--eval-n", type=int, default=60_000, help="held-out rows every model is scored on")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

csv = a.csv
if csv is None:
    from pathlib import Path
    csv = next((str(f) for f in (Path("data/train.csv"), Path("data/all_data.csv")) if f.exists()), None)
    if csv is None:
        raise SystemExit("Put train.csv or all_data.csv in the data/ folder.")
print(f"Reading {csv} ...", flush=True)
df = data.load_jigsaw(csv, nrows=a.nrows)
parts = data.split(df, seed=a.seed)
frames = {
    "train": parts["train"],
    "val": data.sample(parts["val"], a.val_n, a.seed),
    "eval": data.sample(parts["test"], a.eval_n, a.seed),
}
data.save(frames)
print(f"Loaded {len(df):,} comments. Toxic rate {df.toxic.mean():.1%}. "
      f"Identity mentioned in {df.any_identity.mean():.1%}.")
for k, v in frames.items():
    print(f"  {k:5s} {len(v):>9,} rows  toxic {v.toxic.mean():.1%}  identity {v.any_identity.mean():.1%}")
