"""Load the Jigsaw Unintended Bias (Civil Comments) data and make fixed splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The nine identity groups Jigsaw used in its official bias metric.
IDENTITIES = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]

DATA_DIR = Path("data")


def load_jigsaw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read Kaggle's train.csv or all_data.csv and add binary labels.

    toxic = 1 when at least half of the human raters called the comment toxic.
    An identity flag is True when at least half of raters said the comment mentions it.
    Identity columns are missing for most rows, and missing counts as "not mentioned".
    """
    header = set(pd.read_csv(path, nrows=0).columns)
    # train.csv calls the toxicity score "target"; all_data.csv calls it "toxicity".
    label = "target" if "target" in header else "toxicity" if "toxicity" in header else None
    missing = [c for c in ["id", "comment_text", *IDENTITIES] if c not in header]
    if label is None or missing:
        raise ValueError(f"{path} is missing columns: {missing or ['target or toxicity']}")
    cols = ["id", "comment_text", label, *IDENTITIES]
    df = pd.read_csv(path, usecols=cols, nrows=nrows).rename(columns={label: "target"})
    df = df.dropna(subset=["comment_text", "target"]).reset_index(drop=True)
    df["comment_text"] = df["comment_text"].astype(str)
    df["identity_labeled"] = df[IDENTITIES].notna().any(axis=1)
    for col in IDENTITIES:
        df[col] = df[col].fillna(0.0) >= 0.5
    df["any_identity"] = df[IDENTITIES].any(axis=1)
    df["toxic"] = (df["target"] >= 0.5).astype(int)
    return df[["id", "comment_text", "target", "toxic", "identity_labeled", "any_identity", *IDENTITIES]]


def split(df: pd.DataFrame, seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.1) -> dict[str, pd.DataFrame]:
    """Random 80/10/10 split, fixed by seed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    n_val = int(len(df) * val_frac)
    test_idx, val_idx, train_idx = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def sample(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Uniform random sample of n rows (all rows when n is None or larger than the frame)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def save(frames: dict[str, pd.DataFrame], data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(data_dir / f"{name}.parquet", index=False)


def load(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/01_prepare.py first.")
    return pd.read_parquet(path)


def save_preds(model: str, split_name: str, ids, scores, data_dir: Path = DATA_DIR) -> Path:
    out = data_dir / "preds"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model}__{split_name}.parquet"
    pd.DataFrame({"id": np.asarray(ids), "score": np.asarray(scores, dtype=float)}).to_parquet(path, index=False)
    return path


def load_preds(model: str, split_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    path = data_dir / "preds" / f"{model}__{split_name}.parquet"
    return pd.read_parquet(path) if path.exists() else None
