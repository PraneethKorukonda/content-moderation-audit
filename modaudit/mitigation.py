"""Bias fixes: sample reweighting and neutral identity sentences for augmentation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import IDENTITIES


def bias_weights(df: pd.DataFrame, label: str = "toxic") -> np.ndarray:
    """Upweight the rows that teach a model not to equate an identity word with toxicity.

    Every row starts at 1. Rows that mention an identity get +1. Toxic rows with no identity
    (background positives) get +1, and non-toxic rows that mention an identity (subgroup
    negatives) get +2, since those are the false positives we want to cut. Weights are
    rescaled to average 1 so the overall loss scale does not change.
    """
    y = df[label].to_numpy()
    ident = df["any_identity"].to_numpy().astype(bool)
    w = np.ones(len(df))
    w += ident
    w += (y == 1) & ~ident
    w += 2 * ((y == 0) & ident)
    return w / w.mean()


IDENTITY_TERMS = {
    "male": ["male", "a man"],
    "female": ["female", "a woman"],
    "homosexual_gay_or_lesbian": ["gay", "lesbian", "homosexual"],
    "christian": ["christian", "catholic"],
    "jewish": ["jewish", "a jew"],
    "muslim": ["muslim", "islamic"],
    "black": ["black", "african american"],
    "white": ["white"],
    "psychiatric_or_mental_illness": ["mentally ill", "bipolar", "depressed"],
}

TEMPLATES = [
    "I am {t}.",
    "My neighbor is {t} and we get along well.",
    "Being {t} is nothing to be ashamed of.",
    "My best friend is {t}.",
    "The new teacher at our school is {t}.",
    "I had lunch with a coworker who is {t}.",
    "Our team captain is {t} and works hard.",
    "She told me she is {t} and I was glad she shared that.",
]


def neutral_identity_sentences(repeats: int = 30, seed: int = 0) -> pd.DataFrame:
    """Non-toxic sentences that mention each identity (the approach in Dixon et al., 2018).

    They give the model examples where an identity word appears with a non-toxic label.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for group, terms in IDENTITY_TERMS.items():
        for _ in range(repeats):
            t = terms[rng.integers(len(terms))]
            tpl = TEMPLATES[rng.integers(len(TEMPLATES))]
            row = {"id": -1, "comment_text": tpl.format(t=t), "target": 0.0, "toxic": 0,
                   "identity_labeled": True, "any_identity": True}
            row.update({g: g == group for g in IDENTITIES})
            rows.append(row)
    return pd.DataFrame(rows)
