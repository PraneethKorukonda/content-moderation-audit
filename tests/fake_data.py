"""Small fake dataset with the same columns as Kaggle's train.csv, for tests only."""
import numpy as np
import pandas as pd

from modaudit.data import IDENTITIES

INSULTS = ["idiot", "stupid", "moron", "pathetic", "loser", "dumb"]
NEUTRAL = ["the", "council", "voted", "budget", "road", "school", "weather", "game", "tax", "plan", "today", "city"]
ID_WORDS = {"muslim": "muslim", "homosexual_gay_or_lesbian": "gay", "black": "black", "christian": "christian"}


def make(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        group = rng.choice(list(ID_WORDS)) if rng.random() < 0.3 else None
        toxic = rng.random() < (0.35 if group else 0.12)  # identity comments are toxic more often -> bias
        words = list(rng.choice(NEUTRAL, size=8))
        if toxic:
            words += list(rng.choice(INSULTS, size=2))
        if group:
            words.insert(int(rng.integers(0, len(words))), ID_WORDS[group])
        row = {"id": i, "comment_text": " ".join(words), "target": 0.8 if toxic else 0.1}
        labeled = group is not None or rng.random() < 0.2
        for g in IDENTITIES:
            row[g] = (1.0 if g == group else 0.0) if labeled else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
