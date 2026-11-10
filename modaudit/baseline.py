"""TF-IDF + logistic regression moderator."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def make_model(max_features: int = 200_000, c: float = 4.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features,
                                  sublinear_tf=True, strip_accents="unicode", lowercase=True)),
        ("lr", LogisticRegression(C=c, solver="liblinear", max_iter=1000)),
    ])


def fit(model: Pipeline, texts, y, sample_weight=None) -> Pipeline:
    model.fit(list(texts), np.asarray(y), lr__sample_weight=sample_weight)
    return model


def score(model: Pipeline, texts) -> np.ndarray:
    return model.predict_proba(list(texts))[:, 1]


def top_toxic_words(model: Pipeline, k: int = 200, min_len: int = 3) -> list[str]:
    """Single words the model leans on most to call a comment toxic (largest positive weights)."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    coef = model.named_steps["lr"].coef_[0]
    words = []
    for i in np.argsort(coef)[::-1]:
        w = vocab[i]
        if " " not in w and w.isalpha() and len(w) >= min_len:
            words.append(w)
        if len(words) >= k:
            break
    return words
