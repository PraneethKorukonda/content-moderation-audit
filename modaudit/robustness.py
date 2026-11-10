"""Disguise the insult words in toxic comments and see whether moderators still catch them."""
from __future__ import annotations

import re

import numpy as np

LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}


def leet(word: str) -> str:
    return "".join(LEET.get(c, c) for c in word)


def dotted(word: str) -> str:
    return ".".join(word)


def stretched(word: str) -> str:
    """Repeat the first vowel three times: idiot -> iiidiot."""
    for i, c in enumerate(word):
        if c in "aeiou":
            return word[:i] + c * 3 + word[i + 1:]
    return word + word[-1] * 2


def swapped(word: str) -> str:
    """Swap the two middle letters: stupid -> stpuid."""
    if len(word) < 4:
        return word
    m = len(word) // 2
    return word[:m - 1] + word[m] + word[m - 1] + word[m + 1:]


PERTURBATIONS = {"leetspeak": leet, "dotted": dotted, "stretched": stretched, "swapped": swapped}


def perturb(text: str, words: set[str], kind: str) -> tuple[str, int]:
    """Disguise every whole-word occurrence of a target word. Returns (new text, number changed)."""
    fn = PERTURBATIONS[kind]
    count = 0

    def repl(m):
        nonlocal count
        w = m.group(0)
        if w.lower() in words:
            count += 1
            out = fn(w.lower())
            return out.upper() if w.isupper() else out
        return w

    return re.sub(r"[A-Za-z]+", repl, text), count


def build_attack_set(df, words: list[str], n: int = 1000, seed: int = 0):
    """Toxic comments that contain at least one target word, with each disguise applied."""
    wset = set(words)
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    pool = df[(df["toxic"] == 1) & df["comment_text"].str.contains(pattern, regex=True)]
    if len(pool) > n:
        pool = pool.sample(n=n, random_state=seed)
    pool = pool.reset_index(drop=True)
    variants = {"clean": pool["comment_text"].tolist()}
    for kind in PERTURBATIONS:
        variants[kind] = [perturb(t, wset, kind)[0] for t in pool["comment_text"]]
    return pool, variants


def recall(scores, thr: float) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[~np.isnan(s)]
    return float((s >= thr).mean()) if len(s) else float("nan")
