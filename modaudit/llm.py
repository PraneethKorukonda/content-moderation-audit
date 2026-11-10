"""Claude as a moderator, and as a reviewer that labels why models fail."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MODEL = os.environ.get("MODAUDIT_LLM", "claude-haiku-5-5")
# Room for the model to reason before its one-line answer. Short limits returned empty text on Haiku 5.5.
MAX_TOKENS = int(os.environ.get("MODAUDIT_MAX_TOKENS", "1024"))
CACHE = Path("data/llm_cache.jsonl")

MODERATE_PROMPT = """You are a content moderator for a news comment section.
A comment is toxic if it is rude, disrespectful or unreasonable enough that it would make
someone want to leave the discussion. Mentioning a religion, race, gender, sexual orientation
or mental illness is not toxic by itself.

Comment:
<comment>
{text}
</comment>

How likely is it that most human moderators would call this comment toxic?
Reply with a single integer from 0 to 100 and nothing else."""

FAILURE_CATEGORIES = [
    "identity_mention",   # harmless comment flagged because it names a group
    "obfuscation",        # misspelled or disguised insult the model missed
    "sarcasm_or_implicit",  # hostility without explicit insult words
    "quoted_or_reported",  # toxic words quoted, reported or discussed, not aimed at anyone
    "profanity_not_abusive",  # swearing that raters did not see as toxic
    "label_noise",        # the human label itself looks wrong
    "other",
]

FAILURE_PROMPT = """A toxicity model made a mistake on this comment.

Comment:
<comment>
{text}
</comment>

Human raters' label: {label}
Model's toxicity score: {score:.2f} (it called the comment {verdict})

Which one category best explains the mistake?
- identity_mention: a harmless comment flagged because it names a religion, race, gender, sexual orientation or mental illness
- obfuscation: a misspelled or disguised insult the model missed
- sarcasm_or_implicit: hostility without explicit insult words
- quoted_or_reported: toxic words quoted, reported or discussed rather than aimed at someone
- profanity_not_abusive: swearing that raters did not see as toxic
- label_noise: the human label itself looks wrong
- other

Reply with the category name only."""


def _client():
    if os.environ.get("MODAUDIT_FAKE_LLM"):
        return "fake"  # offline dry run: no API calls, made-up answers
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=...")
    return anthropic.Anthropic(max_retries=6)


def _load_cache() -> dict:
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("out"):  # empty answers are retried, not reused
                    cache[rec["key"]] = rec["out"]
    return cache


def _append_cache(key: str, out: str) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as f:
        f.write(json.dumps({"key": key, "out": out}) + "\n")


def _call(client, model: str, prompt: str, max_tokens: int, kind: str = "mod") -> str:
    if client == "fake":
        return "other" if "category" in prompt else str(abs(hash(prompt)) % 101)
    resp = client.messages.create(model=model, max_tokens=max_tokens,
                                  messages=[{"role": "user", "content": prompt}])
    usage = getattr(resp, "usage", None)
    if usage is not None:
        _log_usage(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0), kind)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    if not text and getattr(resp, "stop_reason", "") == "refusal":
        return "REFUSED"
    return text


USAGE = Path("data/llm_usage.jsonl")


def _log_usage(model: str, input_tokens: int, output_tokens: int, kind: str = "mod") -> None:
    """Record real token counts (including any reasoning tokens) so the cascade can price Claude calls."""
    USAGE.parent.mkdir(parents=True, exist_ok=True)
    with USAGE.open("a") as f:
        f.write(json.dumps({"model": model, "kind": kind, "in": int(input_tokens), "out": int(output_tokens)}) + "\n")


def mean_usage() -> dict | None:
    """Average input and output tokens per moderation call, from the usage log."""
    if not USAGE.exists():
        return None
    recs = [json.loads(x) for x in USAGE.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r.get("kind") == "mod"]  # moderation calls only
    if not recs:
        return None
    return {"in": sum(r["in"] for r in recs) / len(recs), "out": sum(r["out"] for r in recs) / len(recs), "n": len(recs)}


def parse_score(text: str) -> float:
    """Use the last 1-3 digit number in the reply (the answer usually comes last). Refusals count as missing."""
    if not text or text == "REFUSED":
        return float("nan")
    nums = re.findall(r"(?<!\d)\d{1,3}(?!\d)", text)
    if not nums:
        return float("nan")
    return min(int(nums[-1]), 100) / 100.0


def parse_category(text: str) -> str:
    t = (text or "").strip().lower()
    for c in FAILURE_CATEGORIES:
        if c in t:
            return c
    return "other"


def _run(jobs: list[tuple[str, str]], model: str, max_tokens: int, workers: int, kind: str = "mod") -> dict:
    """jobs: (cache_key, prompt). Returns key -> raw text. Cached answers are reused."""
    cache = _load_cache()
    todo = [(k, p) for k, p in jobs if k not in cache]
    if todo:
        client = _client()

        def work(job):
            k, p = job
            try:
                return k, _call(client, model, p, max_tokens, kind)
            except Exception as e:  # keep going; the row is reported as unparseable
                if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError"):
                    raise SystemExit(f"Stopping: {type(e).__name__}. Check your API key and model name. ({e})")
                print(f"  call failed: {type(e).__name__}: {e}", flush=True)
                return k, None

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, out in ex.map(work, todo):
                done += 1
                if out:
                    cache[k] = out
                    _append_cache(k, out)
                if done % 100 == 0:
                    print(f"  {done}/{len(todo)} calls", flush=True)
    return {k: cache.get(k) for k, _ in jobs}


def moderate(ids, texts, model: str = DEFAULT_MODEL, workers: int = 8, tag: str = "clean") -> list[float]:
    jobs = [(f"mod|{model}|{tag}|{i}", MODERATE_PROMPT.format(text=t[:4000])) for i, t in zip(ids, texts)]
    raw = _run(jobs, model, max_tokens=MAX_TOKENS, workers=workers, kind="mod")
    return [parse_score(raw[k]) for k, _ in jobs]


def label_failures(rows, model: str = DEFAULT_MODEL, workers: int = 8) -> list[str]:
    """rows: iterable of dicts with id, comment_text, toxic, score, threshold."""
    jobs = []
    for r in rows:
        verdict = "toxic" if r["score"] >= r["threshold"] else "not toxic"
        label = "toxic" if r["toxic"] == 1 else "not toxic"
        prompt = FAILURE_PROMPT.format(text=r["comment_text"][:4000], label=label, score=r["score"], verdict=verdict)
        jobs.append((f"fail|{model}|{r['id']}", prompt))
    raw = _run(jobs, model, max_tokens=MAX_TOKENS, workers=workers, kind="fail")
    return [parse_category(raw[k]) for k, _ in jobs]
