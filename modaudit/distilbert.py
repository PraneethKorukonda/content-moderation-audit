"""Fine-tune DistilBERT as a toxicity classifier. Uses Apple's GPU (mps) when available."""
from __future__ import annotations

import numpy as np


def device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(texts, y, weights=None, model_name: str = "distilbert-base-uncased", max_len: int = 128,
          epochs: int = 1, batch_size: int = 16, lr: float = 5e-5, seed: int = 0, log_every: int = 200,
          grad_accum: int = 2):
    """Effective batch = batch_size * grad_accum. Small batches plus accumulation keep GPU memory low on a laptop."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(dev)

    texts = list(texts)
    y = np.asarray(y, dtype=np.float32)
    w = np.ones(len(texts), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    idx = list(range(len(texts)))

    def collate(batch):
        enc = tok([texts[i] for i in batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y[batch])
        enc["weights"] = torch.tensor(w[batch])
        return enc

    loader = DataLoader(idx, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * len(loader)
    updates = max(1, steps // grad_accum)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * updates), updates)
    is_mps = dev.type == "mps"
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    step = 0
    for _ in range(epochs):
        for batch in loader:
            labels = batch.pop("labels").to(dev)
            bw = batch.pop("weights").to(dev)
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(**batch).logits.squeeze(-1)
            loss = (loss_fn(logits, labels) * bw).mean()
            (loss / grad_accum).backward()
            step += 1
            if step % grad_accum == 0 or step == steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if is_mps and step % 50 == 0:
                torch.mps.empty_cache()  # return cached GPU memory so long runs don't run out
            if step % log_every == 0:
                print(f"  step {step}/{steps}  loss {loss.item():.4f}", flush=True)
    model.eval()
    return model, tok


def score(model, tok, texts, max_len: int = 128, batch_size: int = 64) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    texts = list(texts)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy())
            if dev.type == "mps" and (i // batch_size) % 50 == 0:
                torch.mps.empty_cache()
            if (i // batch_size) % 200 == 0 and i:
                print(f"  scored {i:,}/{len(texts):,}", flush=True)
    return np.concatenate(out) if out else np.array([])


def save(model, tok, path) -> None:
    model.save_pretrained(path)
    tok.save_pretrained(path)


def load(path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(path).to(device())
    model.eval()
    return model, AutoTokenizer.from_pretrained(path)
