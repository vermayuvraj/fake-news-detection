"""Fine-tunes DistilBERT under the same three protocols as the linear models.

The comparison is only meaningful if the transformer sees the same documents in
the same splits, so this script rebuilds the primary frame while preserving the
original row index, replicates each splitting protocol exactly (asserting the
resulting split sizes against the linear-model runs), and then feeds DistilBERT
a *minimally* cleaned version of those same documents.

Why minimal cleaning: the linear pipeline maps everything outside [a-z'] to
whitespace, which suits a bag-of-words model but discards punctuation and casing
that a subword transformer can use. Applying the same destructive normalisation
would handicap the transformer for reasons unrelated to the research question.
The leakage controls are identical (metadata excluded, source tag stripped,
duplicates removed); only the surface normalisation differs, and the manuscript
says so.

Usage:  python experiments/run_transformer.py
"""

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from lib import PrepConfig, load_raw, LABEL_FAKE, LABEL_REAL, SEED  # noqa: E402
from src.text_clean import clean_text, strip_source_artifacts  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH = 16
EPOCHS = 2
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Split sizes produced by the linear-model runs; asserted so that any drift in
# the shared preprocessing code is caught immediately rather than silently
# invalidating the comparison.
EXPECTED = {
    "random":         {"train": 27177, "val": 3883, "test": 7766},
    "topic_disjoint": {"train": 24326, "val": 3476, "test": 11024},
    "temporal":       {"train": 25758, "val": 3679, "test": 7361},
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_WS = re.compile(r"\s+")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def minimal_clean(text):
    """Leakage controls only: strip the source tag, URLs and e-mails."""
    if not isinstance(text, str):
        return ""
    text = strip_source_artifacts(text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Frame construction: identical rows/order to lib.build_frame, plus raw_idx
# and the minimally cleaned text.
# --------------------------------------------------------------------------- #
def build_aligned_frame(raw, cfg=PrepConfig()):
    title = raw["title"].fillna("").astype(str)
    body = raw["text"].fillna("").astype(str)
    combined = title + " " + body

    content = combined.apply(
        lambda t: clean_text(t, strip_artifacts=cfg.strip_reuters))
    out = pd.DataFrame({
        "content": content,
        "bert_text": combined.apply(minimal_clean),
        "label": raw["label"].values,
        "subject": raw["subject"].values,
        "date": raw["date_parsed"].values,
    })
    out = out[out["content"].str.len() >= cfg.min_chars]
    if cfg.dedup:
        out = out.drop_duplicates(subset="content", keep="first")
    return out.reset_index(drop=True)


def split_random_idx(frame, seed=SEED):
    from sklearn.model_selection import train_test_split
    X, y = frame["content"], frame["label"]
    Xt, Xte, yt, _ = train_test_split(X, y, test_size=0.20, stratify=y,
                                      random_state=seed)
    Xtr, Xva, _, _ = train_test_split(Xt, yt, test_size=0.10 / 0.80,
                                      stratify=yt, random_state=seed)
    return Xtr.index, Xva.index, Xte.index


def split_topic_idx(frame, seed=SEED):
    from sklearn.model_selection import train_test_split
    train_subj = {"politicsNews", "News", "politics", "left-news"}
    test_subj = {"worldnews", "Government News", "US_News", "Middle-east"}
    tr = frame[frame["subject"].isin(train_subj)]
    te = frame[frame["subject"].isin(test_subj)]
    Xtr, Xva, _, _ = train_test_split(tr["content"], tr["label"],
                                      test_size=0.125, stratify=tr["label"],
                                      random_state=seed)
    return Xtr.index, Xva.index, te.index


def split_temporal_idx(frame, seed=SEED):
    f = frame.dropna(subset=["date"]).copy()
    lo = max(f.loc[f.label == LABEL_REAL, "date"].min(),
             f.loc[f.label == LABEL_FAKE, "date"].min())
    hi = min(f.loc[f.label == LABEL_REAL, "date"].max(),
             f.loc[f.label == LABEL_FAKE, "date"].max())
    f = f[(f["date"] >= lo) & (f["date"] <= hi)]
    f = f.sort_values("date", kind="mergesort")   # stable, as in lib
    n = len(f)
    n_tr, n_va = int(0.70 * n), int(0.10 * n)
    return (f.iloc[:n_tr].index, f.iloc[n_tr:n_tr + n_va].index,
            f.iloc[n_tr + n_va:].index)


SPLITTERS = {"random": split_random_idx,
             "topic_disjoint": split_topic_idx,
             "temporal": split_temporal_idx}


# --------------------------------------------------------------------------- #
# Data plumbing
# --------------------------------------------------------------------------- #
class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "label": self.labels[i]}


def make_collate(tok):
    def collate(batch):
        labels = torch.tensor([b.pop("label") for b in batch])
        enc = tok.pad(batch, return_tensors="pt")
        enc["labels"] = labels
        return enc
    return collate


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(**batch).logits.float()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def full_metrics(y, score, target_prior, m_repeats=20, seed=SEED):
    """Same metric set as the linear-model shift analysis."""
    pred = (score >= 0.5).astype(int)
    prec, rec, _ = precision_recall_curve(y, score)
    denom = prec + rec
    f1_curve = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec),
                         where=denom > 0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_neg = int(round(len(pos) * (1 - target_prior) / target_prior))
    matched = []
    if 0 < n_neg <= len(neg):
        rng = np.random.default_rng(seed)
        for _ in range(m_repeats):
            sub = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            matched.append(f1_score(y[sub], pred[sub], zero_division=0))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_default_threshold": float(f1_score(y, pred, zero_division=0)),
        "f1_threshold_oracle": float(np.max(f1_curve)),
        "auc_roc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_prior_matched_mean": float(np.mean(matched)) if matched else None,
        "f1_prior_matched_std": float(np.std(matched)) if matched else None,
        "n_test": int(len(y)),
        "fake_ratio_test": float(y.mean()),
    }


def run_protocol(frame, proto, tok, target_prior, max_len=MAX_LEN):
    tr_i, va_i, te_i = SPLITTERS[proto](frame)
    sizes = {"train": len(tr_i), "val": len(va_i), "test": len(te_i)}
    if proto in EXPECTED:
        assert sizes == EXPECTED[proto], (
            f"{proto} split {sizes} != linear-model split {EXPECTED[proto]}; "
            "the comparison would not be like-for-like")
    log(f"  {proto}: sizes {sizes} (match linear-model splits)")

    collate = make_collate(tok)
    dl = {}
    for name, idx, shuf in [("train", tr_i, True), ("val", va_i, False),
                            ("test", te_i, False)]:
        ds = TextDS(frame.loc[idx, "bert_text"], frame.loc[idx, "label"],
                    tok, max_len)
        dl[name] = DataLoader(ds, batch_size=BATCH if shuf else 64,
                             shuffle=shuf, collate_fn=collate)

    set_seeds()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total = len(dl["train"]) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total, pct_start=0.1, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best = {"val_f1": -1.0, "state": None, "epoch": None}
    t_start = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        for step, batch in enumerate(dl["train"], 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % 400 == 0:
                log(f"    epoch {ep} step {step}/{len(dl['train'])} "
                    f"loss {loss.item():.4f}")
        yv, sv = predict(model, dl["val"])
        vf1 = f1_score(yv, (sv >= 0.5).astype(int))
        log(f"    epoch {ep} done in {(time.perf_counter()-t0)/60:.1f} min, "
            f"val F1 = {vf1:.4f}")
        if vf1 > best["val_f1"]:
            best = {"val_f1": float(vf1), "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
    train_min = (time.perf_counter() - t_start) / 60

    model.load_state_dict(best["state"])          # selection on validation only
    yt, st = predict(model, dl["test"])
    out = {"sizes": sizes, "val_f1_selected": best["val_f1"],
           "selected_epoch": best["epoch"], "train_minutes": round(train_min, 2),
           "max_len": max_len,
           "test": full_metrics(yt, st, target_prior)}
    del model, best
    torch.cuda.empty_cache()
    log(f"  {proto}: test F1 = {out['test']['f1_default_threshold']:.4f}  "
        f"AP = {out['test']['average_precision']:.4f}")
    return out


def main():
    log(f"device = {DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    raw = load_raw()
    frame = build_aligned_frame(raw)
    target_prior = float(frame["label"].mean())
    log(f"aligned frame: {len(frame):,} documents, fake ratio {target_prior:.4f}")
    assert len(frame) == 38826, f"frame size {len(frame)} != 38826"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    R = {"config": {"model": MODEL_NAME, "max_len": MAX_LEN, "batch": BATCH,
                    "epochs": EPOCHS, "lr": LR, "seed": SEED,
                    "device": DEVICE,
                    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
                    "precision": "fp16 autocast",
                    "selection": "validation F1, per-epoch checkpoint"},
         "target_prior": target_prior,
         "protocols": {}}

    for proto in ["random", "topic_disjoint", "temporal"]:
        log(f"protocol: {proto}")
        R["protocols"][proto] = run_protocol(frame, proto, tok, target_prior)

    # Truncation sensitivity: the linear model reads whole documents, so check
    # that the 256-token limit is not what drives the transformer's result.
    log("sensitivity: random protocol at max_len=512")
    R["sensitivity_maxlen512_random"] = run_protocol(
        frame, "random", tok, target_prior, max_len=512)

    path = RESULTS / "transformer.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    log(f"DONE -> {path}")


if __name__ == "__main__":
    main()
