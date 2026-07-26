"""Multi-seed DistilBERT runs to put error bars on the capacity-control result.

Repeats the random and topic-disjoint protocols across five seeds, reusing the
exact data plumbing of run_transformer.py so the only thing that varies is the
seed. Writes per-seed metrics and their mean/std to transformer_multiseed.json,
and saves the seed-42 random-protocol model for reuse by the cross-corpus
experiment (avoids retraining it).

Usage:  python experiments/run_transformer_multiseed.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_transformer as rt  # noqa: E402  (reuse all plumbing)

SEEDS = [42, 0, 1, 2, 3]
PROTOCOLS = ["random", "topic_disjoint"]
RESULTS = HERE / "results"
MODEL_SAVE = HERE.parent / "models" / "bert_isot_random_seed42"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def train_one(frame, proto, tok, target_prior, seed, save_dir=None):
    tr_i, va_i, te_i = rt.SPLITTERS[proto](frame, seed=42)  # fixed split, vary init
    sizes = {"train": len(tr_i), "val": len(va_i), "test": len(te_i)}
    collate = rt.make_collate(tok)
    dl = {}
    for name, idx, shuf in [("train", tr_i, True), ("val", va_i, False),
                            ("test", te_i, False)]:
        ds = rt.TextDS(frame.loc[idx, "bert_text"], frame.loc[idx, "label"],
                       tok, rt.MAX_LEN)
        dl[name] = DataLoader(ds, batch_size=rt.BATCH if shuf else 64,
                              shuffle=shuf, collate_fn=collate)

    rt.set_seeds(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        rt.MODEL_NAME, num_labels=2).to(rt.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=rt.LR)
    total = len(dl["train"]) * rt.EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=rt.LR, total_steps=total, pct_start=0.1,
        anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=(rt.DEVICE == "cuda"))

    best = {"val_f1": -1.0, "state": None}
    for ep in range(1, rt.EPOCHS + 1):
        model.train()
        for batch in dl["train"]:
            batch = {k: v.to(rt.DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(rt.DEVICE == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
        yv, sv = rt.predict(model, dl["val"])
        vf1 = f1_score(yv, (sv >= 0.5).astype(int))
        if vf1 > best["val_f1"]:
            best = {"val_f1": float(vf1),
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    yt, st = rt.predict(model, dl["test"])
    metrics = rt.full_metrics(yt, st, target_prior)
    metrics["val_f1_selected"] = best["val_f1"]

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        log(f"    saved model -> {save_dir}")

    del model, best
    torch.cuda.empty_cache()
    return metrics


def aggregate(runs):
    keys = ["f1_default_threshold", "f1_threshold_oracle", "average_precision",
            "balanced_accuracy", "f1_prior_matched_mean", "auc_roc",
            "precision", "recall", "val_f1_selected"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "min": float(np.min(vals)), "max": float(np.max(vals)),
                  "n": len(vals)}
    return out


def main():
    log(f"device = {rt.DEVICE} ({torch.cuda.get_device_name(0)})")
    raw = rt.load_raw()
    frame = rt.build_aligned_frame(raw)
    target_prior = float(frame["label"].mean())
    tok = AutoTokenizer.from_pretrained(rt.MODEL_NAME)

    R = {"config": {"model": rt.MODEL_NAME, "seeds": SEEDS,
                    "protocols": PROTOCOLS, "epochs": rt.EPOCHS,
                    "max_len": rt.MAX_LEN, "note": "fixed data split (seed 42), "
                    "varied model init/shuffle seed"},
         "target_prior": target_prior, "per_seed": {}, "aggregate": {}}

    for proto in PROTOCOLS:
        R["per_seed"][proto] = {}
        runs = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            save = MODEL_SAVE if (proto == "random" and seed == 42) else None
            m = train_one(frame, proto, tok, target_prior, seed, save_dir=save)
            runs.append(m)
            R["per_seed"][proto][str(seed)] = m
            log(f"  {proto} seed {seed}: F1def={m['f1_default_threshold']:.4f} "
                f"AP={m['average_precision']:.4f} "
                f"F1match={m['f1_prior_matched_mean']:.4f} "
                f"({(time.perf_counter()-t0)/60:.1f} min)")
            # incremental save so partial results survive interruption
            R["aggregate"][proto] = aggregate(runs)
            with open(RESULTS / "transformer_multiseed.json", "w",
                      encoding="utf-8") as f:
                json.dump(R, f, indent=2)
        log(f"[{proto}] AP {R['aggregate'][proto]['average_precision']['mean']:.4f}"
            f" +/- {R['aggregate'][proto]['average_precision']['std']:.4f}  |  "
            f"F1match {R['aggregate'][proto]['f1_prior_matched_mean']['mean']:.4f}"
            f" +/- {R['aggregate'][proto]['f1_prior_matched_mean']['std']:.4f}")

    log("DONE -> results/transformer_multiseed.json")


if __name__ == "__main__":
    main()
