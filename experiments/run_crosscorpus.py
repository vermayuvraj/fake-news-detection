"""Cross-corpus transfer: train on ISOT, test on the LIAR benchmark.

This is the strongest generalisation test in the study. The two corpora differ
in almost every surface respect: ISOT contains full articles (~415 tokens) whose
classes come from disjoint publisher sets, whereas LIAR \cite{wang2017liar}
contains short single-sentence political claims (~18 tokens) all fact-checked by
one organisation (PolitiFact), so the provenance-style signal that ISOT models
rely on is absent by construction.

Label mapping. LIAR is six-way; we binarise with the conventional cut used in
the fake-news literature: {pants-fire, false, barely-true} -> fake (1),
{half-true, mostly-true, true} -> real (0). The barely-true/half-true boundary
is the debatable one; we report it explicitly and note that shifting it would
change the LIAR class prior but not the qualitative near-chance result.

We evaluate the released linear models (trained on all of ISOT) and the saved
seed-42 DistilBERT ISOT model. A majority-class baseline is reported for
reference, since on a near-balanced external set the honest question is whether
the transferred model beats "always predict the majority".

Usage:  python experiments/run_crosscorpus.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from lib import (PrepConfig, build_frame, load_raw, make_models,  # noqa: E402
                 make_vectorizer, scores_positive)
from src.text_clean import clean_text  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
LIAR_DIR = HERE.parent / "data" / "liar"
BERT_MODEL = HERE.parent / "models" / "bert_isot_random_seed42"

FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"half-true", "mostly-true", "true"}


def log(m):
    print(m, flush=True)


def load_liar_test():
    df = pd.read_csv(LIAR_DIR / "test.tsv", sep="\t", header=None)
    df = df[[1, 2]].rename(columns={1: "liar_label", 2: "statement"})
    df = df[df["liar_label"].isin(FAKE_LABELS | REAL_LABELS)].copy()
    df["label"] = df["liar_label"].apply(lambda x: 1 if x in FAKE_LABELS else 0)
    df["statement"] = df["statement"].fillna("").astype(str)
    return df


def metrics_block(y, pred, score):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def main():
    R = {}
    liar = load_liar_test()
    y = liar["label"].values
    prior = float(y.mean())
    R["liar"] = {"n_test": int(len(liar)), "fake_ratio": prior,
                 "mean_words": float(liar["statement"].str.split().apply(len).mean()),
                 "label_mapping": {"fake": sorted(FAKE_LABELS),
                                   "real": sorted(REAL_LABELS)}}
    log(f"LIAR test: n={len(liar)}, fake ratio={prior:.4f}, "
        f"mean words={R['liar']['mean_words']:.1f}")

    # Majority-class reference: predict the single most frequent label.
    maj = int(prior >= 0.5)          # 1 (fake) only if fake is the majority
    maj_pred = np.full_like(y, maj)
    R["baselines"] = {
        "majority_class": {
            "predicts": "fake" if maj == 1 else "real",
            "accuracy": float(accuracy_score(y, maj_pred)),
            "f1": float(f1_score(y, maj_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, maj_pred)),
        },
        "random_ranker_ap": prior,   # expected AP of a random scorer
    }
    log(f"majority-class ({R['baselines']['majority_class']['predicts']}): "
        f"acc={R['baselines']['majority_class']['accuracy']:.4f} "
        f"F1={R['baselines']['majority_class']['f1']:.4f} "
        f"balAcc={R['baselines']['majority_class']['balanced_accuracy']:.4f}")

    # ---- Linear models: train on ALL of ISOT, test on LIAR ----------------
    log("\ntraining linear models on all ISOT, testing on LIAR ...")
    isot = build_frame(load_raw(), PrepConfig())
    vec = make_vectorizer()
    X_isot = vec.fit_transform(isot["content"])
    liar_clean = liar["statement"].apply(lambda t: clean_text(t, strip_artifacts=True))
    X_liar = vec.transform(liar_clean)

    R["linear"] = {"n_train_isot": int(len(isot)),
                   "liar_docs_all_zero_features":
                   int((np.asarray((X_liar != 0).sum(axis=1)).ravel() == 0).sum())}
    for name in ["LogisticRegression", "PassiveAggressive"]:
        model = make_models()[name]
        model.fit(X_isot, isot["label"])
        pred = model.predict(X_liar)
        score = scores_positive(model, X_liar)
        R["linear"][name] = metrics_block(y, pred, score)
        log(f"  {name:<20} acc={R['linear'][name]['accuracy']:.4f} "
            f"F1={R['linear'][name]['f1']:.4f} "
            f"AUC={R['linear'][name]['auc_roc']:.4f} "
            f"AP={R['linear'][name]['average_precision']:.4f} "
            f"balAcc={R['linear'][name]['balanced_accuracy']:.4f}")

    # ---- DistilBERT: reuse the saved ISOT model ---------------------------
    if BERT_MODEL.exists():
        log("\nevaluating saved DistilBERT (ISOT, seed 42) on LIAR ...")
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)
        from src.text_clean import strip_source_artifacts
        import re
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL).to(dev)
        model.eval()
        _ws = re.compile(r"\s+")
        texts = [(_ws.sub(" ", strip_source_artifacts(t)).strip())
                 for t in liar["statement"]]
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tok(texts[i:i + 64], truncation=True, max_length=256,
                          padding=True, return_tensors="pt").to(dev)
                with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    logits = model(**enc).logits.float()
                probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        score = np.concatenate(probs)
        pred = (score >= 0.5).astype(int)
        R["distilbert"] = metrics_block(y, pred, score)
        log(f"  DistilBERT           acc={R['distilbert']['accuracy']:.4f} "
            f"F1={R['distilbert']['f1']:.4f} "
            f"AUC={R['distilbert']['auc_roc']:.4f} "
            f"AP={R['distilbert']['average_precision']:.4f} "
            f"balAcc={R['distilbert']['balanced_accuracy']:.4f}")
    else:
        log(f"\n[skip] DistilBERT model not found at {BERT_MODEL}; "
            "run the multi-seed script first, then re-run this.")
        R["distilbert"] = None

    with open(RESULTS / "crosscorpus.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    log("\nDONE -> results/crosscorpus.json")


if __name__ == "__main__":
    main()
