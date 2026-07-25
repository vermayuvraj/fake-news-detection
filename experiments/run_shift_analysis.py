"""Supplementary experiments: metadata-only baseline and prior-controlled shift.

Two questions the main experiment cannot answer on its own:

1.  How much of the benchmark is solvable from metadata alone? We train on the
    `subject` field only, with the article text removed entirely.

2.  The shift protocols alter the class prior as well as the topic mix, so a
    raw F1 comparison conflates prior shift with genuine loss of discrimination.
    We therefore report, for every protocol: threshold-oracle F1 (the best F1
    any threshold on the model's own scores could achieve), average precision,
    balanced accuracy, and F1 on prior-matched subsamples of the shifted test
    set in which the fake ratio is resampled to match the random-split prior.

Usage:  python experiments/run_shift_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_recall_curve)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from lib import (PrepConfig, build_frame, load_raw, make_models,  # noqa: E402
                 make_vectorizer, scores_positive, split_random,
                 split_temporal, split_topic_disjoint, SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
OUT = {}
raw = load_raw()
primary_frame = build_frame(raw, PrepConfig())


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# 1. Metadata-only baseline: the `subject` field, with no article text at all.
# --------------------------------------------------------------------------- #
log("A: metadata-only baseline (subject field only, no article text)")
meta = raw[["subject", "label"]].copy()
meta["content"] = meta["subject"].fillna("").astype(str).str.lower()
(X_tr, y_tr), _, (X_te, y_te) = split_random(meta.rename(columns={"content": "content"}))
vec = make_vectorizer({"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0,
                       "stop_words": None, "max_features": None})
A = vec.fit_transform(X_tr)
B = vec.transform(X_te)
clf = make_models()["LogisticRegression"]
clf.fit(A, y_tr)
pred = clf.predict(B)
OUT["metadata_only_baseline"] = {
    "features": "subject field only (article text discarded)",
    "vocab": int(A.shape[1]),
    "n_train": int(A.shape[0]), "n_test": int(B.shape[0]),
    "test_f1": float(f1_score(y_te, pred)),
    "test_accuracy": float((pred == np.asarray(y_te)).mean()),
    "test_auc": float(average_precision_score(y_te, scores_positive(clf, B))),
}
log(f"   subject-only test F1 = {OUT['metadata_only_baseline']['test_f1']:.4f}")

# Date separability: the two classes do not span identical date ranges.
r = raw.dropna(subset=["date_parsed"])
OUT["date_ranges"] = {
    "real_min": str(r.loc[r.label == 0, "date_parsed"].min()),
    "real_max": str(r.loc[r.label == 0, "date_parsed"].max()),
    "fake_min": str(r.loc[r.label == 1, "date_parsed"].min()),
    "fake_max": str(r.loc[r.label == 1, "date_parsed"].max()),
    "n_fake_outside_real_window": int(
        ((r.label == 1) &
         ((r.date_parsed < r.loc[r.label == 0, "date_parsed"].min()) |
          (r.date_parsed > r.loc[r.label == 0, "date_parsed"].max()))).sum()),
}

# --------------------------------------------------------------------------- #
# 2. Prior-controlled shift analysis.
# --------------------------------------------------------------------------- #
log("B: prior-controlled shift analysis")
PROTOCOLS = {"random": split_random,
             "topic_disjoint": split_topic_disjoint,
             "temporal": split_temporal}
TARGET_PRIOR = float(primary_frame["label"].mean())   # the random-split prior
OUT["target_prior"] = TARGET_PRIOR
OUT["shift_analysis"] = {}

for proto, splitter in PROTOCOLS.items():
    (Xtr, ytr), _, (Xte, yte) = splitter(primary_frame)
    vec = make_vectorizer()
    Atr = vec.fit_transform(Xtr)
    Ate = vec.transform(Xte)
    yte = np.asarray(yte)
    OUT["shift_analysis"][proto] = {"n_test": int(len(yte)),
                                    "fake_ratio_test": float(yte.mean()),
                                    "models": {}}
    for name in ["LogisticRegression", "PassiveAggressive", "LinearSVC"]:
        model = make_models()[name]
        model.fit(Atr, ytr)
        s = scores_positive(model, Ate)
        pred = model.predict(Ate)

        # Threshold-oracle F1: best F1 achievable by any cutoff on these scores.
        prec, rec, _ = precision_recall_curve(yte, s)
        denom = prec + rec
        f1_curve = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec),
                             where=denom > 0)

        # Prior-matched resampling: keep every positive, subsample negatives so
        # that the fake ratio matches the random-split prior. Repeated 20 times.
        pos = np.flatnonzero(yte == 1)
        neg = np.flatnonzero(yte == 0)
        n_neg_needed = int(round(len(pos) * (1 - TARGET_PRIOR) / TARGET_PRIOR))
        matched = []
        if 0 < n_neg_needed <= len(neg):
            rng = np.random.default_rng(SEED)
            for _ in range(20):
                sub = np.concatenate([pos, rng.choice(neg, n_neg_needed, replace=False)])
                matched.append(f1_score(yte[sub], pred[sub], zero_division=0))

        OUT["shift_analysis"][proto]["models"][name] = {
            "f1_default_threshold": float(f1_score(yte, pred, zero_division=0)),
            "f1_threshold_oracle": float(np.max(f1_curve)),
            "average_precision": float(average_precision_score(yte, s)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "f1_prior_matched_mean": float(np.mean(matched)) if matched else None,
            "f1_prior_matched_std": float(np.std(matched)) if matched else None,
            "n_prior_matched": int(len(pos) + n_neg_needed) if matched else None,
        }
        m = OUT["shift_analysis"][proto]["models"][name]
        log(f"   {proto:<15} {name:<20} F1={m['f1_default_threshold']:.4f} "
            f"oracleF1={m['f1_threshold_oracle']:.4f} "
            f"AP={m['average_precision']:.4f} "
            f"balAcc={m['balanced_accuracy']:.4f} "
            f"F1@matchedPrior={m['f1_prior_matched_mean']:.4f}")

path = RESULTS / "shift_analysis.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2)
log(f"DONE -> {path}")
