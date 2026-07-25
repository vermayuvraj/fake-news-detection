"""Runs every experiment reported in the paper and writes results/results.json.

Usage:  python experiments/run_experiments.py

The script is deterministic (single seed, fixed splits, deterministic solvers),
so re-running it reproduces the numbers used in the manuscript.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import lib  # noqa: E402
from lib import (PrepConfig, build_frame, evaluate, load_raw, make_models,  # noqa: E402
                 make_vectorizer, run_once, split_random, split_temporal,
                 split_topic_disjoint, scores_positive, SEED)

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)
R = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(str(type(o)))


# --------------------------------------------------------------------------- #
log("Loading raw data")
raw = load_raw()

# 1. Dataset statistics -------------------------------------------------------
log("E1: dataset statistics")
reuters_real = raw.loc[raw.label == 1 - 1, "text"]  # label 0 = real
real_txt = raw.loc[raw.label == 0, "text"].fillna("")
fake_txt = raw.loc[raw.label == 1, "text"].fillna("")
primary_frame = build_frame(raw, PrepConfig(name="primary"))
lengths = primary_frame["content"].str.split().apply(len)

R["dataset"] = {
    "n_real_raw": int((raw.label == 0).sum()),
    "n_fake_raw": int((raw.label == 1).sum()),
    "n_raw_total": int(len(raw)),
    "real_subjects": raw.loc[raw.label == 0, "subject"].value_counts().to_dict(),
    "fake_subjects": raw.loc[raw.label == 1, "subject"].value_counts().to_dict(),
    "pct_real_with_reuters": float(real_txt.str.contains(r"\(Reuters\)", regex=True).mean() * 100),
    "pct_fake_with_reuters": float(fake_txt.str.contains(r"\(Reuters\)", regex=True).mean() * 100),
    "dup_bodies_fake": int(fake_txt.duplicated().sum()),
    "dup_bodies_real": int(real_txt.duplicated().sum()),
    "n_short_fake_bodies": int((fake_txt.str.len() < 50).sum()),
    "n_after_clean_dedup": int(len(primary_frame)),
    "fake_ratio_after": float(primary_frame.label.mean()),
    "date_min": str(raw.date_parsed.min()), "date_max": str(raw.date_parsed.max()),
    "words_mean": float(lengths.mean()), "words_median": float(lengths.median()),
    "words_mean_real": float(lengths[primary_frame.label == 0].mean()),
    "words_mean_fake": float(lengths[primary_frame.label == 1].mean()),
}

# 2. Primary result: all four models, random 70/10/20 -------------------------
log("E2: primary run (4 models)")
primary, objs = run_once(
    primary_frame, splitter=split_random,
    model_names=tuple(make_models().keys()), return_objects=True)
R["primary"] = primary

sel = max(primary["models"], key=lambda m: primary["models"][m]["val"]["f1"])
R["selected_model"] = sel
log(f"    selected by validation F1: {sel}")

# 3. Bootstrap confidence intervals on the test set --------------------------
log("E3: bootstrap 95% CI (2000 resamples)")
y_te = np.asarray(objs["y_te"])
best = objs["models"][sel]
pred_te = best.predict(objs["Xte"])
score_te = scores_positive(best, objs["Xte"])
rng = np.random.default_rng(SEED)
B = 2000
boot = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc_roc": []}
for _ in range(B):
    idx = rng.integers(0, len(y_te), len(y_te))
    yt, yp, ys = y_te[idx], pred_te[idx], score_te[idx]
    if yt.min() == yt.max():
        continue
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    boot["accuracy"].append((tp + tn) / len(yt))
    boot["precision"].append(p)
    boot["recall"].append(r)
    boot["f1"].append(2 * p * r / (p + r) if p + r else 0.0)
    boot["auc_roc"].append(roc_auc_score(yt, ys))
R["bootstrap_ci"] = {
    k: {"mean": float(np.mean(v)),
        "lo95": float(np.percentile(v, 2.5)),
        "hi95": float(np.percentile(v, 97.5))}
    for k, v in boot.items()
}

# 4. McNemar tests between the top models ------------------------------------
log("E4: McNemar tests")
preds = {m: objs["models"][m].predict(objs["Xte"]) for m in objs["models"]}
def mcnemar(a, b):
    """Exact McNemar test on the discordant pairs of two classifiers."""
    a_ok, b_ok = (a == y_te), (b == y_te)
    n01 = int((~a_ok & b_ok).sum())   # a wrong, b right
    n10 = int((a_ok & ~b_ok).sum())   # a right, b wrong
    n = n01 + n10
    p_exact = float(stats.binomtest(n01, n, 0.5).pvalue) if n else 1.0
    chi2 = ((abs(n01 - n10) - 1) ** 2) / n if n else 0.0
    return {"n01": n01, "n10": n10, "n_discordant": n,
            "chi2_cc": float(chi2), "p_exact": p_exact}
R["mcnemar"] = {
    f"{sel}_vs_LinearSVC": mcnemar(preds[sel], preds["LinearSVC"]),
    f"{sel}_vs_LogisticRegression": mcnemar(preds[sel], preds["LogisticRegression"]),
    f"{sel}_vs_MultinomialNB": mcnemar(preds[sel], preds["MultinomialNB"]),
    "LinearSVC_vs_LogisticRegression": mcnemar(preds["LinearSVC"],
                                               preds["LogisticRegression"]),
}

# 5. 5-fold cross-validation on train+val ------------------------------------
log("E5: 5-fold stratified CV")
(X_tr, y_tr), (X_va, y_va), _ = split_random(primary_frame)
X_cv = pd.concat([X_tr, X_va]).reset_index(drop=True)
y_cv = pd.concat([y_tr, y_va]).reset_index(drop=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv = {m: [] for m in make_models()}
for k, (itr, ite) in enumerate(skf.split(X_cv, y_cv), 1):
    vec = make_vectorizer()
    A = vec.fit_transform(X_cv.iloc[itr]); B_ = vec.transform(X_cv.iloc[ite])
    for name, model in make_models().items():
        model.fit(A, y_cv.iloc[itr])
        cv[name].append(f1_score(y_cv.iloc[ite], model.predict(B_)))
    log(f"    fold {k}/5 done")
R["cv_5fold_f1"] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                        "folds": [float(x) for x in v]} for m, v in cv.items()}

# 6. Leakage ablation (the core experiment) ----------------------------------
log("E6: leakage ablation")
ablation_configs = [
    PrepConfig(name="A0_subject_metadata", include_subject=True,
               strip_reuters=False, dedup=False),
    PrepConfig(name="A1_naive", strip_reuters=False, dedup=False),
    PrepConfig(name="A2_dedup_only", strip_reuters=False, dedup=True),
    PrepConfig(name="A3_strip_only", strip_reuters=True, dedup=False),
    PrepConfig(name="A4_primary", strip_reuters=True, dedup=True),
]
R["leakage_ablation"] = {}
for cfg in ablation_configs:
    fr = build_frame(raw, cfg)
    res = run_once(fr, model_names=("LogisticRegression",))
    R["leakage_ablation"][cfg.name] = {
        "n_documents": int(len(fr)),
        "config": {"include_subject": cfg.include_subject,
                   "strip_reuters": cfg.strip_reuters, "dedup": cfg.dedup},
        **res["models"]["LogisticRegression"],
        "sizes": res["sizes"], "vocab": res["vocab"],
    }
    log(f"    {cfg.name}: test F1 = "
        f"{res['models']['LogisticRegression']['test']['f1']:.4f}")

# Quantify duplicate contamination in the no-dedup condition.
fr_nodedup = build_frame(raw, PrepConfig(name="x", strip_reuters=True, dedup=False))
(Xtr_d, _), _, (Xte_d, _) = split_random(fr_nodedup)
train_set = set(Xtr_d)
n_leaked = int(sum(1 for t in Xte_d if t in train_set))
R["duplicate_contamination"] = {
    "n_test": int(len(Xte_d)), "n_test_also_in_train": n_leaked,
    "pct_test_contaminated": float(100 * n_leaked / len(Xte_d)),
}
log(f"    duplicate contamination: {n_leaked}/{len(Xte_d)} test docs also in train")

# 7. Field ablation -----------------------------------------------------------
log("E7: field ablation (title / body / both)")
R["field_ablation"] = {}
for fields in ["title", "body", "both"]:
    fr = build_frame(raw, PrepConfig(name=fields, fields=fields))
    res = run_once(fr, model_names=("LogisticRegression",))
    R["field_ablation"][fields] = {
        "n_documents": int(len(fr)), "vocab": res["vocab"],
        **res["models"]["LogisticRegression"]}
    log(f"    {fields}: test F1 = {res['models']['LogisticRegression']['test']['f1']:.4f}")

# 8. Representation ablation --------------------------------------------------
log("E8: representation ablation")
rep_variants = {
    "unigram": {"ngram_range": (1, 1)},
    "uni+bigram": {"ngram_range": (1, 2)},
    "uni+bi+trigram": {"ngram_range": (1, 3)},
    "vocab_5k": {"max_features": 5_000},
    "vocab_10k": {"max_features": 10_000},
    "vocab_25k": {"max_features": 25_000},
    "vocab_unlimited": {"max_features": None},
    "no_sublinear_tf": {"sublinear_tf": False},
    "keep_stopwords": {"stop_words": None},
}
R["representation_ablation"] = {}
for name, params in rep_variants.items():
    res = run_once(primary_frame, tfidf_params=params,
                   model_names=("LogisticRegression",))
    R["representation_ablation"][name] = {
        "params": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in params.items()},
        "vocab": res["vocab"], **res["models"]["LogisticRegression"]}
    log(f"    {name}: vocab={res['vocab']:,} test F1 = "
        f"{res['models']['LogisticRegression']['test']['f1']:.4f}")

# 9. Distribution shift -------------------------------------------------------
log("E9: distribution shift (random / topic-disjoint / temporal)")
R["shift"] = {}
for name, splitter in [("random", split_random),
                       ("topic_disjoint", split_topic_disjoint),
                       ("temporal", split_temporal)]:
    res = run_once(primary_frame, splitter=splitter,
                   model_names=("LogisticRegression", "PassiveAggressive",
                                "LinearSVC", "MultinomialNB"))
    R["shift"][name] = res
    log(f"    {name}: LR test F1 = {res['models']['LogisticRegression']['test']['f1']:.4f}"
        f"  sizes={res['sizes']}")

# Class balance of each shift protocol's test set (for honest reporting).
R["shift_balance"] = {}
for name, splitter in [("random", split_random),
                       ("topic_disjoint", split_topic_disjoint),
                       ("temporal", split_temporal)]:
    _, _, (Xs, ys) = splitter(primary_frame)
    R["shift_balance"][name] = {"n_test": int(len(ys)),
                                "fake_ratio_test": float(np.mean(ys))}

# 10. Shortcut concentration: remove the top-K most predictive unigrams -------
log("E10: top-feature removal probe")
vec0 = make_vectorizer()
A0 = vec0.fit_transform(X_tr)
lr0 = make_models()["LogisticRegression"]; lr0.fit(A0, y_tr)
names0 = np.array(vec0.get_feature_names_out())
order = np.argsort(-np.abs(lr0.coef_.ravel()))
ranked_unigrams = [names0[i] for i in order if " " not in names0[i]]
R["feature_removal"] = {}
for K in [0, 10, 50, 100, 500, 1000]:
    removed = ranked_unigrams[:K]
    res = run_once(primary_frame, model_names=("LogisticRegression",),
                   extra_stopwords=removed if K else None)
    R["feature_removal"][str(K)] = {
        "vocab": res["vocab"], **res["models"]["LogisticRegression"]}
    log(f"    K={K}: test F1 = {res['models']['LogisticRegression']['test']['f1']:.4f}")
R["top_unigrams_removed_first_20"] = ranked_unigrams[:20]

# 11. Learning curve ----------------------------------------------------------
log("E11: learning curve")
R["learning_curve"] = {}
vec_lc = make_vectorizer(); A_lc = vec_lc.fit_transform(X_tr)
_, _, (X_te_lc, y_te_lc) = split_random(primary_frame)
B_lc = vec_lc.transform(X_te_lc)
rng2 = np.random.default_rng(SEED)
y_tr_arr = np.asarray(y_tr)
for frac in [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]:
    n = max(50, int(frac * A_lc.shape[0]))
    idx = rng2.permutation(A_lc.shape[0])[:n]
    m = make_models()["LogisticRegression"]
    m.fit(A_lc[idx], y_tr_arr[idx])
    R["learning_curve"][f"{frac:.3f}"] = {
        "n_train": int(n),
        "test_f1": float(f1_score(y_te_lc, m.predict(B_lc))),
        "test_auc": float(roc_auc_score(y_te_lc, scores_positive(m, B_lc))),
    }
    log(f"    n={n}: test F1 = {R['learning_curve'][f'{frac:.3f}']['test_f1']:.4f}")

# 12. Label-permutation control ----------------------------------------------
log("E12: label-permutation control")
perm_frame = primary_frame.copy()
perm_frame["label"] = np.random.default_rng(SEED).permutation(perm_frame["label"].values)
res_perm = run_once(perm_frame, model_names=("LogisticRegression",))
R["label_permutation_control"] = res_perm["models"]["LogisticRegression"]
log(f"    permuted-label test AUC = "
    f"{res_perm['models']['LogisticRegression']['test']['auc_roc']:.4f} (expect ~0.5)")

# 13. Runtime, complexity and artefact size ----------------------------------
log("E13: runtime and inference cost")
import joblib
from sklearn.pipeline import Pipeline
pipe = Pipeline([("tfidf", objs["vec"]), ("clf", best)])
tmp = RESULTS_DIR / "_size_probe.joblib"
joblib.dump(pipe, tmp)
size_mb = tmp.stat().st_size / 1024 ** 2
tmp.unlink()

_, _, (X_te_raw, _) = split_random(primary_frame)
sample = list(X_te_raw)[:1000]
t0 = time.perf_counter(); pipe.predict(sample); t_batch = time.perf_counter() - t0
t0 = time.perf_counter()
for s in sample[:200]:
    pipe.predict([s])
t_single = (time.perf_counter() - t0) / 200

R["runtime"] = {
    "vectorize_seconds": primary["vectorize_seconds"],
    "fit_seconds_per_model": {m: primary["models"][m]["fit_seconds"]
                              for m in primary["models"]},
    "model_size_mb": round(size_mb, 2),
    "batch_1000_seconds": round(t_batch, 3),
    "throughput_docs_per_sec_batch": round(1000 / t_batch, 1),
    "single_doc_latency_ms": round(t_single * 1000, 3),
    "nnz_train": primary["nnz_train"], "density_train": primary["density_train"],
    "avg_nnz_per_doc": round(primary["nnz_train"] / primary["sizes"]["train"], 1),
}

# 14. Environment -------------------------------------------------------------
R["environment"] = {
    "python": platform.python_version(),
    "numpy": np.__version__, "scipy": scipy.__version__,
    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    "seed": SEED,
}

out = RESULTS_DIR / "results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(R, f, indent=2, default=jsonable)
log(f"DONE -> {out}")
