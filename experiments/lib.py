"""Configurable pipeline used by the paper's experiments.

This module deliberately re-uses the project's own cleaning function
(`src.text_clean.clean_text`) so that every experiment reported in the paper
runs the same preprocessing code as the released pipeline. Everything else
(field selection, de-duplication, splitting protocol, vectoriser settings) is
exposed as a parameter so that a single ablation axis can be varied at a time.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, precision_score,
    recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_clean import clean_text  # noqa: E402  (single source of truth)

DATA_DIR = PROJECT_ROOT / "data"
SEED = 42
LABEL_REAL, LABEL_FAKE = 0, 1

# Subject values, used by the topic-disjoint shift protocol.
REAL_SUBJECTS = ["politicsNews", "worldnews"]
FAKE_SUBJECTS = ["News", "politics", "left-news",
                 "Government News", "US_News", "Middle-east"]

DEFAULT_TFIDF = dict(
    ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=50_000,
    sublinear_tf=True, strip_accents="unicode", stop_words="english",
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_raw():
    """Load both CSVs into one labelled frame (title, text, subject, date, label)."""
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")
    fake["label"] = LABEL_FAKE
    true["label"] = LABEL_REAL
    df = pd.concat([fake, true], ignore_index=True)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    return df


@dataclass
class PrepConfig:
    """One point in the preprocessing design space."""
    fields: str = "both"          # 'title' | 'body' | 'both'
    strip_reuters: bool = True    # remove the (Reuters) source artefact
    dedup: bool = True            # drop duplicate cleaned documents
    include_subject: bool = False # append the `subject` metadata to the text
    min_chars: int = 20
    name: str = "primary"


def build_frame(df, cfg: PrepConfig):
    """Apply a PrepConfig to the raw frame, returning columns [content, label, ...]."""
    title = df["title"].fillna("").astype(str)
    body = df["text"].fillna("").astype(str)
    if cfg.fields == "title":
        raw = title
    elif cfg.fields == "body":
        raw = body
    else:
        raw = title + " " + body

    if cfg.include_subject:
        raw = df["subject"].fillna("").astype(str) + " " + raw

    content = raw.apply(lambda t: clean_text(t, strip_artifacts=cfg.strip_reuters))

    out = pd.DataFrame({
        "content": content,
        "label": df["label"].values,
        "subject": df["subject"].values,
        "date": df["date_parsed"].values,
    })
    out = out[out["content"].str.len() >= cfg.min_chars]
    if cfg.dedup:
        out = out.drop_duplicates(subset="content", keep="first")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splitting protocols
# --------------------------------------------------------------------------- #
def split_random(frame, seed=SEED):
    """Stratified 70/10/20, matching the released pipeline exactly."""
    X, y = frame["content"], frame["label"]
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.10 / 0.80, stratify=y_tmp, random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def split_topic_disjoint(frame, seed=SEED):
    """Train and test drawn from disjoint topic/subject pools.

    Real articles: politicsNews -> train pool, worldnews -> test pool.
    Fake articles: News/politics/left-news -> train pool, the remaining three
    subjects -> test pool. No subject appears on both sides, so the classifier
    cannot rely on topic-specific vocabulary shared between the two splits.
    """
    train_subj = {"politicsNews", "News", "politics", "left-news"}
    test_subj = {"worldnews", "Government News", "US_News", "Middle-east"}
    tr = frame[frame["subject"].isin(train_subj)]
    te = frame[frame["subject"].isin(test_subj)]
    # Carve a validation slice out of the training pool only.
    X_tr, X_va, y_tr, y_va = train_test_split(
        tr["content"], tr["label"], test_size=0.125,
        stratify=tr["label"], random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (te["content"], te["label"])


def split_temporal(frame, seed=SEED, overlap_only=True):
    """Chronological split: earliest articles train, latest articles test.

    Restricted by default to the window in which both classes are present, so
    that the temporal test set is not trivially dominated by one class.
    """
    f = frame.dropna(subset=["date"]).copy()
    if overlap_only:
        lo = max(f.loc[f.label == LABEL_REAL, "date"].min(),
                 f.loc[f.label == LABEL_FAKE, "date"].min())
        hi = min(f.loc[f.label == LABEL_REAL, "date"].max(),
                 f.loc[f.label == LABEL_FAKE, "date"].max())
        f = f[(f["date"] >= lo) & (f["date"] <= hi)]
    f = f.sort_values("date", kind="mergesort").reset_index(drop=True)
    n = len(f)
    n_tr, n_va = int(0.70 * n), int(0.10 * n)
    tr, va, te = f.iloc[:n_tr], f.iloc[n_tr:n_tr + n_va], f.iloc[n_tr + n_va:]
    return ((tr["content"], tr["label"]), (va["content"], va["label"]),
            (te["content"], te["label"]))


# --------------------------------------------------------------------------- #
# Features and models
# --------------------------------------------------------------------------- #
def make_vectorizer(tfidf_params=None, extra_stopwords=None):
    params = dict(DEFAULT_TFIDF)
    if tfidf_params:
        params.update(tfidf_params)
    if extra_stopwords:
        base = ENGLISH_STOP_WORDS if params.get("stop_words") == "english" else frozenset()
        params["stop_words"] = list(base | set(extra_stopwords))
    return TfidfVectorizer(**params)


def make_models(seed=SEED):
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1000, random_state=seed),
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LinearSVC": LinearSVC(C=1.0, random_state=seed),
        "PassiveAggressive": PassiveAggressiveClassifier(
            C=1.0, max_iter=1000, random_state=seed),
    }


def scores_positive(model, X):
    """Score for the positive (fake) class: probability if available, else margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    s = scores_positive(model, X)
    fpr, tpr, _ = roc_curve(y, s)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc_roc": float(auc(fpr, tpr)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)),
    }


def run_once(frame, splitter=split_random, tfidf_params=None,
             model_names=("LogisticRegression",), extra_stopwords=None,
             seed=SEED, return_objects=False):
    """Fit vectoriser on train, train the requested models, score val and test."""
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splitter(frame, seed)
    vec = make_vectorizer(tfidf_params, extra_stopwords)
    t0 = time.perf_counter()
    Xtr = vec.fit_transform(X_tr)
    t_vec = time.perf_counter() - t0
    Xva, Xte = vec.transform(X_va), vec.transform(X_te)

    all_models = make_models(seed)
    out = {"sizes": {"train": int(Xtr.shape[0]), "val": int(Xva.shape[0]),
                     "test": int(Xte.shape[0])},
           "vocab": int(Xtr.shape[1]), "vectorize_seconds": round(t_vec, 3),
           "nnz_train": int(Xtr.nnz),
           "density_train": float(Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1])),
           "models": {}}

    fitted = {}
    for name in model_names:
        model = all_models[name]
        t0 = time.perf_counter()
        model.fit(Xtr, y_tr)
        t_fit = time.perf_counter() - t0
        fitted[name] = model
        out["models"][name] = {
            "val": evaluate(model, Xva, y_va),
            "test": evaluate(model, Xte, y_te),
            "fit_seconds": round(t_fit, 3),
        }
    if return_objects:
        return out, dict(vec=vec, models=fitted, Xtr=Xtr, Xva=Xva, Xte=Xte,
                         y_tr=y_tr, y_va=y_va, y_te=y_te)
    return out
