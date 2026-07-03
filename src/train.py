"""Train the models, pick the best on validation, evaluate on test, save results.

Split roles: train = fit vectorizer + models, val = choose the best model,
test = report the final numbers once.
"""

import json
import random

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import config
from src import evaluate
from src.data_prep import prepare_data
from src.features import build_vectorizer
from src.models import build_models


def set_seeds():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)


def run():
    set_seeds()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # Fit TF-IDF on the training text only, then transform val/test.
    vectorizer = build_vectorizer()
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)
    print(f"[feat] tfidf vocab size = {len(vectorizer.vocabulary_):,}")

    # Train each candidate and score it on the validation set.
    print("\n[fit] training candidates ...")
    models = build_models()
    val_results, fitted = {}, {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        fitted[name] = model
        val_results[name] = evaluate.compute_metrics(model, Xva, y_val)
        m = val_results[name]
        print(f"   {name:<22} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc_roc']:.4f}")

    # Pick the best by validation F1, then test it once.
    best_name = max(val_results, key=lambda n: val_results[n][config.SELECTION_METRIC])
    best_model = fitted[best_name]
    print(f"\n[pick] best on validation ({config.SELECTION_METRIC}): {best_name}")

    test_metrics = evaluate.compute_metrics(best_model, Xte, y_test)
    print("[test] final metrics:")
    for k, v in test_metrics.items():
        print(f"       {k:<10} {v:.4f}")

    y_test_pred = best_model.predict(Xte)
    report_txt = classification_report(
        y_test, y_test_pred, target_names=config.CLASS_NAMES, digits=4)
    print("\n" + report_txt)

    evaluate.plot_confusion_matrix(
        best_model, Xte, y_test, config.FIGURES_DIR / "confusion_matrix.png")
    evaluate.plot_roc_curve(
        best_model, Xte, y_test, config.FIGURES_DIR / "roc_curve.png")
    made_feat_plot = evaluate.plot_top_features(
        best_model, vectorizer, config.FIGURES_DIR / "top_features.png")

    # Save the fitted pipeline so predict.py can reuse it.
    pipeline = Pipeline([("tfidf", vectorizer), ("clf", best_model)])
    model_path = config.MODELS_DIR / "fake_news_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n[save] model -> {model_path}")

    results = {
        "selected_model": best_name,
        "selection_metric": config.SELECTION_METRIC,
        "validation_comparison": val_results,
        "test_metrics": test_metrics,
        "test_classification_report": report_txt,
        "data": {
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
            "split": {"train": config.TRAIN_FRAC,
                      "val": config.VAL_FRAC,
                      "test": config.TEST_FRAC},
        },
        "config_snapshot": {
            "random_seed": config.RANDOM_SEED,
            "tfidf_params": config.TFIDF_PARAMS,
            "min_clean_chars": config.MIN_CLEAN_CHARS,
            "drop_duplicates": config.DROP_DUPLICATES,
            "strip_source_artifacts": config.STRIP_SOURCE_ARTIFACTS,
            "label_convention": {"fake": config.LABEL_FAKE, "real": config.LABEL_REAL},
        },
        "figures": {
            "confusion_matrix": "reports/figures/confusion_matrix.png",
            "roc_curve": "reports/figures/roc_curve.png",
            "top_features": ("reports/figures/top_features.png"
                             if made_feat_plot else None),
        },
    }
    metrics_path = config.REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[save] metrics -> {metrics_path}")

    return results


if __name__ == "__main__":
    run()
