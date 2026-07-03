"""Make a single summary image (metrics + confusion matrix + ROC) for sharing.

Run: python scripts/make_summary_figure.py  ->  reports/figures/summary.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve

import config
from src.data_prep import prepare_data
from src.evaluate import compute_metrics
from src.models import positive_scores


def main():
    (_, _), (_, _), (X_test, y_test) = prepare_data(verbose=False)
    pipeline = joblib.load(config.MODELS_DIR / "fake_news_model.joblib")
    X_test_clean = [t for t in X_test]  # already cleaned in prepare_data

    metrics = compute_metrics(pipeline, X_test_clean, y_test)
    y_pred = pipeline.predict(X_test_clean)
    y_score = positive_scores(pipeline, X_test_clean)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.suptitle("Fake News Detection  ·  TF-IDF + Linear Classifier",
                 fontsize=15, fontweight="bold")

    # Panel 1: metrics bars.
    names = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    vals = [metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["auc_roc"]]
    bars = axes[0].barh(names[::-1], vals[::-1], color="#1f77b4")
    axes[0].set_xlim(0.9, 1.0)
    axes[0].set_title("Test-set metrics")
    for b, v in zip(bars, vals[::-1]):
        axes[0].text(v - 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                     va="center", ha="right", color="white", fontweight="bold")

    # Panel 2: confusion matrix.
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1], labels=config.CLASS_NAMES)
    axes[1].set_yticks([0, 1], labels=config.CLASS_NAMES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    # Panel 3: ROC.
    axes[2].plot(fpr, tpr, color="#1f77b4", label=f"AUC = {roc_auc:.4f}")
    axes[2].plot([0, 1], [0, 1], "--", color="grey")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC curve")
    axes[2].legend(loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = config.FIGURES_DIR / "summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
