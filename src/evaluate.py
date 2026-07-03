"""Metrics and figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve,
)

import config
from src.models import positive_scores


def compute_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(auc(fpr, tpr)),
    }


def plot_confusion_matrix(model, X, y_true, path):
    cm = confusion_matrix(y_true, model.predict(X))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=config.CLASS_NAMES)
    ax.set_yticks([0, 1], labels=config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(model, X, y_true, path):
    y_score = positive_scores(model, X)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_features(model, vectorizer, path, top_n=20):
    # Only linear models expose coef_. Positive weight -> fake, negative -> real.
    if not hasattr(model, "coef_"):
        return False
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_.ravel()
    top_fake = np.argsort(coefs)[-top_n:][::-1]
    top_real = np.argsort(coefs)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].barh(range(top_n), coefs[top_real][::-1], color="#2ca02c")
    axes[0].set_yticks(range(top_n), labels=feature_names[top_real][::-1])
    axes[0].set_title(f"Top {top_n} words -> REAL")
    axes[0].set_xlabel("coefficient")
    axes[1].barh(range(top_n), coefs[top_fake][::-1], color="#d62728")
    axes[1].set_yticks(range(top_n), labels=feature_names[top_fake][::-1])
    axes[1].set_title(f"Top {top_n} words -> FAKE")
    axes[1].set_xlabel("coefficient")
    fig.suptitle("Most informative words")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
