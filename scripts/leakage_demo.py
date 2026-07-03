"""Quick check on why we strip the Reuters source tag.

Trains the same Logistic Regression twice on the same split - once keeping the
Reuters datelines and once removing them - and prints both test scores. Keeping
the tag makes the model look better only because it learns the source, not the
actual difference between real and fake news.

Run: python scripts/leakage_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data_prep import load_raw, build_clean_frame, split_70_10_20  # noqa: E402
from src.features import build_vectorizer  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def run_once(strip_artifacts):
    original = config.STRIP_SOURCE_ARTIFACTS
    config.STRIP_SOURCE_ARTIFACTS = strip_artifacts
    try:
        clean = build_clean_frame(load_raw())
        (Xtr, ytr), _, (Xte, yte) = split_70_10_20(clean)
        vec = build_vectorizer()
        clf = LogisticRegression(solver="liblinear", max_iter=1000,
                                 random_state=config.RANDOM_SEED)
        clf.fit(vec.fit_transform(Xtr), ytr)
        return compute_metrics(clf, vec.transform(Xte), yte)
    finally:
        config.STRIP_SOURCE_ARTIFACTS = original


def main():
    leaky = run_once(strip_artifacts=False)
    honest = run_once(strip_artifacts=True)
    print(f"{'metric':<12}{'with Reuters tag':>18}{'tag removed':>15}")
    print("-" * 45)
    for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        print(f"{k:<12}{leaky[k]:>18.4f}{honest[k]:>15.4f}")


if __name__ == "__main__":
    main()
