"""Basic EDA plots: class balance and article length. Saves to reports/figures/.

Run: python scripts/eda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.data_prep import load_raw, build_clean_frame


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()

    # Class balance on the raw data.
    counts = raw["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(["real", "fake"], [counts.get(0, 0), counts.get(1, 0)],
           color=["#2ca02c", "#d62728"])
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("articles")
    ax.set_title("Class balance (raw dataset)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # Article length (word count) by class, after cleaning.
    clean = build_clean_frame(raw)
    clean = clean.assign(nwords=clean["content"].str.split().apply(len))
    real_len = clean.loc[clean["label"] == config.LABEL_REAL, "nwords"]
    fake_len = clean.loc[clean["label"] == config.LABEL_FAKE, "nwords"]
    cap = int(clean["nwords"].quantile(0.98))  # trim the long tail for readability

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(real_len.clip(upper=cap), bins=50, alpha=0.6, label="real", color="#2ca02c")
    ax.hist(fake_len.clip(upper=cap), bins=50, alpha=0.6, label="fake", color="#d62728")
    ax.set_xlabel("words per article (cleaned)")
    ax.set_ylabel("count")
    ax.set_title("Article length by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "text_length.png", dpi=150)
    plt.close(fig)

    print("Saved class_distribution.png and text_length.png to reports/figures/")


if __name__ == "__main__":
    main()
