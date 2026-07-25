"""Generates every figure used in the paper as vector PDF.

Usage:  python experiments/make_figures.py
Figures are written to  ../paper/figures/
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from lib import (PrepConfig, build_frame, load_raw, make_models,  # noqa: E402
                 make_vectorizer, scores_positive, split_random,
                 split_topic_disjoint)

FIG = HERE.parent.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = HERE / "results"
R = json.load(open(RES / "results.json", encoding="utf-8"))
S = json.load(open(RES / "shift_analysis.json", encoding="utf-8"))

# Match LaTeX body text: serif family, small sizes, vector output.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = 3.45      # IEEE single-column width in inches
COL2 = 7.16     # full text width
C_REAL, C_FAKE, C_MAIN, C_ALT = "#2f7d32", "#c0392b", "#1f4e79", "#d68910"


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p.name)


# --------------------------------------------------------------------------- #
# Fig. 1: leakage ablation
# --------------------------------------------------------------------------- #
order = ["A1_naive", "A2_dedup_only", "A3_strip_only", "A4_primary"]
labels = ["Naive\n(no mitigation)", "De-duplicated\nonly",
          "Source tag\nstripped only", "Both\n(primary)"]
vals = [R["leakage_ablation"][k]["test"]["f1"] for k in order]
meta_f1 = S["metadata_only_baseline"]["test_f1"]

fig, ax = plt.subplots(figsize=(COL, 2.25))
bars = ax.bar(range(len(vals)), vals, color=[C_ALT, C_ALT, C_ALT, C_MAIN], width=0.62)
ax.axhline(meta_f1, ls="--", lw=0.9, color=C_REAL)
ax.text(len(vals) - 0.5, meta_f1 - 0.0016,
        f"metadata-only baseline (F1 = {meta_f1:.3f})",
        ha="right", va="top", fontsize=6.5, color=C_REAL)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0004, f"{v:.4f}",
            ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylim(0.975, 1.001)
ax.set_ylabel("Test $F_1$")
save(fig, "fig_ablation.pdf")

# --------------------------------------------------------------------------- #
# Fig. 2: distribution shift, prior-controlled
# --------------------------------------------------------------------------- #
protos = ["random", "topic_disjoint", "temporal"]
pnames = ["Random\n(standard)", "Topic-disjoint", "Temporal"]
model = "PassiveAggressive"
series = {
    "$F_1$ (default threshold)": [S["shift_analysis"][p]["models"][model]["f1_default_threshold"] for p in protos],
    "$F_1$ (prior-matched)": [S["shift_analysis"][p]["models"][model]["f1_prior_matched_mean"] for p in protos],
    "Average precision": [S["shift_analysis"][p]["models"][model]["average_precision"] for p in protos],
}
fig, ax = plt.subplots(figsize=(COL, 2.3))
w = 0.26
x = np.arange(len(protos))
for i, (lab, ys) in enumerate(series.items()):
    b = ax.bar(x + (i - 1) * w, ys, w, label=lab,
               color=[C_MAIN, C_ALT, C_REAL][i])
    for bb, v in zip(b, ys):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=5.6, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(pnames, fontsize=6.8)
ax.set_ylim(0.70, 1.06)
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.set_ylabel("Score")
# Legend below the axes so it cannot overlap the shorter bars.
ax.legend(frameon=False, ncol=3, fontsize=6.2,
          loc="upper center", bbox_to_anchor=(0.5, -0.22),
          handlelength=1.1, columnspacing=1.0)
save(fig, "fig_shift.pdf")

# --------------------------------------------------------------------------- #
# Fig. 3: learning curve
# --------------------------------------------------------------------------- #
ks = sorted(R["learning_curve"], key=lambda k: R["learning_curve"][k]["n_train"])
ns = [R["learning_curve"][k]["n_train"] for k in ks]
f1s = [R["learning_curve"][k]["test_f1"] for k in ks]
full = R["leakage_ablation"]["A4_primary"]["test"]["f1"]

fig, ax = plt.subplots(figsize=(COL, 2.1))
ax.semilogx(ns, f1s, "o-", color=C_MAIN, ms=3.2, lw=1.2)
ax.axhline(full, ls=":", lw=0.9, color="grey")
ax.text(ns[-1], full - 0.02, f"full training set ({ns[-1]:,} docs)",
        ha="right", va="top", fontsize=6.2, color="grey")
ax.annotate(f"{f1s[2]:.3f} with only {ns[2]:,} documents",
            xy=(ns[2], f1s[2]), xytext=(ns[2] * 1.5, f1s[2] - 0.13),
            fontsize=6.2, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlabel("Labelled training documents (log scale)")
ax.set_ylabel("Test $F_1$")
ax.set_ylim(0.62, 1.01)
save(fig, "fig_learning_curve.pdf")

# --------------------------------------------------------------------------- #
# Fig. 4: shortcut concentration (top-feature removal)
# --------------------------------------------------------------------------- #
Ks = sorted(R["feature_removal"], key=int)
kx = [int(k) for k in Ks]
ky = [R["feature_removal"][k]["test"]["f1"] for k in Ks]
fig, ax = plt.subplots(figsize=(COL, 2.1))
ax.plot(kx, ky, "s-", color=C_FAKE, ms=3.2, lw=1.2)
for xx, yy in zip(kx, ky):
    ax.text(xx, yy + 0.004, f"{yy:.3f}", ha="center", va="bottom", fontsize=6)
ax.set_xlabel("Number of highest-weight unigrams removed ($K$)")
ax.set_ylabel("Test $F_1$")
ax.set_ylim(0.915, 0.995)
save(fig, "fig_feature_removal.pdf")

# --------------------------------------------------------------------------- #
# Recompute the primary and topic-disjoint fits for the remaining figures.
# --------------------------------------------------------------------------- #
print("refitting for confusion/ROC/coefficient figures ...")
raw = load_raw()
frame = build_frame(raw, PrepConfig())
(Xtr, ytr), _, (Xte, yte) = split_random(frame)
vec = make_vectorizer()
Atr, Ate = vec.fit_transform(Xtr), vec.transform(Xte)
fitted = {}
for name, m in make_models().items():
    m.fit(Atr, ytr)
    fitted[name] = m

# Fig. 5: confusion matrix of the selected model
sel = R["selected_model"]
cm = confusion_matrix(yte, fitted[sel].predict(Ate))
fig, ax = plt.subplots(figsize=(COL * 0.72, 1.95))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], labels=["real", "fake"])
ax.set_yticks([0, 1], labels=["real", "fake"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
save(fig, "fig_confusion.pdf")

# Fig. 6: ROC curves, random vs topic-disjoint
(Xtr2, ytr2), _, (Xte2, yte2) = split_topic_disjoint(frame)
vec2 = make_vectorizer()
Atr2, Ate2 = vec2.fit_transform(Xtr2), vec2.transform(Xte2)
fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, 2.35), sharey=True)
for ax, (Ax, yx, Atrx, ytrx, title) in zip(
        axes, [(Ate, yte, Atr, ytr, "Random split"),
               (Ate2, yte2, Atr2, ytr2, "Topic-disjoint split")]):
    for name, colr in zip(["PassiveAggressive", "LinearSVC",
                           "LogisticRegression", "MultinomialNB"],
                          [C_MAIN, C_ALT, C_REAL, "#7d3c98"]):
        m = make_models()[name]
        m.fit(Atrx, ytrx)
        s = scores_positive(m, Ax)
        fpr, tpr, _ = roc_curve(yx, s)
        ax.plot(fpr, tpr, lw=1.1, color=colr,
                label=f"{name} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right", fontsize=6)
axes[0].set_ylabel("True positive rate")
save(fig, "fig_roc.pdf")

# Fig. 7: most informative coefficients (logistic regression probe)
lr = fitted["LogisticRegression"]
names = np.array(vec.get_feature_names_out())
coefs = lr.coef_.ravel()
K = 15
top_fake = np.argsort(coefs)[-K:]
top_real = np.argsort(coefs)[:K]
fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.78, 2.7))
axes[0].barh(range(K), coefs[top_real], color=C_REAL, height=0.7)
axes[0].set_yticks(range(K), labels=names[top_real], fontsize=6.2)
axes[0].set_title("Evidence for \\textit{real}")
axes[0].set_xlabel("Coefficient")
axes[1].barh(range(K), coefs[top_fake], color=C_FAKE, height=0.7)
axes[1].set_yticks(range(K), labels=names[top_fake], fontsize=6.2)
axes[1].set_title("Evidence for \\textit{fake}")
axes[1].set_xlabel("Coefficient")
for a in axes:
    a.tick_params(axis="y", length=0)
plt.rcParams["text.usetex"] = False
axes[0].set_title("Evidence for 'real'")
axes[1].set_title("Evidence for 'fake'")
save(fig, "fig_coefficients.pdf")

# Fig. 8 (appendix): corpus composition and length distribution
fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.75, 2.1))
ds = R["dataset"]
axes[0].bar(["real", "fake"], [ds["n_real_raw"], ds["n_fake_raw"]],
            color=[C_REAL, C_FAKE], width=0.55)
for i, v in enumerate([ds["n_real_raw"], ds["n_fake_raw"]]):
    axes[0].text(i, v + 250, f"{v:,}", ha="center", fontsize=6.5)
axes[0].set_ylabel("Articles (raw corpus)")
axes[0].set_ylim(0, 26000)

lengths = frame["content"].str.split().apply(len)
cap = int(lengths.quantile(0.98))
axes[1].hist(lengths[frame.label == 0].clip(upper=cap), bins=45, alpha=0.62,
             label="real", color=C_REAL)
axes[1].hist(lengths[frame.label == 1].clip(upper=cap), bins=45, alpha=0.62,
             label="fake", color=C_FAKE)
axes[1].set_xlabel("Tokens per document (cleaned)")
axes[1].set_ylabel("Count")
axes[1].legend(frameon=False)
save(fig, "fig_corpus.pdf")

# --------------------------------------------------------------------------- #
# Fig. 9: TF-IDF vs DistilBERT across protocols (only if the run exists)
# --------------------------------------------------------------------------- #
tpath = RES / "transformer.json"
if tpath.exists():
    T = json.load(open(tpath, encoding="utf-8"))
    protos = ["random", "topic_disjoint", "temporal"]
    pn = ["Random\n(standard)", "Topic-disjoint", "Temporal"]
    lin = [S["shift_analysis"][p]["models"]["PassiveAggressive"] for p in protos]
    trf = [T["protocols"][p]["test"] for p in protos]

    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.74, 2.6), sharex=True)
    x = np.arange(len(protos))
    w = 0.36
    for ax, key, title in zip(
            axes,
            ["average_precision", "f1_prior_matched_mean"],
            ["Average precision", "$F_1$ (prior-matched)"]):
        a = [d[key] for d in lin]
        b = [d[key] for d in trf]
        r1 = ax.bar(x - w / 2, a, w, label="TF-IDF + PA", color=C_MAIN)
        r2 = ax.bar(x + w / 2, b, w, label="DistilBERT", color=C_ALT)
        for rr, vals in ((r1, a), (r2, b)):
            for bb, v in zip(rr, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=5.8,
                        rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(pn, fontsize=6.6)
        # Range must accommodate the transformer's prior-matched collapse (0.689).
        ax.set_ylim(0.60, 1.09)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_title(title)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, ncol=2, fontsize=6.4, loc="upper center",
                   bbox_to_anchor=(1.02, -0.19), handlelength=1.1,
                   columnspacing=1.4)
    save(fig, "fig_transformer.pdf")
else:
    print("skipping fig_transformer.pdf (transformer.json not present yet)")

print("all figures written to", FIG)
