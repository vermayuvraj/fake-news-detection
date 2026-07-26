"""Cross-checks numeric claims in main.tex against the experiment result files.

Every quantitative statement in the manuscript should trace to
experiments/results/{results,shift_analysis}.json. This script asserts that the
values actually present in the LaTeX source match those files, so a stale edit
cannot silently desynchronise the paper from the experiments.

Usage:  python verify_numbers.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = (HERE / "main.tex").read_text(encoding="utf-8")
RES = HERE.parent / "Yuvraj_Verma" / "experiments" / "results"
R = json.load(open(RES / "results.json", encoding="utf-8"))
S = json.load(open(RES / "shift_analysis.json", encoding="utf-8"))

ok, bad = [], []


def present(label, expected, fmt="{:.4f}"):
    """Assert that a formatted value literally appears in the manuscript."""
    s = fmt.format(expected) if not isinstance(expected, str) else expected
    if s in TEX:
        ok.append(f"{label}: {s}")
    else:
        bad.append(f"{label}: expected '{s}' in main.tex - NOT FOUND")


sel = R["selected_model"]
pri = R["primary"]["models"]
te = pri[sel]["test"]

# --- headline result ------------------------------------------------------
present("test accuracy", te["accuracy"])
present("test precision", te["precision"])
present("test recall", te["recall"])
present("test F1", te["f1"])
present("test AUC", te["auc_roc"])
present("confusion TP", f"{te['tp']:,}".replace(",", "{,}"))
present("confusion TN", f"{te['tn']:,}".replace(",", "{,}"))
present("confusion FP", str(te["fp"]))
present("confusion FN", str(te["fn"]))

# --- all four models, validation + test ----------------------------------
for m in pri:
    present(f"{m} val F1", pri[m]["val"]["f1"])
    present(f"{m} test F1", pri[m]["test"]["f1"])
    present(f"{m} test AUC", pri[m]["test"]["auc_roc"])

# --- bootstrap CI ---------------------------------------------------------
for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
    ci = R["bootstrap_ci"][k]
    present(f"CI {k} lo", f"{ci['lo95']:.4f}".lstrip("0"))
    present(f"CI {k} hi", f"{ci['hi95']:.4f}".lstrip("0"))

# --- cross-validation -----------------------------------------------------
for m, v in R["cv_5fold_f1"].items():
    present(f"CV {m} mean", v["mean"])
    present(f"CV {m} sd", f"{v['std']:.4f}".lstrip("0"))

# --- dataset --------------------------------------------------------------
d = R["dataset"]
present("n real", f"{d['n_real_raw']:,}".replace(",", "{,}"))
present("n fake", f"{d['n_fake_raw']:,}".replace(",", "{,}"))
present("n raw total", f"{d['n_raw_total']:,}".replace(",", "{,}"))
present("n after clean", f"{d['n_after_clean_dedup']:,}".replace(",", "{,}"))
present("fake ratio", f"{d['fake_ratio_after']:.3f}")
present("pct real reuters", f"{d['pct_real_with_reuters']:.2f}")
present("dup fake bodies", f"{d['dup_bodies_fake']:,}".replace(",", "{,}"))
present("short fake bodies", str(d["n_short_fake_bodies"]))
present("mean tokens real", f"{d['words_mean_real']:.1f}")
present("mean tokens fake", f"{d['words_mean_fake']:.1f}")
for k, v in d["real_subjects"].items():
    present(f"subject {k}", f"{v:,}".replace(",", "{,}"))
for k, v in d["fake_subjects"].items():
    present(f"subject {k}", f"{v:,}".replace(",", "{,}"))

# --- split sizes ----------------------------------------------------------
for proto in ["random", "topic_disjoint", "temporal"]:
    sz = R["shift"][proto]["sizes"]
    for part in ["train", "val", "test"]:
        present(f"{proto} {part}", f"{sz[part]:,}".replace(",", "{,}"))
    present(f"{proto} prior",
            f"{R['shift_balance'][proto]['fake_ratio_test']:.3f}")

# --- leakage ablation -----------------------------------------------------
for key in ["A1_naive", "A2_dedup_only", "A3_strip_only", "A4_primary"]:
    a = R["leakage_ablation"][key]["test"]
    present(f"ablation {key} F1", a["f1"])
    present(f"ablation {key} acc", a["accuracy"])
dc = R["duplicate_contamination"]
present("dup contamination n", f"{dc['n_test_also_in_train']:,}".replace(",", "{,}"))
present("dup contamination pct", f"{dc['pct_test_contaminated']:.2f}")
present("metadata-only F1", f"{S['metadata_only_baseline']['test_f1']:.4f}")

# --- representation and field ablation ------------------------------------
for k, v in R["representation_ablation"].items():
    present(f"repr {k} F1", v["test"]["f1"])
    present(f"repr {k} vocab", f"{v['vocab']:,}".replace(",", "{,}"))
for k, v in R["field_ablation"].items():
    present(f"field {k} F1", v["test"]["f1"])

# --- shift analysis -------------------------------------------------------
for proto, pv in S["shift_analysis"].items():
    for m, mv in pv["models"].items():
        present(f"shift {proto} {m} F1def", mv["f1_default_threshold"])
        present(f"shift {proto} {m} F1orc", mv["f1_threshold_oracle"])
        present(f"shift {proto} {m} AP", mv["average_precision"])
        if mv["f1_prior_matched_mean"] is not None:
            present(f"shift {proto} {m} F1match", mv["f1_prior_matched_mean"])

# --- feature removal and learning curve -----------------------------------
for K, v in R["feature_removal"].items():
    present(f"removal K={K} F1", v["test"]["f1"])
for frac, v in R["learning_curve"].items():
    present(f"LC n={v['n_train']} F1", v["test_f1"])
    present(f"LC n={v['n_train']} n", f"{v['n_train']:,}".replace(",", "{,}"))

# --- runtime --------------------------------------------------------------
rt = R["runtime"]
present("model size", f"{rt['model_size_mb']:.2f}")
present("throughput", f"{rt['throughput_docs_per_sec_batch']:,.0f}".replace(",", "{,}"))
present("latency", f"{rt['single_doc_latency_ms']:.2f}")
present("avg nnz", f"{rt['avg_nnz_per_doc']:.1f}")
present("vectorize s", f"{rt['vectorize_seconds']:.2f}")

# --- control --------------------------------------------------------------
present("permutation AUC", R["label_permutation_control"]["test"]["auc_roc"])

# --- transformer capacity control -----------------------------------------
TPATH = RES / "transformer.json"
if TPATH.exists():
    T = json.load(open(TPATH, encoding="utf-8"))
    for proto, pv in T["protocols"].items():
        t = pv["test"]
        for k in ["precision", "recall", "f1_default_threshold",
                  "f1_threshold_oracle", "average_precision",
                  "balanced_accuracy", "f1_prior_matched_mean"]:
            present(f"bert {proto} {k}", t[k])
        present(f"bert {proto} val F1", pv["val_f1_selected"])
        present(f"bert {proto} AUC", t["auc_roc"])
    s = T["sensitivity_maxlen512_random"]["test"]
    present("bert 512 F1", s["f1_default_threshold"])
    # Linear-model validation F1 per protocol, quoted in the comparison table.
    for proto in ["random", "topic_disjoint", "temporal"]:
        present(f"PA {proto} val F1",
                R["shift"][proto]["models"]["PassiveAggressive"]["val"]["f1"])
        present(f"PA {proto} test AUC",
                R["shift"][proto]["models"]["PassiveAggressive"]["test"]["auc_roc"])
    for proto, pv in T["protocols"].items():
        present(f"bert {proto} train min", f"{pv['train_minutes']:.1f}")
    present("bert 512 train min",
            f"{T['sensitivity_maxlen512_random']['train_minutes']:.1f}")
else:
    print("NOTE: transformer.json absent; capacity-control checks skipped")

# --- multi-seed transformer -----------------------------------------------
MPATH = RES / "transformer_multiseed.json"
if MPATH.exists():
    M = json.load(open(MPATH, encoding="utf-8"))
    for proto in ["random", "topic_disjoint"]:
        agg = M["aggregate"][proto]
        for key in ["average_precision", "f1_prior_matched_mean",
                    "f1_default_threshold", "balanced_accuracy",
                    "f1_threshold_oracle", "auc_roc", "val_f1_selected"]:
            present(f"ms {proto} {key} mean", agg[key]["mean"])
            present(f"ms {proto} {key} std", f"{agg[key]['std']:.4f}".lstrip("0"))
    # best-seed topic AP quoted in the text
    td_aps = [M["per_seed"]["topic_disjoint"][s]["average_precision"]
              for s in M["per_seed"]["topic_disjoint"]]
    import builtins
    present("ms topic best AP", f"{builtins.max(td_aps):.4f}")
else:
    print("NOTE: transformer_multiseed.json absent; multi-seed checks skipped")

# --- cross-corpus transfer (LIAR) -----------------------------------------
CPATH = RES / "crosscorpus.json"
if CPATH.exists():
    CC = json.load(open(CPATH, encoding="utf-8"))
    present("liar n", f"{CC['liar']['n_test']:,}".replace(",", "{,}"))
    present("liar prior", f"{CC['liar']['fake_ratio']:.3f}")
    present("liar words", f"{CC['liar']['mean_words']:.1f}")
    present("liar majority acc", f"{CC['baselines']['majority_class']['accuracy']:.4f}")
    for m in ["LogisticRegression", "PassiveAggressive"]:
        d = CC["linear"][m]
        for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "balanced_accuracy"]:
            present(f"cc {m} {k}", d[k])
    if CC.get("distilbert"):
        for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "balanced_accuracy"]:
            present(f"cc bert {k}", CC["distilbert"][k])
else:
    print("NOTE: crosscorpus.json absent; cross-corpus checks skipped")

# --- derived quantities the paper states ----------------------------------
derived = {
    "mitigation cost (A1-A4)":
        R["leakage_ablation"]["A1_naive"]["test"]["f1"]
        - R["leakage_ablation"]["A4_primary"]["test"]["f1"],
    "dedup cost (A1-A2)":
        R["leakage_ablation"]["A1_naive"]["test"]["f1"]
        - R["leakage_ablation"]["A2_dedup_only"]["test"]["f1"],
    "strip cost (A1-A3)":
        R["leakage_ablation"]["A1_naive"]["test"]["f1"]
        - R["leakage_ablation"]["A3_strip_only"]["test"]["f1"],
    "topic AP drop (PA)":
        S["shift_analysis"]["random"]["models"]["PassiveAggressive"]["average_precision"]
        - S["shift_analysis"]["topic_disjoint"]["models"]["PassiveAggressive"]["average_precision"],
    "topic F1 drop (PA)":
        S["shift_analysis"]["random"]["models"]["PassiveAggressive"]["f1_default_threshold"]
        - S["shift_analysis"]["topic_disjoint"]["models"]["PassiveAggressive"]["f1_default_threshold"],
    "topic F1 drop (LR)":
        S["shift_analysis"]["random"]["models"]["LogisticRegression"]["f1_default_threshold"]
        - S["shift_analysis"]["topic_disjoint"]["models"]["LogisticRegression"]["f1_default_threshold"],
    "LR topic AP drop":
        S["shift_analysis"]["random"]["models"]["LogisticRegression"]["average_precision"]
        - S["shift_analysis"]["topic_disjoint"]["models"]["LogisticRegression"]["average_precision"],
    "deletion K=1000 drop":
        R["feature_removal"]["0"]["test"]["f1"] - R["feature_removal"]["1000"]["test"]["f1"],
    "representation spread":
        max(v["test"]["f1"] for v in R["representation_ablation"].values())
        - min(v["test"]["f1"] for v in R["representation_ablation"].values()),
    "bert topic AP drop":
        T["protocols"]["random"]["test"]["average_precision"]
        - T["protocols"]["topic_disjoint"]["test"]["average_precision"],
    "bert topic F1match drop":
        T["protocols"]["random"]["test"]["f1_prior_matched_mean"]
        - T["protocols"]["topic_disjoint"]["test"]["f1_prior_matched_mean"],
    "LC 543 as pct of full":
        100 * R["learning_curve"]["0.020"]["test_f1"]
        / R["leakage_ablation"]["A4_primary"]["test"]["f1"],
}
print("Derived quantities (verify these appear correctly in the prose):")
for k, v in derived.items():
    print(f"   {k:<28} = {v:.4f}   ({v*100:.2f} points)" if abs(v) < 2
          else f"   {k:<28} = {v:.2f}")

# --- report ---------------------------------------------------------------
print()
print("=" * 66)
print(f"Numeric verification: {len(ok)} matched, {len(bad)} missing")
print("=" * 66)
for b in bad:
    print("  x", b)
print()
print("RESULT:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
