"""Build the comparison table and figures from the CSVs written by run_pipeline.py.

Outputs (in results/):
  test_metrics.csv      val and test metrics per model, 95% bootstrap CIs over prints
  comparison.csv / .md  one row per model: accuracy, CV, LOPO, cost
  figures/*.png         ROC curves, resolution sweep, robustness, CNN training, CV/LOPO

Usage:
  python scripts/make_report.py [--n-boot 2000]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from fdm_defect import config  # noqa: E402
from fdm_defect.evaluate import METRIC_NAMES, bootstrap_ci, group_level  # noqa: E402

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
MODELS = ("logreg", "random_forest", "cnn")
LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest", "cnn": "MobileNetV3-Small"}

# Reference palette (dataviz skill), light mode: categorical slots 1-3 follow
# the model everywhere; markers give a second, colour-free cue.
COLOURS = {"logreg": "#2a78d6", "random_forest": "#eb6834", "cnn": "#1baf7a"}
MARKERS = {"logreg": "o", "random_forest": "s", "cnn": "^"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    """Set matplotlib defaults to the reference palette and recessive axes."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 1.75, "lines.markersize": 6.5, "legend.frameon": False,
        "legend.labelcolor": INK_2, "figure.dpi": 150,
    })


def save(fig, name):
    """Save ``fig`` as ``results/figures/<name>`` and close it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def metrics_with_ci(preds, n_boot, seed):
    """Metrics with print-level bootstrap CIs for every (model, split) in ``preds``."""
    rows = []
    for (model, split), part in preds.groupby(["model", "split"]):
        ci = bootstrap_ci(part["label"], part["proba"], part["print_id"], n_boot=n_boot, seed=seed)
        ci.insert(0, "split", split)
        ci.insert(0, "model", model)
        prints = group_level(part["label"], part["proba"], part["print_id"])
        correct = int(((prints["proba"] >= 0.5) == prints["label"]).sum())
        ci["prints_correct"] = correct
        ci["prints"] = len(prints)
        rows.append(ci)
    return pd.concat(rows, ignore_index=True)


def fmt_ci(row):
    """Format a metric row as ``estimate [lo, hi]``."""
    return f"{row['estimate']:.2f} [{row['lo']:.2f}, {row['hi']:.2f}]"


def comparison_table(metrics, cv, lopo, profiling, fit_times):
    """One row per model: test metrics with CIs, CV, LOPO and cost."""
    prof = profiling.set_index(["model", "metric"])["value"]
    test = metrics[metrics["split"] == "test"].set_index(["model", "metric"])
    rows = []
    for model in MODELS:
        t = lambda m: test.loc[(model, m)]  # noqa: E731
        row = {"Model": LABELS[model]}
        for m, name in [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1"),
                        ("specificity", "Specificity"), ("roc_auc", "ROC-AUC")]:
            row[f"Test {name} [95% CI]"] = fmt_ci(t(m))
        row["Test prints correct"] = f"{t('recall')['prints_correct']}/{t('recall')['prints']}"
        if model in cv["model"].values:
            c = cv[cv["model"] == model]["roc_auc"]
            row["5-fold CV ROC-AUC (mean ± sd)"] = f"{c.mean():.2f} ± {c.std():.2f}"
            lp = lopo[lopo["model"] == model]["roc_auc"]
            row["LOPO ROC-AUC (mean, worst)"] = f"{lp.mean():.2f}, {lp.min():.2f}"
        else:
            row["5-fold CV ROC-AUC (mean ± sd)"] = "not run"
            row["LOPO ROC-AUC (mean, worst)"] = "not run"
        if model == "cnn":
            row["Parameters"] = f"{prof[('cnn', 'parameters')] / 1e6:.2f} M ({prof[('cnn', 'trainable_parameters')] / 1e6:.2f} M trained)"
            row["GFLOP / image"] = f"{prof[('cnn', 'gflops_128px')]:.3f}"
            row["Inference ms / image (batch 1)"] = f"{prof[('cnn', 'predict_time_batch1')]:.1f}"
        else:
            unit = "tree nodes" if model == "random_forest" else "coefficients"
            row["Parameters"] = f"{prof[(model, 'parameters')]:,.0f} {unit}"
            row["GFLOP / image"] = "not counted"
            # Feature extraction is part of the baseline's inference cost.
            ms = prof[("features", "time_per_image")] + prof[(model, "predict_time_batch1")]
            row["Inference ms / image (batch 1)"] = f"{ms:.1f} (incl. features)"
        # 3 significant figures, so tiny values (the 44-coefficient model) don't show as 0.00.
        row["Model size (MB)"] = f"{prof[(model, 'model_size')]:.3g}"
        row["Training time (s)"] = f"{fit_times.set_index('model').loc[model, 'fit_seconds']:.3g}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(table, metrics, path):
    """Write ``table`` with an explanatory header to ``path`` as Markdown."""
    n_boot = int(metrics["n_valid"].min())
    lines = [
        "# Model comparison",
        "",
        "Test split: 8 prints (4 defective), 960 images, scored once per model. Frame-level",
        "metrics at a 0.5 threshold; 95% CIs are percentile bootstrap intervals resampling",
        f"whole prints (at least {n_boot} valid resamples). CV and leave-one-printer-out (LOPO)",
        "use all 52 prints with the 128 px features and were run for the baselines only.",
        "Baseline inference time includes feature extraction; JPEG decoding is excluded for",
        "every model. All timings on the same CPU.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "|".join("---" for _ in table.columns) + "|",
        *("| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_roc(preds):
    """Figure: ROC curves of the three models on the test split."""
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (4, 3)))
    for model in MODELS:
        part = preds[(preds["model"] == model) & (preds["split"] == "test")]
        fpr, tpr, _ = roc_curve(part["label"], part["proba"])
        auc = roc_auc_score(part["label"], part["proba"])
        ax.plot(fpr, tpr, color=COLOURS[model], label=f"{LABELS[model]} (AUC {auc:.2f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (recall)", title="ROC curves, test split")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    save(fig, "roc_test.png")


def fig_sweep(sweep):
    """Figure: validation ROC-AUC and recall against input resolution."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharex=True)
    for ax, metric, title in [(axes[0], "roc_auc", "Validation ROC-AUC"), (axes[1], "recall", "Validation recall")]:
        for model in MODELS:
            s = sweep[sweep["model"] == model].sort_values("size")
            ax.plot(s["size"], s[metric], color=COLOURS[model], marker=MARKERS[model],
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABELS[model])
            last = s.iloc[-1]
            ax.annotate(f"{last[metric]:.2f}", (last["size"], last[metric]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK_2, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sweep["size"].unique()))
        ax.set_xticklabels([f"{s}" for s in sorted(sweep["size"].unique())])
        ax.minorticks_off()
        ax.set(xlabel="Image side length (px)", title=title, ylim=(0, 1.02))
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "resolution_sweep.png")


def condition_label(kind, level):
    """Human-readable label for a corruption and its level."""
    if kind == "clean":
        return "Clean"
    return {"blur": f"Blur σ={level:g} px", "brightness": f"Brightness ×{level:g}",
            "noise": f"Noise σ={level:g}", "jpeg": f"JPEG quality {level:g}"}[kind]


def fig_robustness(rob):
    """Figure: test ROC-AUC and recall under each corruption (dot plot)."""
    rob = rob.copy()
    rob["condition"] = [condition_label(k, lv) for k, lv in zip(rob["corruption"], rob["level"])]
    order = list(dict.fromkeys(rob["condition"]))
    y = {c: i for i, c in enumerate(reversed(order))}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    offsets = {"logreg": 0.18, "random_forest": 0.0, "cnn": -0.18}
    for ax, metric, title in [(axes[0], "roc_auc", "Test ROC-AUC"), (axes[1], "recall", "Test recall")]:
        for model in MODELS:
            r = rob[rob["model"] == model]
            clean = r[r["corruption"] == "clean"][metric].iloc[0]
            ax.axvline(clean, color=COLOURS[model], lw=0.8, alpha=0.5, ls=(0, (2, 2)))
            ax.scatter(r[metric], [y[c] + offsets[model] for c in r["condition"]], s=42,
                       color=COLOURS[model], marker=MARKERS[model], edgecolor=SURFACE, linewidth=1.2,
                       label=LABELS[model], zorder=3)
        ax.set(xlim=(0, 1.02), title=title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(y.values()), list(y.keys()))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.03), ncol=3)
    fig.text(0.55, 0.0, "Dashed lines: each model's clean score", ha="center", color=MUTED, fontsize=8)
    save(fig, "robustness.png")


def fig_training(history):
    """Figure: CNN training and validation loss per epoch, best epoch marked."""
    best = int(history["best_epoch"].iloc[0])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(history["epoch"], history["train_loss"], color=INK_2, ls=(0, (4, 2)), marker="o",
            markersize=4.5, label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], color=COLOURS["cnn"], marker=MARKERS["cnn"],
            markeredgecolor=SURFACE, label="Validation loss")
    ax.axvline(best, color=MUTED, lw=0.8)
    ax.annotate(f"best epoch {best}\n(weights restored)", (best, 0), xytext=(6, 4),
                textcoords="offset points", va="bottom", color=INK_2, fontsize=8)
    ax.set(xlabel="Epoch", ylabel="Weighted BCE loss", title="MobileNetV3-Small, head-only training")
    ax.set_xticks(history["epoch"])
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right")
    save(fig, "cnn_training.png")


def fig_cv_lopo(cv, lopo):
    """Figure: grouped CV folds and leave-one-printer-out ROC-AUC for the baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), gridspec_kw={"width_ratios": [1, 2]})
    ax = axes[0]
    for i, model in enumerate(("logreg", "random_forest")):
        f = cv[cv["model"] == model]["roc_auc"]
        jitter = np.linspace(-0.12, 0.12, len(f))
        ax.scatter(i + jitter, f, color=COLOURS[model], marker=MARKERS[model], s=40, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3)
        ax.hlines(f.mean(), i - 0.25, i + 0.25, color=INK, lw=1.5, zorder=4)
        ax.annotate(f"mean {f.mean():.2f}", (i + 0.27, f.mean()), va="center", color=INK_2, fontsize=8)
    ax.set_xticks([0, 1], ["Logistic\nregression", "Random\nforest"])
    ax.set(xlim=(-0.5, 1.9), ylim=(0, 1.02), ylabel="ROC-AUC", title="5-fold CV over prints")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    printers = sorted(lopo["held_out"].unique())
    width = 0.38
    for j, model in enumerate(("logreg", "random_forest")):
        f = lopo[lopo["model"] == model].set_index("held_out").loc[printers]
        xs = np.arange(len(printers)) + (j - 0.5) * (width + 0.02)
        ax.bar(xs, f["roc_auc"], width=width, color=COLOURS[model], label=LABELS[model],
               hatch=None if j == 0 else "////", edgecolor=SURFACE, linewidth=0.8)
        for x_, v in zip(xs, f["roc_auc"]):
            ax.annotate(f"{v:.2f}", (x_, v), xytext=(0, 2), textcoords="offset points", ha="center",
                        color=INK_2, fontsize=7.5)
    n_prints = lopo.drop_duplicates("held_out").set_index("held_out")["held_out_prints"]
    ax.set_xticks(range(len(printers)), [f"Printer {p}\n({n_prints[p]} prints)" for p in printers])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.annotate("chance", (len(printers) - 0.5, 0.5), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", color=MUTED, fontsize=8, annotation_clip=False)
    ax.set_xlim(-0.6, len(printers) - 0.4)
    ax.set(ylim=(0, 1.08), title="Leave-one-printer-out (train on the other 4 printers)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, "cv_lopo.png")


def main(n_boot=2000):
    """Build the metrics table, comparison table and all figures from the result CSVs."""
    style()
    preds = pd.read_csv(RESULTS / "predictions.csv")
    cv = pd.read_csv(RESULTS / "cv_folds.csv")
    lopo = pd.read_csv(RESULTS / "lopo.csv")
    profiling = pd.read_csv(RESULTS / "profiling.csv")
    fit_times = pd.concat([pd.read_csv(RESULTS / "baseline_fit_times.csv"),
                           pd.read_csv(RESULTS / "cnn_fit_time.csv")[["model", "fit_seconds"]]])

    metrics = metrics_with_ci(preds, n_boot, config.DEFAULT_SEED)
    metrics.to_csv(RESULTS / "test_metrics.csv", index=False)
    table = comparison_table(metrics, cv, lopo, profiling, fit_times)
    table.to_csv(RESULTS / "comparison.csv", index=False)
    write_markdown(table, metrics, RESULTS / "comparison.md")

    fig_roc(preds)
    fig_sweep(pd.read_csv(RESULTS / "resolution_sweep.csv"))
    fig_robustness(pd.read_csv(RESULTS / "robustness.csv"))
    fig_training(pd.read_csv(RESULTS / "cnn_history.csv"))
    fig_cv_lopo(cv, lopo)
    print((RESULTS / "comparison.md").read_text(encoding="utf-8"))
    print(f"figures in {FIGURES}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    main(ap.parse_args().n_boot)
