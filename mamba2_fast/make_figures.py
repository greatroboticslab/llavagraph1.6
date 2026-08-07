"""
make_figures.py
================
Generates the figures for the mamba2_fast track from the structured outputs
that finetune_mamba2.py and eval_mamba2.py already save:

  results/training_history.csv  -> learning_curve.png
  results/eval_results.jsonl    -> accuracy_comparison.png, speed_comparison.png

The Zamba2-7B few-shot baseline numbers are hardcoded below — they come from
a completed, separate run (mamba_zamba/eval_zamba.py); there is no live file
to read them from.

Run locally after copying results/ back from the cluster:
    rsync -avz hamilton:/projects/ya4v/llavagraph1.6/mamba2_fast/results/ ./results/
    python make_figures.py
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

RESULTS_DIR = Path("./results")
FIG_DIR = Path("./figures")
FIG_DIR.mkdir(exist_ok=True)

# ── palette (validated categorical set — see dataviz skill references) ──────
BLUE   = "#2a78d6"   # slot 1 — the story: this model's result
GREY   = "#898781"   # muted ink — reference/baseline, not a competing hue
INK    = "#0b0b0b"
SECOND = "#52514e"
GRID   = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": GRID,
    "axes.labelcolor": SECOND,
    "text.color": INK,
    "xtick.color": SECOND,
    "ytick.color": SECOND,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# ── Zamba2-7B few-shot baseline (mamba_zamba/eval_zamba.py, separate run) ───
BASELINE = {
    "per_type": {
        "sine":   {"n": 120, "bleu1": 0.414, "rouge_l": 0.226},
        "square": {"n": 120, "bleu1": 0.368, "rouge_l": 0.203},
        "ramp":   {"n": 80,  "bleu1": 0.370, "rouge_l": 0.204},
        "pulse":  {"n": 75,  "bleu1": 0.364, "rouge_l": 0.197},
        "noise":  {"n": 92,  "bleu1": 0.386, "rouge_l": 0.208},
    },
    "overall": {"bleu1": 0.382, "rouge_l": 0.209},
    "avg_ms": 21600,  # 2.92h / 487 samples
    "label": "Zamba2-7B\n(few-shot)",
}


def style_axes(ax, hide_top_right=True):
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


# ── 1. Learning curve ────────────────────────────────────────────────────────

def plot_learning_curve():
    path = RESULTS_DIR / "training_history.csv"
    if not path.exists():
        print(f"SKIP learning curve — {path} not found")
        return
    epochs, train_loss, eval_loss = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(float(row["epoch"]))
            train_loss.append(float(row["train_loss"]) if row["train_loss"] else None)
            eval_loss.append(float(row["eval_loss"]))

    # train_loss is already corrected for the gradient_accumulation logging
    # quirk in finetune_mamba2.py (confirmed via diagnose_loss_gap.py), so
    # both series are now on the same scale and belong on one axis.
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    style_axes(ax)
    ax.plot(epochs, train_loss, color=GREY, linewidth=2, label="train loss")
    ax.plot(epochs, eval_loss, color=BLUE, linewidth=2, label="eval loss")
    for series, color in [(train_loss, GREY), (eval_loss, BLUE)]:
        ax.scatter([epochs[-1]], [series[-1]], s=64, color=color, zorder=5,
                   edgecolors=SURFACE, linewidths=2)
        ax.annotate(f"{series[-1]:.3f}", (epochs[-1], series[-1]),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    color=SECOND, fontsize=10)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Mamba2-780M fine-tuning: learning curve", color=INK, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curve.png")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'learning_curve.png'}")


# ── 2. Accuracy comparison (BLEU-1 / ROUGE-L, per waveform type) ────────────

def plot_accuracy_comparison():
    path = RESULTS_DIR / "eval_results.jsonl"
    if not path.exists():
        print(f"SKIP accuracy comparison — {path} not found")
        return
    per_type_bleu, per_type_rouge = defaultdict(list), defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "bleu1" not in r:
                continue
            per_type_bleu[r["waveform"]].append(r["bleu1"])
            per_type_rouge[r["waveform"]].append(r["rouge_l"])

    types = ["sine", "square", "ramp", "pulse", "noise"]
    types = [t for t in types if t in per_type_bleu]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    for ax, (metric_name, per_type, base_key) in zip(
        axes, [("BLEU-1", per_type_bleu, "bleu1"), ("ROUGE-L", per_type_rouge, "rouge_l")]
    ):
        style_axes(ax)
        x = range(len(types))
        width = 0.36
        base_vals = [BASELINE["per_type"][t][base_key] for t in types]
        ours_vals = [sum(per_type[t]) / len(per_type[t]) for t in types]

        ax.bar([i - width/2 for i in x], base_vals, width=width, color=GREY,
               label="Zamba2-7B (few-shot)", zorder=3)
        ax.bar([i + width/2 for i in x], ours_vals, width=width, color=BLUE,
               label="Mamba2-780M (ours)", zorder=3)

        for i, v in enumerate(ours_vals):
            ax.annotate(f"{v:.2f}", (i + width/2, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8.5, color=SECOND)

        ax.set_xticks(list(x))
        ax.set_xticklabels(types)
        ax.set_title(metric_name, loc="left", fontsize=12, color=INK)
        ax.set_ylim(0, max(max(base_vals), max(ours_vals)) * 1.25)

    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Text-quality accuracy by waveform type", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "accuracy_comparison.png")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'accuracy_comparison.png'}")


# ── 3. Speed comparison ──────────────────────────────────────────────────────

def plot_speed_comparison():
    path = RESULTS_DIR / "eval_results.jsonl"
    if not path.exists():
        print(f"SKIP speed comparison — {path} not found")
        return
    ms_values = [json.loads(l)["ms"] for l in open(path) if "ms" in json.loads(l)]
    if not ms_values:
        print("SKIP speed comparison — no 'ms' field in eval_results.jsonl")
        return
    # Median, not mean — a one-time CUDA-kernel warmup on the first sample
    # is a real but non-representative outlier (see conversation history).
    ours_ms = sorted(ms_values)[len(ms_values) // 2]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    style_axes(ax)
    labels = [BASELINE["label"], "Mamba2-780M\n(ours, fine-tuned)"]
    values = [BASELINE["avg_ms"], ours_ms]
    colors = [GREY, BLUE]

    bars = ax.bar(labels, values, width=0.5, color=colors, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("ms / sample (log scale)")
    ax.set_title(f"Inference latency — {BASELINE['avg_ms']/ours_ms:.1f}x faster",
                 loc="left", fontsize=13, color=INK)

    for bar, v in zip(bars, values):
        label = f"{v/1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"
        ax.annotate(label, (bar.get_x() + bar.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=10, color=INK)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "speed_comparison.png")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'speed_comparison.png'}")


if __name__ == "__main__":
    plot_learning_curve()
    plot_accuracy_comparison()
    plot_speed_comparison()
