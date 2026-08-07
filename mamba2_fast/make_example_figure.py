"""
make_example_figure.py
=======================
Builds a 3-panel example figure (measured waveform | real model output |
predicted corrected waveform), matching the visual layout of
mamba_zamba/make_figures.py — but unlike that script, nothing here is
hardcoded. Every number and every word of the diagnosis/correction text is
read directly from results/eval_results.jsonl (a real, already-run
inference on the fine-tuned model) and data/test.jsonl (the real input
features), keyed by --idx.

The correction vector shown is exactly what compute_cv() in eval_mamba2.py
computed for this sample — parsed back out of the real "generated" field,
not recomputed by a second, potentially-diverging code path.

Usage:
    python make_example_figure.py --idx 1
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import numpy as np
import textwrap

RESULTS_PATH = Path("./results/eval_results.jsonl")
TEST_PATH    = Path("./data/test.jsonl")
OUT_DIR      = Path("./figures")
OUT_DIR.mkdir(exist_ok=True)

# Real hardware waveform images, keyed by the exact filename that appears in
# mamba_llm/features_full.csv for that sample (verified by matching measured
# peak_nm/rms_nm/phase_lag between the CSV and the input this example uses —
# see conversation notes: the naive filename-pattern guess was WRONG once,
# corrected against the actual numbers before use here).
REAL_IMAGES = {
    1: "../data/ori_plots/sine_Plots/sine_100Hz_100Hz_2_absolute_slice_50_200.png",
}

CV_MARKER = "CORRECTION VECTOR:"
CV_LINE_RE = re.compile(r"^\s*(\w+)\s*=\s*([+-]?[\d.]+)\s*$")


def load_real_record(idx):
    with open(RESULTS_PATH) as f:
        results = [json.loads(l) for l in f]
    with open(TEST_PATH) as f:
        test_ex = [json.loads(l) for l in f]
    record = next(r for r in results if r["idx"] == idx)
    ex = test_ex[idx]
    return record, ex


def parse_generated(full_text):
    """Split the real 'generated' field into (diagnosis_correction_text, cv_dict).
    Parses the SAME text the model actually produced — not a re-derivation."""
    idx = full_text.find(CV_MARKER)
    text = full_text[:idx].strip() if idx != -1 else full_text.strip()
    cv = {}
    if idx != -1:
        for line in full_text[idx + len(CV_MARKER):].splitlines():
            m = CV_LINE_RE.match(line)
            if m:
                cv[m.group(1)] = float(m.group(2))
    diagnosis, correction = "", ""
    for line in text.splitlines():
        if line.startswith("DIAGNOSIS:"):
            diagnosis = line[len("DIAGNOSIS:"):].strip()
        elif line.startswith("CORRECTION:"):
            correction = line[len("CORRECTION:"):].strip()
    return diagnosis, correction, cv


def parse_input_features(input_text):
    f = {}
    m = re.search(r"COMMANDED WAVEFORM: (\w+) at ([\d.]+) Hz", input_text)
    if m:
        f["waveform"], f["freq"] = m.group(1), float(m.group(2))
    for key, pat in {
        "target_peak": r"target peak ±([\d.]+) nm",
        "rms": r"RMS displacement \(nm\):\s*([\d.]+)",
        "peak": r"Peak displacement \(nm\):\s*([\d.]+)",
        "thd": r"THD \(%\):\s*([\d.]+)",
        "phase_lag": r"Phase lag \(deg\):\s*([-\d.]+)",
        "fopdt_lag": r"FOPDT predicted phase lag \(deg\):\s*([-\d.]+)",
        "crest": r"Crest factor:\s*([\d.]+)",
    }.items():
        m = re.search(pat, input_text)
        if m:
            f[key] = float(m.group(1))
    return f


def synth_corrected_sine(freq_hz, target_peak_nm):
    # Same method as mamba_zamba/make_figures.py: after the correction
    # vector is applied to the control input, the OUTPUT is predicted to
    # return to a clean sine at the commanded target amplitude — this is
    # an analytical prediction, not a re-measurement (no hardware run of
    # the correction has happened yet). Labeled as such in the figure.
    t = np.linspace(0, 0.150, 6000)
    y = target_peak_nm * np.sin(2 * math.pi * freq_hz * t)
    return t * 1000, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=1)
    args = parser.parse_args()

    record, ex = load_real_record(args.idx)
    diagnosis, correction, cv = parse_generated(record["generated"])
    feats = parse_input_features(ex["input"])
    wtype = ex["waveform"]

    img_path = REAL_IMAGES.get(args.idx)
    if img_path is None or not Path(img_path).exists():
        raise SystemExit(
            f"No verified real hardware image registered for idx={args.idx}. "
            f"Add an entry to REAL_IMAGES after confirming (via "
            f"mamba_llm/features_full.csv peak_nm/rms_nm/phase_lag) that the "
            f"image actually corresponds to this sample — do not guess by "
            f"filename pattern alone (see the docstring)."
        )

    t_corr, y_corr = synth_corrected_sine(feats["freq"], feats["target_peak"])
    amp = feats["target_peak"]

    fig = plt.figure(figsize=(26, 9.5))
    fig.patch.set_facecolor("#F4F4F4")
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.55, 0.95],
                            left=0.02, right=0.98, top=0.85, bottom=0.08, wspace=0.06)

    # ── panel 1: real measured waveform ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    img = imread(img_path)
    ax1.imshow(img, aspect="auto")
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_edgecolor("#CC0000"); sp.set_linewidth(2.0)
    ax1.set_title(f"Measured Waveform — {wtype.upper()} at {feats['freq']:.0f} Hz\n"
                  f"(real hardware data, piezoelectric actuator)",
                  fontsize=14, fontweight="bold", color="#B71C1C", pad=10)

    # ── panel 2: real model output ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    color = "#1565C0"
    y_cur = [0.97]

    def put(text, bold=False, fcolor="#111111", size=13.0, indent=0.02, step=0.046):
        if y_cur[0] < 0.01:
            return
        ax2.text(indent, y_cur[0], text, transform=ax2.transAxes, fontsize=size,
                  color=fcolor, fontweight="bold" if bold else "normal",
                  verticalalignment="top", clip_on=True)
        y_cur[0] -= step

    put("DIAGNOSIS (real Mamba2-780M output, zero-shot)", bold=True, fcolor=color,
        size=15.0, step=0.055)
    for line in textwrap.wrap(diagnosis, 62):
        put(line)
    y_cur[0] -= 0.018

    put("CORRECTION (real Mamba2-780M output)", bold=True, fcolor=color,
        size=15.0, step=0.055)
    for line in textwrap.wrap(correction, 62):
        put(line)
    y_cur[0] -= 0.018

    put("CORRECTION VECTOR", bold=True, fcolor="#444444", size=13.5, step=0.052)
    put("(analytically computed, not model-generated)",
        fcolor="#777777", size=10.5, step=0.040)
    y_cur[0] -= 0.006
    for k, v in cv.items():
        put(f"  {k:<26} = {v:+.4f}", fcolor="#222222", size=11.5, indent=0.04, step=0.042)

    y_cur[0] -= 0.018
    put(f"BLEU-1={record['bleu1']:.3f}  ROUGE-L={record['rouge_l']:.3f}  "
        f"({record['tokens']} tokens, {record['ms']:.0f} ms)",
        fcolor="#777777", size=10.5, step=0.040)

    ax2.annotate("", xy=(1.04, 0.50), xytext=(0.97, 0.50), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=2.0, mutation_scale=18))

    # ── panel 3: predicted corrected waveform ────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(t_corr, y_corr, color="#1B5E20", linewidth=2.0, alpha=0.9)
    ax3.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Time (ms)", fontsize=12)
    ax3.set_ylabel("Displacement (nm)", fontsize=12)
    ax3.set_ylim(-amp * 1.25, amp * 1.25)
    ax3.set_xlim(0, 150)
    ax3.grid(True, alpha=0.2, linewidth=0.5)
    for sp in ax3.spines.values():
        sp.set_edgecolor("#1B5E20"); sp.set_linewidth(1.5)
    ax3.set_title("Predicted After Correction\n(analytically computed — not yet re-measured on hardware)",
                  fontsize=13, fontweight="bold", color="#1B5E20", pad=10)

    h2 = cv.get("harmonic_2_amp", 0.0)
    stats = (f"Phase lag:  {feats.get('phase_lag', 0):.1f}° → 0° (target)\n"
             f"Amplitude:  ±{amp:.1f} nm (restored to command)\n"
             f"2nd harmonic feedforward:  {h2:+.4f}")
    ax3.text(0.04, 0.97, stats, transform=ax3.transAxes, fontsize=11,
              verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                        edgecolor="#A5D6A7", alpha=0.95))

    fig.suptitle(f"Mamba2-780M (fine-tuned)  |  Piezoelectric Actuator Waveform Diagnosis  |  "
                 f"{wtype.upper()} at {feats['freq']:.0f} Hz  |  test sample idx={args.idx}",
                 fontsize=16, fontweight="bold", color="#111111", y=0.95)

    out_path = OUT_DIR / f"example_idx{args.idx}_{wtype}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
