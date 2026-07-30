"""
visualize_results.py
====================
Generates side-by-side figures: synthesized waveform (left) + LLM report (right).
Reads eval_results.jsonl produced by eval_zamba.py.

Usage:
    python visualize_results.py [--results results/eval_results.jsonl]
                                [--out figures/]
                                [--n 2]
"""

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/eval_results.jsonl")
parser.add_argument("--out",     default="figures")
parser.add_argument("--n",       type=int, default=2,
                    help="samples per waveform type")
args = parser.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)

data = [json.loads(l) for l in open(args.results) if l.strip()
        if "error" not in l]

WAVEFORM_TYPES = ["sine", "square", "ramp", "pulse", "noise"]


# ── feature parsing ───────────────────────────────────────────────────────────

def parse_features(input_text):
    f = {}
    m = re.search(r"COMMANDED WAVEFORM: (\w+) at ([\d.]+) Hz", input_text)
    if m:
        f["wtype"] = m.group(1)
        f["freq"]  = float(m.group(2))
    patterns = {
        "rms":          r"RMS displacement \(nm\):\s*([\d.]+)",
        "peak":         r"Peak displacement \(nm\):\s*([\d.]+)",
        "phase_lag":    r"Phase lag \(deg\):\s*([-\d.]+)",
        "h2":           r"2nd harmonic ratio:\s*([\d.]+)",
        "h3":           r"3rd harmonic ratio:\s*([\d.]+)",
        "thd":          r"THD \(%\):\s*([\d.]+)",
        "duty_cycle":   r"Duty cycle:\s*([\d.]+)",
        "target_peak":  r"target peak ±([\d.]+) nm",
    }
    for key, pat in patterns.items():
        m = re.search(pat, input_text)
        if m:
            f[key] = float(m.group(1))
    return f


# ── waveform synthesizer ──────────────────────────────────────────────────────

def synthesize(ex):
    ref = ex.get("reference", "")
    inp = re.search(r"(?s).*?(?=DIAGNOSIS:)", ref)
    features = parse_features(ex.get("reference", "") + ex.get("generated", ""))

    wtype = ex.get("waveform", "sine")
    freq  = features.get("freq", 10.0)
    amp   = features.get("peak", features.get("rms", 100.0) * math.sqrt(2))
    phase = math.radians(features.get("phase_lag", 0.0))
    h2    = features.get("h2", 0.0)
    h3    = features.get("h3", 0.0)

    T     = max(4 / freq, 0.05)
    t     = np.linspace(0, T, 2000)
    omega = 2 * math.pi * freq

    if wtype == "sine":
        y = amp * (np.sin(omega * t + phase)
                   + h2 * np.sin(2 * omega * t)
                   + h3 * np.sin(3 * omega * t))
    elif wtype == "square":
        dc   = features.get("duty_cycle", 0.5)
        base = amp * np.sign(np.sin(omega * t + phase))
        y    = base + h2 * amp * np.sin(2 * omega * t)
    elif wtype == "ramp":
        phase_t = (omega * t + phase) % (2 * math.pi)
        y = amp * (phase_t / math.pi - 1.0)
        y += h3 * amp * np.sin(3 * omega * t)
    elif wtype == "pulse":
        base = np.zeros_like(t)
        idx  = (omega * t + phase) % (2 * math.pi) < 0.3
        base[idx] = amp
        ringing = 0.3 * amp * np.exp(-5 * ((omega * t + phase) % (2 * math.pi) - 0.3))
        ringing[~idx] = 0
        y = base + ringing
    elif wtype == "noise":
        rng = np.random.default_rng(42)
        rms = features.get("rms", 50.0)
        y   = rng.normal(0, rms, len(t))
        resonance_freq = freq
        y  += 2 * rms * np.sin(2 * math.pi * resonance_freq * t)
    else:
        y = amp * np.sin(omega * t + phase)

    return t, y


# ── text formatter ────────────────────────────────────────────────────────────

def wrap_text(text, width=55):
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.extend(textwrap.wrap(para.strip(), width))
            lines.append("")
    return "\n".join(lines)


def extract_sections(generated):
    diag, corr, cv = "", "", ""
    m = re.search(r"DIAGNOSIS:(.*?)(?=\n\nCORRECTION:|$)", generated, re.DOTALL)
    if m:
        diag = m.group(1).strip()
    m = re.search(r"CORRECTION:(.*?)(?=\n\nCORRECTION VECTOR:|$)", generated, re.DOTALL)
    if m:
        corr = m.group(1).strip()
    m = re.search(r"CORRECTION VECTOR:(.*?)$", generated, re.DOTALL)
    if m:
        cv = m.group(1).strip()
    return diag, corr, cv


# ── figure builder ────────────────────────────────────────────────────────────

WTYPE_COLOR = {
    "sine":   "#2196F3",
    "square": "#4CAF50",
    "ramp":   "#FF9800",
    "pulse":  "#9C27B0",
    "noise":  "#F44336",
}


def make_figure(ex, out_path):
    wtype = ex["waveform"]
    color = WTYPE_COLOR.get(wtype, "#333333")

    t, y = synthesize(ex)
    diag, corr, cv = extract_sections(ex["generated"])

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4],
                            left=0.04, right=0.97, top=0.88, bottom=0.10,
                            wspace=0.06)

    # ── left: waveform plot ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#FFFFFF")
    ax.plot(t * 1000, y, color=color, linewidth=1.6, alpha=0.9)
    ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Displacement (nm)", fontsize=11)
    ax.set_title(f"Measured Waveform  [{wtype}]", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    stats_text = (
        f"BLEU-1:   {ex['bleu1']:.3f}\n"
        f"ROUGE-L:  {ex['rouge_l']:.3f}\n"
        f"Tokens:   {ex['tokens']}\n"
        f"Time:     {ex['seconds']:.1f}s"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF2FF",
                      edgecolor="#CCCCCC", alpha=0.9))

    # ── right: LLM report ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.axis("off")
    for spine in ax2.spines.values():
        spine.set_visible(False)

    y_cursor = 0.97
    line_h   = 0.038

    def put(text, bold=False, color="#111111", indent=0, size=9.5):
        nonlocal y_cursor
        ax2.text(0.02 + indent, y_cursor, text, transform=ax2.transAxes,
                 fontsize=size, color=color,
                 fontweight="bold" if bold else "normal",
                 verticalalignment="top")
        y_cursor -= line_h

    put("DIAGNOSIS", bold=True, color=color, size=10.5)
    y_cursor -= 0.005
    for line in textwrap.wrap(diag, 72):
        put(line, indent=0.01)
    y_cursor -= 0.01

    put("CORRECTION", bold=True, color=color, size=10.5)
    y_cursor -= 0.005
    for line in textwrap.wrap(corr, 72):
        put(line, indent=0.01)
    y_cursor -= 0.01

    put("CORRECTION VECTOR", bold=True, color="#555555", size=9.5)
    y_cursor -= 0.005
    for cv_line in cv.strip().split("\n"):
        put(cv_line.strip(), indent=0.01, color="#333333", size=8.8)

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Zamba2-7B-instruct  |  Piezoelectric Actuator Diagnostic Report  "
        f"|  Sample #{ex['idx']}",
        fontsize=12, fontweight="bold", color="#222222", y=0.96
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── generate figures ──────────────────────────────────────────────────────────

counts = {w: 0 for w in WAVEFORM_TYPES}
for ex in data:
    wt = ex.get("waveform", "sine")
    if wt in counts and counts[wt] < args.n:
        out_path = Path(args.out) / f"{wt}_{counts[wt]+1:02d}.png"
        make_figure(ex, out_path)
        counts[wt] += 1

print(f"\nAll figures saved to ./{args.out}/")
