"""
make_figures.py
===============
Generates 3-panel figures for the README:
  Left:   Original distorted waveform (synthesized from measured features)
  Center: LLM-generated DIAGNOSIS + CORRECTION text
  Right:  Corrected waveform (synthesized by applying the CORRECTION VECTOR)

Run locally:
    cd mamba_zamba && python make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import textwrap
import math
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── actual LLM output from eval run (job 53656) ───────────────────────────────

EXAMPLES = {
    "sine": {
        "features": {
            "wtype":       "sine",
            "freq":        20.1,
            "target_peak": 335.7,
            "rms":         206.2,
            "phase_lag":   172.2,
            "h2":          0.1624,
            "h3":          0.0992,
            "amp_drift":   18.87,
            "thd":         22.01,
            "crest":       1.628,
        },
        "cv": {
            "phase_offset_deg": -172.2,
            "amplitude_scale":  +1.1512,
            "harmonic_2_amp":   -0.1624,
            "harmonic_3_amp":   -0.0992,
            "dc_offset_nm":      0.0,
            "hysteresis_ff_nm":  0.0,
        },
        "input_summary": (
            "COMMANDED WAVEFORM: sine at 20.1 Hz  |  Target peak: ±335.7 nm\n"
            "Phase lag: 172.2°  |  THD: 22.01%  |  Crest factor: 1.628\n"
            "2nd harmonic: 0.1624  |  3rd harmonic: 0.0992\n"
            "Amplitude drift Q1→Q4: 18.87 nm"
        ),
        "diagnosis": (
            "The commanded sine at 20.1 Hz displays substantial nonlinearity and distortion. "
            "Despite achieving the desired peak displacement of 335.7 nm, the phase lag of 172.2° "
            "greatly surpasses the expected 1.84°, indicating strong nonlinear effects. The THD of "
            "22.01% and elevated 2nd/3rd harmonics confirm the output is far from sinusoidal."
        ),
        "correction": (
            "Apply a feedforward phase lead of 172.2° at 20.1 Hz to cancel the excess lag. "
            "Simultaneously cancel the 2nd harmonic (ratio 0.1624) and 3rd harmonic (ratio 0.0992) "
            "via harmonic feedforward to restore sinusoidal shape."
        ),
    },
    "square": {
        "features": {
            "wtype":          "square",
            "freq":           100.0,
            "target_peak":    521.6,
            "rms":            400.0,
            "duty_cycle":     0.4763,
            "edge_sharpness": 0.9166,
            "h2":             0.0259,
            "thd":            56.93,
        },
        "cv": {
            "duty_cycle_trim":        +0.0237,
            "edge_sharpness_deficit": +0.0834,
            "harmonic_2_amp":         -0.0259,
            "dc_offset_nm":            0.0,
        },
        "input_summary": (
            "COMMANDED WAVEFORM: square at 100 Hz  |  Target peak: ±521.6 nm\n"
            "THD: 56.93%  |  Duty cycle: 0.4763  |  Edge sharpness: 0.9166\n"
            "2nd harmonic: 0.0259  |  Amplitude drift: 0 nm"
        ),
        "diagnosis": (
            "The commanded square wave at 100 Hz shows a duty cycle of 0.4763 (ideal: 0.500) "
            "and reduced edge sharpness of 0.9166 (ideal: 1.000), indicating bandwidth-limited "
            "transitions. THD of 56.93% exceeds the theoretical 48.3% for an ideal square wave, "
            "with a 2nd harmonic ratio of 0.0259 introducing even-order distortion."
        ),
        "correction": (
            "Trim the duty cycle by +0.0237 to restore symmetry. Apply edge pre-emphasis to "
            "recover the 0.0834 sharpness deficit at the rising and falling transitions. "
            "Cancel the 2nd harmonic (ratio 0.0259) via feedforward to remove even-order distortion."
        ),
    },
}

COLORS = {"sine": "#1565C0", "square": "#2E7D32"}


# ── waveform synthesizers ─────────────────────────────────────────────────────

def synth_sine(f, corrected=False):
    freq  = f["freq"]
    amp   = f["target_peak"]
    phase = math.radians(f["phase_lag"])
    h2    = f["h2"]
    h3    = f["h3"]

    T     = 4.0 / freq
    t     = np.linspace(0, T, 4000)
    omega = 2 * math.pi * freq

    if corrected:
        cv     = EXAMPLES["sine"]["cv"]
        phase  = phase + math.radians(cv["phase_offset_deg"])   # cancels lag
        amp    = amp * cv["amplitude_scale"]
        h2     = h2  + cv["harmonic_2_amp"]                     # cancels h2
        h3     = h3  + cv["harmonic_3_amp"]                     # cancels h3

    y = (amp * np.sin(omega * t + phase)
         + h2 * amp * np.sin(2 * omega * t)
         + h3 * amp * np.sin(3 * omega * t))
    return t * 1000, y   # time in ms


def synth_square(f, corrected=False):
    freq  = f["freq"]
    amp   = f["target_peak"]
    duty  = f["duty_cycle"]
    edge  = f["edge_sharpness"]
    h2    = f["h2"]

    T     = 3.0 / freq
    t     = np.linspace(0, T, 8000)
    omega = 2 * math.pi * freq
    phase = (omega * t) % (2 * math.pi)

    if corrected:
        cv   = EXAMPLES["square"]["cv"]
        duty = duty + cv["duty_cycle_trim"]        # restores 0.500
        edge = min(edge + cv["edge_sharpness_deficit"], 1.0)   # restores 1.000
        h2   = h2  + cv["harmonic_2_amp"]          # cancels h2

    # Square wave with soft edges (edge_sharpness controls steepness)
    steepness = 20 + edge * 80     # sharper when edge_sharpness closer to 1
    y = amp * np.tanh(steepness * np.sin(phase)) * (2 * duty)
    y += h2 * amp * np.sin(2 * omega * t)
    return t * 1000, y


SYNTH = {"sine": synth_sine, "square": synth_square}


# ── figure builder ────────────────────────────────────────────────────────────

def make_figure(wtype):
    ex     = EXAMPLES[wtype]
    f      = ex["features"]
    cv     = ex["cv"]
    color  = COLORS[wtype]
    synth  = SYNTH[wtype]

    t_orig, y_orig = synth(f, corrected=False)
    t_corr, y_corr = synth(f, corrected=True)

    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor("#F4F4F4")
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.0, 1.3, 1.0],
        left=0.03, right=0.98,
        top=0.87,  bottom=0.10,
        wspace=0.06,
    )

    # ── panel 1: original distorted waveform ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FFFFFF")
    ax1.plot(t_orig, y_orig, color=color, linewidth=1.8, alpha=0.85)
    ax1.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle="--")
    ax1.set_xlabel("Time (ms)", fontsize=10)
    ax1.set_ylabel("Displacement (nm)", fontsize=10)
    ax1.set_title("Measured Waveform\n(distorted)", fontsize=11,
                  fontweight="bold", color="#B71C1C", pad=6)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    for sp in ax1.spines.values():
        sp.set_linewidth(0.8)

    # input feature summary box
    ax1.text(0.03, 0.98, ex["input_summary"],
             transform=ax1.transAxes, fontsize=7.2,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFFDD",
                       edgecolor="#CCCCCC"))

    # ── panel 2: LLM text output ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FAFAFA")
    ax2.axis("off")

    y_cur = 0.97
    def put(text, bold=False, color="#111111", size=9.2, indent=0.02, gap=0.038):
        nonlocal y_cur
        ax2.text(indent, y_cur, text, transform=ax2.transAxes,
                 fontsize=size, color=color,
                 fontweight="bold" if bold else "normal",
                 verticalalignment="top")
        y_cur -= gap

    put("DIAGNOSIS", bold=True, color=color, size=11, gap=0.01)
    y_cur -= 0.01
    for line in textwrap.wrap(ex["diagnosis"], 68):
        put(line, size=8.8, gap=0.034)
    y_cur -= 0.018

    put("CORRECTION", bold=True, color=color, size=11, gap=0.01)
    y_cur -= 0.01
    for line in textwrap.wrap(ex["correction"], 68):
        put(line, size=8.8, gap=0.034)
    y_cur -= 0.018

    put("CORRECTION VECTOR  (analytically computed)", bold=True,
        color="#555555", size=9.5, gap=0.01)
    y_cur -= 0.01
    for k, v in cv.items():
        put(f"  {k:<28} = {v:+.4f}", size=8.8, color="#333333",
            indent=0.04, gap=0.030)

    # arrow annotation
    ax2.annotate("", xy=(1.02, 0.5), xytext=(0.96, 0.5),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#333333",
                                 lw=1.5, mutation_scale=16))

    # ── panel 3: corrected waveform ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(t_corr, y_corr, color="#2E7D32", linewidth=1.8, alpha=0.9)
    ax3.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle="--")
    ax3.set_xlabel("Time (ms)", fontsize=10)
    ax3.set_ylabel("Displacement (nm)", fontsize=10)
    ax3.set_title("After Applying\nCorrection Vector", fontsize=11,
                  fontweight="bold", color="#1B5E20", pad=6)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    for sp in ax3.spines.values():
        sp.set_linewidth(0.8)

    # improvement stats
    thd_orig = f.get("thd", 0)
    if wtype == "sine":
        h2_corr  = abs(f["h2"] + cv["harmonic_2_amp"])
        h3_corr  = abs(f["h3"] + cv["harmonic_3_amp"])
        thd_corr = math.sqrt(h2_corr**2 + h3_corr**2) * 100
        stats = (f"THD:  {thd_orig:.1f}% → {thd_corr:.2f}%\n"
                 f"Phase lag:  172.2° → ~0°\n"
                 f"Harmonics:  cancelled")
    else:
        duty_corr  = f["duty_cycle"] + cv["duty_cycle_trim"]
        edge_corr  = f["edge_sharpness"] + cv["edge_sharpness_deficit"]
        stats = (f"Duty cycle:  {f['duty_cycle']:.4f} → {duty_corr:.4f}\n"
                 f"Edge sharpness:  {f['edge_sharpness']:.4f} → {edge_corr:.4f}\n"
                 f"2nd harmonic:  cancelled")

    ax3.text(0.03, 0.98, stats, transform=ax3.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#E8F5E9CC",
                       edgecolor="#A5D6A7"))

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Zamba2-7B-instruct  |  Waveform: {wtype.upper()}  "
        f"|  Piezoelectric Actuator Diagnostic",
        fontsize=12, fontweight="bold", color="#111111", y=0.95,
    )

    out_path = OUT / f"{wtype}_example.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


for wtype in ["sine", "square"]:
    make_figure(wtype)

print("Done.")
