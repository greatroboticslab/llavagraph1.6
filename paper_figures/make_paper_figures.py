"""
make_paper_figures.py
=====================
Generates 3-panel paper figures using REAL measured data:

  Left:   Real waveform image captured from piezoelectric actuator hardware
  Center: Diagnostic report (DIAGNOSIS + CORRECTION + CORRECTION VECTOR)
          from test.jsonl — the ground-truth labels that a fine-tuned
          Zamba2-7B-instruct is trained to reproduce
  Right:  Predicted corrected waveform, synthesized by applying the
          CORRECTION VECTOR to the measured signal parameters

Run locally:
    cd paper_figures && python make_paper_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import numpy as np
import textwrap
import math
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── real waveform images (actual hardware measurements) ───────────────────────

REAL_IMAGES = {
    "sine":   "../data/ori_plots/sine_Plots/sine_100Hz_100Hz_1_absolute_slice_50_200.png",
    "square": "../data/ori_plots/square_Plots/square_100Hz_100Hz_1_absolute_slice_50_200.png",
}

# ── real measured features and reference outputs (from test.jsonl) ────────────
# Sine at 100 Hz sample — measured features extracted from hardware data

EXAMPLES = {
    "sine": {
        "features": {
            "freq":        100.0,
            "target_peak": 495.8,
            "rms":         206.7,
            "phase_lag":   76.1,
            "fopdt_lag":   9.11,
            "h2":          0.01052,
            "h3":          0.01873,
            "amp_drift":   74.86,
            "thd":         3.631,
            "crest":       2.398,
        },
        "cv": {
            "phase_offset_deg": -76.104,
            "amplitude_scale":  +1.6958,
            "harmonic_2_amp":   -0.01052,
            "harmonic_3_amp":   -0.01873,
            "dc_offset_nm":      0.0,
            "hysteresis_ff_nm": -18.62,
        },
        "diagnosis": (
            "The commanded sine wave at 100 Hz exhibits significant nonlinear distortion, "
            "primarily due to hysteresis and a substantial phase lag. The measured phase lag "
            "of 76.1° is far greater than the predicted 9.11°, indicating 66.99° of excess "
            "lag from nonlinear effects, while the crest factor of 2.398 (compared to ideal "
            "1.414) shows a sharp, non-sinusoidal peak. The amplitude drift of 74.86 nm from "
            "Q1 to Q4 suggests a growing, unstable drift in the system response."
        ),
        "correction": (
            "Advance the phase by 66.99° to compensate for the nonlinear lag. Cancel the "
            "3rd harmonic (ratio 0.01873) via feedforward to reduce waveform distortion. "
            "Apply hysteresis feedforward of -18.62 nm to counteract the DC creep drift."
        ),
    },

    "square": {
        "features": {
            "freq":          100.0,
            "target_peak":   521.6,
            "rms":           400.0,
            "duty_cycle":    0.4763,
            "edge_sharpness":0.9166,
            "h2":            0.02587,
            "thd":           56.93,
        },
        "cv": {
            "duty_cycle_trim":        +0.0237,
            "edge_sharpness_deficit": +0.08342,
            "harmonic_2_amp":         -0.02587,
            "dc_offset_nm":            0.0,
        },
        "diagnosis": (
            "The commanded square wave at 100 Hz is significantly distorted due to bandwidth "
            "limitation and nonlinear effects. The measured THD is 56.93%, substantially higher "
            "than the ideal 48.3%, indicating significant waveform deformation. The duty cycle "
            "is 0.4763 (ideal: 0.500) and edge sharpness is 0.9166 (ideal: 1.000), both "
            "indicating the system cannot perfectly track the sharp transitions. Amplitude drift "
            "is stable at 0 nm across all quarters."
        ),
        "correction": (
            "Trim the duty cycle by +0.0237 to restore 50% symmetry. Apply edge pre-emphasis "
            "to recover the 0.0834 sharpness deficit at rising and falling transitions. "
            "Cancel the 2nd harmonic (ratio 0.02587) via feedforward to remove even-order "
            "distortion."
        ),
    },
}

COLORS = {"sine": "#1565C0", "square": "#B71C1C"}


# ── corrected waveform synthesizer ────────────────────────────────────────────
# Applies the CORRECTION VECTOR to the measured signal parameters to produce
# the analytically predicted corrected waveform.

def synth_corrected_sine(f, cv):
    freq  = f["freq"]
    amp   = f["target_peak"]
    # After correction: phase lag is cancelled, harmonics are cancelled
    phase_corrected = math.radians(f["phase_lag"]) + math.radians(cv["phase_offset_deg"])
    amp_corrected   = amp * cv["amplitude_scale"]
    h2_corrected    = f["h2"] + cv["harmonic_2_amp"]   # → ~0
    h3_corrected    = f["h3"] + cv["harmonic_3_amp"]   # → ~0

    T     = 3.0 / freq
    t     = np.linspace(0, T, 6000)
    omega = 2 * math.pi * freq

    y = (amp_corrected * np.sin(omega * t + phase_corrected)
         + h2_corrected * amp_corrected * np.sin(2 * omega * t)
         + h3_corrected * amp_corrected * np.sin(3 * omega * t))
    return t * 1000, y


def synth_corrected_square(f, cv):
    freq          = f["freq"]
    amp           = f["target_peak"]
    duty_corrected = f["duty_cycle"] + cv["duty_cycle_trim"]       # → 0.500
    edge_corrected = f["edge_sharpness"] + cv["edge_sharpness_deficit"]  # → ~1.0
    h2_corrected   = f["h2"] + cv["harmonic_2_amp"]                # → ~0

    T     = 3.0 / freq
    t     = np.linspace(0, T, 8000)
    omega = 2 * math.pi * freq
    phase = (omega * t) % (2 * math.pi)

    steepness = 20 + edge_corrected * 100
    y = amp * np.tanh(steepness * np.sin(phase + math.pi * (1 - 2 * duty_corrected)))
    y += h2_corrected * amp * np.sin(2 * omega * t)
    return t * 1000, y


SYNTH_CORRECTED = {
    "sine":   synth_corrected_sine,
    "square": synth_corrected_square,
}


# ── figure builder ────────────────────────────────────────────────────────────

def make_figure(wtype):
    ex    = EXAMPLES[wtype]
    f     = ex["features"]
    cv    = ex["cv"]
    color = COLORS[wtype]

    t_corr, y_corr = SYNTH_CORRECTED[wtype](f, cv)

    fig = plt.figure(figsize=(21, 7.5))
    fig.patch.set_facecolor("#F2F2F2")
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.1, 1.3, 1.0],
        left=0.02, right=0.98,
        top=0.87,  bottom=0.08,
        wspace=0.05,
    )

    # ── panel 1: real measured waveform (hardware data) ───────────────────────
    ax1 = fig.add_subplot(gs[0])
    img_path = REAL_IMAGES.get(wtype)
    if img_path and Path(img_path).exists():
        img = imread(img_path)
        ax1.imshow(img, aspect="auto")
        ax1.set_xticks([])
        ax1.set_yticks([])
        for sp in ax1.spines.values():
            sp.set_edgecolor("#CC0000")
            sp.set_linewidth(2.0)
    else:
        ax1.text(0.5, 0.5, f"[image not found]\n{img_path}",
                 ha="center", va="center", color="red", fontsize=9)
        ax1.axis("off")

    ax1.set_title(
        f"Measured Waveform — {wtype.upper()} at {int(f['freq'])} Hz\n"
        f"(real hardware data, piezoelectric actuator)",
        fontsize=10, fontweight="bold", color="#B71C1C", pad=8,
    )

    # ── panel 2: diagnostic report ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.axis("off")
    for sp in ax2.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#DDDDDD")

    y_cur = 0.97
    def put(text, bold=False, color="#111111", size=9.0,
            indent=0.03, gap=0.036):
        nonlocal y_cur
        ax2.text(indent, y_cur, text, transform=ax2.transAxes,
                 fontsize=size, color=color,
                 fontweight="bold" if bold else "normal",
                 verticalalignment="top")
        y_cur -= gap

    put("DIAGNOSIS", bold=True, color=color, size=11, gap=0.012)
    y_cur -= 0.006
    for line in textwrap.wrap(ex["diagnosis"], 66):
        put(line, size=8.8, gap=0.032)
    y_cur -= 0.018

    put("CORRECTION", bold=True, color=color, size=11, gap=0.012)
    y_cur -= 0.006
    for line in textwrap.wrap(ex["correction"], 66):
        put(line, size=8.8, gap=0.032)
    y_cur -= 0.018

    put("CORRECTION VECTOR  (analytically computed from measured features)",
        bold=True, color="#444444", size=9.2, gap=0.012)
    y_cur -= 0.006
    for k, v in cv.items():
        put(f"  {k:<28} = {v:+.5f}", size=8.8, color="#222222",
            indent=0.05, gap=0.028)

    # arrow pointing right to corrected waveform
    ax2.annotate("", xy=(1.04, 0.50), xytext=(0.97, 0.50),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#1B5E20",
                                 lw=2.0, mutation_scale=18))
    ax2.text(1.05, 0.52, "apply", transform=ax2.transAxes,
             fontsize=8, color="#1B5E20", fontweight="bold",
             ha="left", va="bottom")
    ax2.text(1.05, 0.47, "CV", transform=ax2.transAxes,
             fontsize=8, color="#1B5E20", fontweight="bold",
             ha="left", va="top")

    # ── panel 3: corrected waveform (analytically predicted) ─────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(t_corr, y_corr, color="#1B5E20", linewidth=2.0, alpha=0.9)
    ax3.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Time (ms)", fontsize=10)
    ax3.set_ylabel("Displacement (nm)", fontsize=10)
    ax3.grid(True, alpha=0.2, linewidth=0.5)
    for sp in ax3.spines.values():
        sp.set_edgecolor("#1B5E20")
        sp.set_linewidth(1.5)

    ax3.set_title(
        "Predicted After Correction\n(analytically computed)",
        fontsize=10, fontweight="bold", color="#1B5E20", pad=8,
    )

    # improvement stats box
    if wtype == "sine":
        h2_after = abs(f["h2"] + cv["harmonic_2_amp"])
        h3_after = abs(f["h3"] + cv["harmonic_3_amp"])
        thd_after = math.sqrt(h2_after**2 + h3_after**2) * 100
        stats = (
            f"Phase lag:  {f['phase_lag']:.1f}° → ~0°\n"
            f"THD:  {f['thd']:.2f}% → {thd_after:.3f}%\n"
            f"Harmonics:  cancelled\n"
            f"Amp scale:  ×{cv['amplitude_scale']:.4f}"
        )
    else:
        stats = (
            f"Duty cycle:  {f['duty_cycle']:.4f} → "
            f"{f['duty_cycle']+cv['duty_cycle_trim']:.4f}\n"
            f"Edge sharp:  {f['edge_sharpness']:.4f} → "
            f"{f['edge_sharpness']+cv['edge_sharpness_deficit']:.4f}\n"
            f"2nd harmonic:  cancelled\n"
            f"THD:  {f['thd']:.2f}% → ~48.3% (ideal)"
        )

    ax3.text(0.04, 0.97, stats, transform=ax3.transAxes,
             fontsize=8.2, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                       edgecolor="#A5D6A7", alpha=0.95))

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Zamba2-7B-instruct  |  Piezoelectric Actuator Waveform Diagnosis  "
        f"|  {wtype.upper()} at {int(f['freq'])} Hz",
        fontsize=12, fontweight="bold", color="#111111", y=0.95,
    )

    out_path = OUT / f"{wtype}_paper.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


for wtype in ["sine", "square"]:
    make_figure(wtype)

print("Done.")
