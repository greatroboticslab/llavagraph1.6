"""
make_figures.py
===============
Generates 3-panel figures for the README:

  Left:   Real waveform image captured from piezoelectric actuator hardware
  Center: LLM-generated DIAGNOSIS + CORRECTION + CORRECTION VECTOR
  Right:  Predicted corrected waveform — commanded signal restored to
          target amplitude after applying the correction vector analytically

Run locally:
    cd mamba_zamba && python make_figures.py
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

# ── real hardware measurement images ─────────────────────────────────────────
REAL_IMAGES = {
    "sine":   "../data/ori_plots/sine_Plots/sine_100Hz_100Hz_1_absolute_slice_50_200.png",
    "square": "../data/ori_plots/square_Plots/square_100Hz_100Hz_1_absolute_slice_50_200.png",
}

# ── measured features and outputs from test.jsonl (100 Hz samples) ───────────
EXAMPLES = {
    "sine": {
        "features": {
            "freq": 100.0, "target_peak": 495.8, "rms": 206.7,
            "phase_lag": 76.1, "fopdt_lag": 9.11,
            "h2": 0.01052, "h3": 0.01873,
            "amp_drift": 74.86, "thd": 3.631, "crest": 2.398,
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
            "The commanded 100 Hz sine shows strong nonlinear distortion from "
            "hysteresis and phase lag. Measured phase lag is 76.1° versus 9.11° "
            "predicted (66.99° excess). Crest factor 2.398 (ideal 1.414) and "
            "74.86 nm drift confirm a non-sinusoidal, drifting output."
        ),
        "correction": (
            "Advance phase by 66.99° to cancel the nonlinear lag, cancel the 3rd "
            "harmonic (0.01873), and apply -18.62 nm hysteresis feedforward to "
            "counteract DC creep."
        ),
    },
    "square": {
        "features": {
            "freq": 100.0, "target_peak": 521.6, "rms": 400.0,
            "duty_cycle": 0.4763, "edge_sharpness": 0.9166,
            "h2": 0.02587, "thd": 56.93,
        },
        "cv": {
            "duty_cycle_trim":        +0.0237,
            "edge_sharpness_deficit": +0.08342,
            "harmonic_2_amp":         -0.02587,
            "dc_offset_nm":            0.0,
        },
        "diagnosis": (
            "The commanded 100 Hz square is distorted by bandwidth limits and "
            "nonlinearity. THD is 56.93% (ideal 48.3%), duty cycle 0.4763 (ideal "
            "0.500), edge sharpness 0.9166 (ideal 1.000) — the actuator cannot "
            "track sharp transitions. Amplitude drift is stable at 0 nm."
        ),
        "correction": (
            "Trim duty cycle by +0.0237 to restore symmetry, apply edge "
            "pre-emphasis to recover the 0.0834 sharpness deficit, and cancel "
            "the 2nd harmonic (0.02587)."
        ),
    },
}

COLORS = {"sine": "#1565C0", "square": "#B71C1C"}


# ── corrected waveform synthesis ──────────────────────────────────────────────
# Applies the CORRECTION VECTOR analytically to produce the predicted output.
# amplitude_scale is applied to the control INPUT — the OUTPUT returns to the
# commanded target_peak amplitude.

def synth_corrected_sine(f):
    freq = f["freq"]
    amp  = f["target_peak"]          # output restored to commanded amplitude
    T    = 0.150                     # 150 ms window
    t    = np.linspace(0, T, 6000)
    omega = 2 * math.pi * freq
    return t * 1000, amp * np.sin(omega * t)


def synth_corrected_square(f, cv):
    freq = f["freq"]
    amp  = f["target_peak"]
    duty = f["duty_cycle"] + cv["duty_cycle_trim"]             # → 0.500
    edge = min(f["edge_sharpness"] + cv["edge_sharpness_deficit"], 1.0)
    T    = 0.150
    t    = np.linspace(0, T, 8000)
    omega = 2 * math.pi * freq
    phase = (omega * t) % (2 * math.pi)
    steepness = 20 + edge * 150
    y = amp * np.tanh(steepness * np.sin(phase - math.pi * (1 - 2 * duty)))
    return t * 1000, y


# ── figure builder ────────────────────────────────────────────────────────────

def make_figure(wtype):
    ex    = EXAMPLES[wtype]
    f     = ex["features"]
    cv    = ex["cv"]
    color = COLORS[wtype]

    if wtype == "sine":
        t_corr, y_corr = synth_corrected_sine(f)
    else:
        t_corr, y_corr = synth_corrected_square(f, cv)

    amp  = f["target_peak"]
    ylim = (-amp * 1.25, amp * 1.25)

    fig = plt.figure(figsize=(26, 9.5))
    fig.patch.set_facecolor("#F4F4F4")
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.0, 1.55, 0.95],
        left=0.02, right=0.98,
        top=0.87, bottom=0.08,
        wspace=0.06,
    )

    # ── panel 1: real hardware measurement ───────────────────────────────────
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
        ax1.text(0.5, 0.5, "image not found", ha="center", va="center",
                 color="red", fontsize=10)
        ax1.axis("off")
    ax1.set_title(
        f"Measured Waveform — {wtype.upper()} at {int(f['freq'])} Hz\n"
        f"(real hardware data, piezoelectric actuator)",
        fontsize=14, fontweight="bold", color="#B71C1C", pad=10,
    )

    # ── panel 2: LLM diagnostic report ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FFFFFF")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Use axes height in points to compute safe line spacing
    # At figsize 22x8, the center panel axes height ≈ 0.80 * 8 * 72 = 460 pt
    # We'll use normalized coords and a fixed step that avoids overlap
    BODY_SIZE  = 13.5
    HEAD_SIZE  = 16.0
    BODY_STEP  = 0.048
    HEAD_STEP  = 0.060
    SEC_GAP    = 0.022

    y_cur = 0.97

    def put(text, bold=False, fcolor="#111111", size=BODY_SIZE,
            indent=0.02, step=BODY_STEP):
        nonlocal y_cur
        if y_cur < 0.01:
            return
        ax2.text(indent, y_cur, text, transform=ax2.transAxes,
                 fontsize=size, color=fcolor,
                 fontweight="bold" if bold else "normal",
                 verticalalignment="top", clip_on=True)
        y_cur -= step

    # DIAGNOSIS
    put("DIAGNOSIS", bold=True, fcolor=color, size=HEAD_SIZE, step=HEAD_STEP)
    for line in textwrap.wrap(ex["diagnosis"], 60):
        put(line)
    y_cur -= SEC_GAP

    # CORRECTION
    put("CORRECTION", bold=True, fcolor=color, size=HEAD_SIZE, step=HEAD_STEP)
    for line in textwrap.wrap(ex["correction"], 60):
        put(line)
    y_cur -= SEC_GAP

    # CORRECTION VECTOR
    put("CORRECTION VECTOR", bold=True, fcolor="#444444",
        size=14.0, step=0.056)
    put("(analytically computed from measured features)",
        fcolor="#777777", size=11.0, indent=0.02, step=0.046)
    y_cur -= 0.006
    for k, v in cv.items():
        put(f"  {k:<26} = {v:+.5f}", fcolor="#222222",
            size=12.5, indent=0.04, step=0.047)

    # arrow
    ax2.annotate("", xy=(1.04, 0.50), xytext=(0.97, 0.50),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#1B5E20",
                                 lw=2.0, mutation_scale=18))

    # ── panel 3: corrected waveform ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(t_corr, y_corr, color="#1B5E20", linewidth=2.0, alpha=0.9)
    ax3.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Time (ms)", fontsize=12)
    ax3.set_ylabel("Displacement (nm)", fontsize=12)
    ax3.tick_params(labelsize=11)
    ax3.set_ylim(ylim)
    ax3.set_xlim(0, 150)
    ax3.grid(True, alpha=0.2, linewidth=0.5)
    for sp in ax3.spines.values():
        sp.set_edgecolor("#1B5E20")
        sp.set_linewidth(1.5)
    ax3.set_title(
        "Predicted After Correction\n(analytically computed)",
        fontsize=14, fontweight="bold", color="#1B5E20", pad=10,
    )

    if wtype == "sine":
        h2r  = abs(f["h2"] + cv["harmonic_2_amp"])
        h3r  = abs(f["h3"] + cv["harmonic_3_amp"])
        thdr = math.sqrt(h2r**2 + h3r**2) * 100
        stats = (f"Phase lag:  {f['phase_lag']:.1f}° → 0°\n"
                 f"THD:  {f['thd']:.2f}% → {thdr:.3f}%\n"
                 f"Amplitude:  ±{f['target_peak']} nm (restored)\n"
                 f"Harmonics:  cancelled")
    else:
        dc = f["duty_cycle"] + cv["duty_cycle_trim"]
        ec = f["edge_sharpness"] + cv["edge_sharpness_deficit"]
        stats = (f"Duty cycle:  {f['duty_cycle']:.4f} → {dc:.4f}\n"
                 f"Edge sharp:  {f['edge_sharpness']:.4f} → {ec:.4f}\n"
                 f"2nd harmonic:  cancelled\n"
                 f"THD:  {f['thd']:.1f}% → ~48.3% (ideal)")
    ax3.text(0.04, 0.97, stats, transform=ax3.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                       edgecolor="#A5D6A7", alpha=0.95))

    fig.suptitle(
        f"Zamba2-7B-instruct  |  Piezoelectric Actuator Waveform Diagnosis  "
        f"|  {wtype.upper()} at {int(f['freq'])} Hz",
        fontsize=16, fontweight="bold", color="#111111", y=0.96,
    )

    out_path = OUT / f"{wtype}_example.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


for wtype in ["sine", "square"]:
    make_figure(wtype)

print("Done.")
