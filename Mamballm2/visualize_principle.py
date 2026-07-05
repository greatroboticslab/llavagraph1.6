"""
visualize_principle.py
======================
Explains the underlying principle:
  1. What "ground truth" actually means
  2. Where each correction value comes from (signal processing, not guessing)
  3. Simulation at the CORRECT time scale (matching real images: 480-740ms, 26 cycles)
  4. How each error in the waveform maps to one correction parameter

Run:
    python visualize_principle.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

np.random.seed(7)

# ── Colors ────────────────────────────────────────────────────────────────────
BG    = "#0d0d1a"
BG2   = "#14142b"
BLUE  = "#00c8ff"
GREEN = "#00ff9f"
RED   = "#ff4d6d"
YLLOW = "#ffd60a"
GRAY  = "#8892a4"
WHITE = "#e8eaf0"
PURP  = "#c77dff"
ORNG  = "#ff9a3c"

# ── Real measured values from features_v3.csv (sine-100Hz-10) ─────────────────
PARAMS = {
    "f_hz":            100,
    "A_nm":            279.80,
    "phase_lag_deg":   -8.96,    # measured
    "fopdt_phase_deg":  9.10,    # FOPDT linear model prediction
    "h3_ratio":         0.031,   # 3rd harmonic / fundamental
    "h2_ratio":         0.010,   # 2nd harmonic / fundamental
    "dc_nm":           -2.10,    # DC offset
    "hyst_nm":          15.83,   # hysteresis half-cycle asymmetry
    "rms_nm":           200.76,  # measured RMS
    "thd_pct":          5.38,    # THD
}

# CORRECT time window: 480-740ms (same as real images)
fs    = 5000
t     = np.linspace(0.480, 0.740, int(0.260 * fs))   # 260ms window
f     = PARAMS["f_hz"]
A     = PARAMS["A_nm"]
w     = 2 * np.pi * f
phi   = np.deg2rad(PARAMS["phase_lag_deg"])

# ── Waveforms ─────────────────────────────────────────────────────────────────
y_ideal = A * np.sin(w * t)

hyst = PARAMS["hyst_nm"] * 0.40 * np.sign(np.cos(w * t))
y_meas = (A * np.sin(w * t + phi)
          + A * PARAMS["h3_ratio"] * np.sin(3 * w * t + phi)
          + A * PARAMS["h2_ratio"] * np.sin(2 * w * t + phi)
          + PARAMS["dc_nm"]
          + hyst
          + np.random.normal(0, A * 0.010, len(t)))

# Decompose the ERROR into individual components
err_phase  = A * np.sin(w * t)        - A * np.sin(w * t + phi)
err_h3     = - A * PARAMS["h3_ratio"] * np.sin(3 * w * t + phi)
err_h2     = - A * PARAMS["h2_ratio"] * np.sin(2 * w * t + phi)
err_dc     = - PARAMS["dc_nm"] * np.ones(len(t))
err_hyst   = -hyst
delta_u    = err_phase + err_h3 + err_h2 + err_dc + err_hyst

y_corrected = y_meas + 0.75 * delta_u + np.random.normal(0, A * 0.005, len(t))

t_ms = t * 1000   # convert to ms for x-axis

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.suptitle(
    "Understanding the Principle: How LLM Correction Vector Fixes the Entire Waveform",
    color=WHITE, fontsize=14, fontweight="bold", y=0.98
)

outer = gridspec.GridSpec(3, 1, figure=fig,
                          height_ratios=[1.6, 1.4, 1.0],
                          hspace=0.40,
                          left=0.05, right=0.97, top=0.95, bottom=0.04)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 0: Real image vs simulation (correct time scale)
# ══════════════════════════════════════════════════════════════════════════════
row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.08)

# Left: load the real PNG
ax_real = fig.add_subplot(row0[0])
try:
    img = mpimg.imread("../vit_mamba/data/train/sine/sine-100Hz-10_absolute_aug_v3.png")
    ax_real.imshow(img)
    ax_real.axis("off")
    ax_real.set_title(
        "REAL image (sine-100Hz-10)  |  window 480-740ms  |  26 cycles visible",
        color=GREEN, fontsize=9, fontweight="bold"
    )
except Exception:
    ax_real.text(0.5, 0.5, "Image not found", transform=ax_real.transAxes,
                 ha="center", va="center", color=GRAY)
    ax_real.axis("off")

# Right: simulation with CORRECT time scale
ax_sim = fig.add_subplot(row0[1])
ax_sim.set_facecolor(BG2)
ax_sim.plot(t_ms, y_ideal, color=GREEN, lw=1.4, label="Ideal reference", zorder=3)
ax_sim.plot(t_ms, y_meas,  color=RED,   lw=1.0, alpha=0.85, label="Simulated measured", zorder=2)
ax_sim.set_xlim(t_ms[0], t_ms[-1])
ax_sim.set_xlabel("Time (ms)", color=GRAY, fontsize=8)
ax_sim.set_ylabel("Displacement (nm)", color=GRAY, fontsize=8)
ax_sim.tick_params(colors=GRAY, labelsize=7)
for sp in ax_sim.spines.values():
    sp.set_edgecolor("#333")
ax_sim.legend(fontsize=8, facecolor=BG, labelcolor=GRAY, loc="upper right")
ax_sim.set_title(
    "SIMULATION at correct time scale  |  same 480-740ms window  |  26 cycles\n"
    f"Errors added: phase lag {PARAMS['phase_lag_deg']}°, "
    f"3rd harmonic {PARAMS['h3_ratio']*100:.1f}%, "
    f"hysteresis ±{PARAMS['hyst_nm']:.1f} nm, "
    f"DC {PARAMS['dc_nm']:.1f} nm",
    color=RED, fontsize=8.5, fontweight="bold"
)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Decompose each error + its correction
# ══════════════════════════════════════════════════════════════════════════════
row1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1],
                                        wspace=0.25, hspace=0.45)

# Helper: zoom into 2 cycles only for clarity in decomposition panels
mask = (t >= 0.480) & (t < 0.480 + 2.0/f)   # first 2 cycles
t_z  = t_ms[mask]

errors = [
    ("Phase Lag\n(−8.96°)",        err_phase[mask],  y_meas[mask],  y_ideal[mask],
     BLUE,  "phase_offset_deg = +8.96",
     "Formula: correction = −measured_phase_lag\n= −(−8.96°) = +8.96°\n"
     "Source: sine_phase_lag_deg column\nin features_v3.csv\n\n"
     "This time-shifts the reference signal\nforward so peaks align perfectly."),

    ("3rd Harmonic\n(ripple 3.1%)", err_h3[mask],    y_meas[mask],  y_ideal[mask],
     ORNG,  "harmonic_3_amp = −0.031",
     "Formula: correction = −harmonic_ratio_3\n= −0.031\n"
     "Source: harmonic_ratio_3 column\nin features_v3.csv\n\n"
     "This subtracts the ripple at 3×100=300Hz,\nremoving the distortion from the wave."),

    ("Hysteresis\n(±15.83 nm)",     err_hyst[mask],  y_meas[mask],  y_ideal[mask],
     PURP,  "hysteresis_ff_nm = −15.83",
     "Formula: correction = −hysteresis_nm\n= −15.83 nm per half-cycle\n"
     "Source: hysteresis_nm column\nin features_v3.csv\n\n"
     "Adds opposite-sign force on rising vs\nfalling stroke to cancel the dead-band."),

    ("DC Offset\n(−2.10 nm)",       err_dc[mask],    y_meas[mask],  y_ideal[mask],
     YLLOW, "dc_offset_nm = +2.10",
     "Formula: correction = −mean_displacement\n= −(−2.10) = +2.10 nm\n"
     "Source: q1_mean_nm...q4_mean_nm average\nin features_v3.csv\n\n"
     "Adds a constant offset to center\nthe waveform at zero baseline."),
]

for col, (title, err_sig, ym, yi, color, param_str, explanation) in enumerate(errors):
    # Top: the error signal + what the correction looks like
    ax_top = fig.add_subplot(row1[0, col])
    ax_top.set_facecolor(BG2)
    ax_top.plot(t_z, yi[: len(t_z)], color=GREEN, lw=1.2, alpha=0.5, label="ideal")
    ax_top.plot(t_z, ym[: len(t_z)], color=RED,   lw=1.0, alpha=0.7, label="measured")
    ax_top.plot(t_z, err_sig,         color=color, lw=1.6, linestyle="--",
                label="correction signal\n(what we ADD)")
    ax_top.set_xlim(t_z[0], t_z[-1])
    ax_top.tick_params(colors=GRAY, labelsize=6)
    for sp in ax_top.spines.values():
        sp.set_edgecolor("#333")
    ax_top.set_title(title, color=color, fontsize=8.5, fontweight="bold")
    ax_top.legend(fontsize=5.5, facecolor=BG, labelcolor=GRAY, loc="lower right")
    if col == 0:
        ax_top.set_ylabel("nm", color=GRAY, fontsize=7)

    # Bottom: explanation text
    ax_bot = fig.add_subplot(row1[1, col])
    ax_bot.set_facecolor(BG2)
    ax_bot.axis("off")
    ax_bot.text(0.05, 0.95, param_str,
                transform=ax_bot.transAxes, va="top", color=color,
                fontsize=8, fontweight="bold", family="monospace")
    ax_bot.text(0.05, 0.72, explanation,
                transform=ax_bot.transAxes, va="top", color=GRAY, fontsize=7.2)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Final corrected waveform + Ground truth concept
# ══════════════════════════════════════════════════════════════════════════════
row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.10)

# Left: before vs after (full 260ms window)
ax_final = fig.add_subplot(row2[0])
ax_final.set_facecolor(BG2)
ax_final.plot(t_ms, y_ideal,     color=GREEN, lw=1.4, alpha=0.9,  label="Ideal reference", zorder=4)
ax_final.plot(t_ms, y_meas,      color=RED,   lw=0.9, alpha=0.65, label="Measured (with errors)", zorder=2)
ax_final.plot(t_ms, y_corrected, color=BLUE,  lw=1.2, alpha=0.90, label="After correction (1 iteration)", zorder=3)
ax_final.set_xlim(t_ms[0], t_ms[-1])
ax_final.set_xlabel("Time (ms)", color=GRAY, fontsize=8)
ax_final.set_ylabel("Displacement (nm)", color=GRAY, fontsize=8)
ax_final.tick_params(colors=GRAY, labelsize=7)
for sp in ax_final.spines.values():
    sp.set_edgecolor("#333")
ax_final.legend(fontsize=8, facecolor=BG, labelcolor=GRAY, loc="upper right")

rms_before = np.sqrt(np.mean((y_meas - y_ideal) ** 2))
rms_after  = np.sqrt(np.mean((y_corrected - y_ideal) ** 2))
ax_final.set_title(
    f"Result: Entire Waveform Corrected (all 26 cycles)\n"
    f"RMS error: {rms_before:.1f} nm  -->  {rms_after:.1f} nm  "
    f"({(1 - rms_after / rms_before) * 100:.0f}% reduction, 1 iteration)",
    color=BLUE, fontsize=9, fontweight="bold"
)

# Right: What "ground truth" means
ax_gt = fig.add_subplot(row2[1])
ax_gt.set_facecolor(BG2)
ax_gt.axis("off")
ax_gt.set_title("What is 'Ground Truth'?", color=YLLOW, fontsize=10, fontweight="bold")

gt_text = """In machine learning, every training sample has:
   INPUT  -->  the waveform image + measured features
   OUTPUT -->  the correct answer (= "ground truth")

The LLM LEARNS to predict the output from the input.
After training, it can produce good outputs for NEW images.

For our system, the "correct answer" is the correction vector:

   phase_offset_deg  =  +8.96    <-- computed from features_v3.csv
   harmonic_3_amp    =  -0.031   <-- computed from features_v3.csv
   hysteresis_ff_nm  =  -15.83   <-- computed from features_v3.csv
   dc_offset_nm      =  +2.10    <-- computed from features_v3.csv

WHY can we compute ground truth from features_v3.csv?
Because each error has a known mathematical inverse:
   Error: phase lag of X deg  -->  Correction: advance by X deg
   Error: harmonic at ratio r  -->  Correction: subtract ratio r
   Error: hysteresis H nm      -->  Correction: feedforward -H nm
   Error: DC offset D nm       -->  Correction: add -D nm

These are NOT guesses -- they follow directly from signal
processing theory. The LLM learns to extract these values
from visual + textual information, then apply them."""

ax_gt.text(0.04, 0.96, gt_text,
           transform=ax_gt.transAxes, va="top", color=GRAY,
           fontsize=8.2, family="monospace",
           bbox=dict(facecolor="#0a0a1f", edgecolor=YLLOW,
                     boxstyle="round,pad=0.5"))

out = "./principle_visualization.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")
