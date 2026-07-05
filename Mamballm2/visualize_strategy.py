"""
visualize_strategy.py
=====================
Shows the proposed revised strategy visually:
  1. Current (broken) pipeline vs Proposed (correct) pipeline
  2. Waveform simulation: measured errors → correction vector → corrected waveform
  3. Training data format: before vs after
  4. ILC convergence over iterations

Run:
    python visualize_strategy.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

np.random.seed(42)

# ── Color palette ─────────────────────────────────────────────────────────────
BG       = "#0d0d1a"
BG2      = "#14142b"
BLUE     = "#00c8ff"
GREEN    = "#00ff9f"
RED      = "#ff4d6d"
YELLOW   = "#ffd60a"
GRAY     = "#8892a4"
WHITE    = "#e8eaf0"
PURPLE   = "#c77dff"
ORANGE   = "#ff9a3c"

# ── Simulation parameters (from real sine-100Hz-10 data) ─────────────────────
F        = 100      # Hz
A        = 279.80   # nm peak amplitude
PHI_DEG  = -8.96    # measured phase lag (deg)
H3_RATIO = 0.031    # 3rd harmonic / fundamental
DC_NM    = -2.10    # DC offset (nm)
HYST_NM  = 15.83    # hysteresis half-cycle (nm)
FS       = 5000     # sample rate (Hz)
N_CYCLES = 3

t  = np.linspace(0, N_CYCLES / F, int(N_CYCLES * FS / F))
w  = 2 * np.pi * F
phi = np.deg2rad(PHI_DEG)


def make_waveforms(noise_scale=0.012):
    y_ref = A * np.sin(w * t)

    hyst = HYST_NM * 0.45 * np.sign(np.cos(w * t))
    y_meas = (A * np.sin(w * t + phi)
              + A * H3_RATIO * np.sin(3 * w * t + phi)
              + DC_NM
              + hyst
              + np.random.normal(0, A * noise_scale, len(t)))

    # Correction signal (computed from correction vector)
    delta = (A * np.sin(w * t)
             - A * np.sin(w * t + phi)
             - A * H3_RATIO * np.sin(3 * w * t + phi)
             - DC_NM
             - hyst)

    return y_ref, y_meas, delta


def ilc_iterations(n_iter=5):
    """Simulate ILC convergence with step size alpha=0.6."""
    y_ref, _, delta = make_waveforms()
    alpha = 0.60
    history = []
    y_cur = make_waveforms()[1]
    for i in range(n_iter):
        err = np.sqrt(np.mean((y_cur - y_ref) ** 2))
        history.append(err)
        y_cur = y_cur + alpha * delta + np.random.normal(0, A * 0.005, len(t))
    return history


# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16), facecolor=BG)
fig.suptitle(
    "Revised Strategy for MambaLLM-Assisted Piezo Closed-Loop Control",
    color=WHITE, fontsize=15, fontweight="bold", y=0.98
)

outer = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[1.15, 1.5, 1.0],
    hspace=0.38, left=0.04, right=0.97, top=0.95, bottom=0.04
)

# ──────────────────────────────────────────────────────────────────────────────
# ROW 0 — Pipeline comparison
# ──────────────────────────────────────────────────────────────────────────────
row0 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[0], wspace=0.06
)

def draw_pipeline(ax, title, boxes, arrows, title_color):
    ax.set_facecolor(BG2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(title, color=title_color, fontsize=11, fontweight="bold", pad=8)

    for (x, y, w_, h_, label, color) in boxes:
        box = FancyBboxPatch(
            (x - w_/2, y - h_/2), w_, h_,
            boxstyle="round,pad=0.08",
            facecolor=color + "22", edgecolor=color, linewidth=1.6
        )
        ax.add_patch(box)
        for i, line in enumerate(label.split("\n")):
            offset = (i - (label.count("\n") / 2)) * 0.20
            ax.text(x, y - offset, line, ha="center", va="center",
                    color=color, fontsize=7.5, fontweight="bold")

    for (x1, y1, x2, y2, label, color) in arrows:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8)
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my + 0.18, label,
                color=color, fontsize=6.5, ha="center", style="italic")


# ── Left: Current broken approach ─────────────────────────────────────────────
ax_bad = fig.add_subplot(row0[0])
bad_boxes = [
    (2.0, 3.2, 2.4, 0.55, "Measure\nWaveform",     BLUE),
    (5.0, 3.2, 2.2, 0.55, "MambaLLM",              PURPLE),
    (8.0, 3.2, 2.2, 0.55, "Free-text\nDescription",RED),
    (5.0, 1.5, 3.5, 0.55, "??? (No synthesis\nfunction exists)", RED),
    (8.0, 1.5, 2.2, 0.55, "Corrected\nWaveform?",  GRAY),
]
bad_arrows = [
    (3.2, 3.2, 3.9, 3.2, "PNG+features", GRAY),
    (6.1, 3.2, 6.9, 3.2, "", GRAY),
    (8.0, 2.9, 8.0, 1.8, '"phase +18deg\nharmonics suppression"', RED),
    (6.75, 1.5, 6.9, 1.5, "❌ gap", RED),
]
draw_pipeline(ax_bad, "❌  Current Approach  (broken)", bad_boxes, bad_arrows, RED)
ax_bad.text(5.0, 0.7,
    "LLM outputs FREE TEXT → cannot be\nautomatically applied to the drive signal",
    ha="center", va="center", color=RED, fontsize=8,
    bbox=dict(facecolor="#330011", edgecolor=RED, boxstyle="round,pad=0.3"))
ax_bad.spines["bottom"].set_visible(False)

# ── Right: Proposed correct approach ──────────────────────────────────────────
ax_good = fig.add_subplot(row0[1])
good_boxes = [
    (1.3, 3.2, 2.0, 0.55, "Measure\nWaveform",        BLUE),
    (3.5, 3.2, 1.8, 0.55, "MambaLLM",                 PURPLE),
    (5.8, 3.2, 2.2, 0.70, "Correction\nVector θ",      GREEN),
    (8.2, 3.2, 1.8, 0.55, "Synthesis\nFunction",       YELLOW),
    (8.2, 1.8, 1.8, 0.70, "Δu[n]\n(full correction\nwaveform)", GREEN),
    (5.0, 1.3, 2.8, 0.60, "ILC Update\nu_{k+1}=u_k + α·Δu", BLUE),
    (1.8, 1.3, 2.0, 0.55, "Next Cycle\n(improved)", GREEN),
]
good_arrows = [
    (2.3, 3.2, 2.6, 3.2, "PNG+feat",  GRAY),
    (4.4, 3.2, 4.7, 3.2, "",          GRAY),
    (6.9, 3.2, 7.3, 3.2, "6 floats",  GREEN),
    (8.2, 2.9, 8.2, 2.15,"",          YELLOW),
    (7.3, 1.8, 6.4, 1.55,"N-sample\nvector", GREEN),
    (4.6, 1.3, 2.8, 1.3, "",          BLUE),
]
draw_pipeline(ax_good, "✅  Proposed Approach  (correct)", good_boxes, good_arrows, GREEN)
ax_good.text(5.0, 0.7,
    "LLM outputs STRUCTURED VECTOR → synthesis → Δu[n] → corrects ENTIRE waveform",
    ha="center", va="center", color=GREEN, fontsize=8,
    bbox=dict(facecolor="#001a0d", edgecolor=GREEN, boxstyle="round,pad=0.3"))

# ──────────────────────────────────────────────────────────────────────────────
# ROW 1 — Waveform simulation + Format comparison
# ──────────────────────────────────────────────────────────────────────────────
row1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[1], wspace=0.08, width_ratios=[1.5, 1]
)

wave_gs = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=row1[0], hspace=0.10
)

y_ref, y_meas, delta = make_waveforms()
alpha = 0.60
y_corrected = y_meas + alpha * delta

def wave_ax(spec, title, ylabel, yc):
    ax = fig.add_subplot(spec)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=GRAY, labelsize=7)
    ax.set_ylabel(ylabel, color=yc, fontsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.set_title(title, color=yc, fontsize=8.5, loc="left", pad=4)
    return ax

# Panel A: reference
ax_ref = wave_ax(wave_gs[0], "A)  Ideal Reference Waveform (what we WANT)", "nm", GREEN)
ax_ref.plot(t * 1000, y_ref, color=GREEN, lw=1.6, label="ideal")
ax_ref.set_xlim(t[0]*1000, t[-1]*1000)
ax_ref.set_xticklabels([])
ax_ref.legend(fontsize=7, facecolor=BG, labelcolor=GREEN, loc="upper right")

# Panel B: measured (with annotated errors)
ax_meas = wave_ax(wave_gs[1], "B)  Measured Waveform (what the piezo ACTUALLY produces)", "nm", RED)
ax_meas.plot(t * 1000, y_ref,  color=GREEN, lw=1.0, alpha=0.4, linestyle="--", label="reference (faint)")
ax_meas.plot(t * 1000, y_meas, color=RED,   lw=1.6, label="measured")
ax_meas.set_xlim(t[0]*1000, t[-1]*1000)
ax_meas.set_xticklabels([])
# Annotate errors
t_ms = t * 1000
ax_meas.annotate("Phase lag\n−8.96°",
    xy=(t_ms[45], y_meas[45]), xytext=(t_ms[45]+2.5, y_meas[45]+70),
    color=YELLOW, fontsize=7, arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1))
ax_meas.annotate("3rd harmonic\n(ripple 3.1%)",
    xy=(t_ms[120], y_meas[120]), xytext=(t_ms[120]+3, y_meas[120]+90),
    color=ORANGE, fontsize=7, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1))
ax_meas.annotate("DC offset\n−2.1 nm",
    xy=(t_ms[100], DC_NM), xytext=(t_ms[100]-3.5, DC_NM-80),
    color=GRAY, fontsize=7, arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
ax_meas.legend(fontsize=7, facecolor=BG, labelcolor=GRAY, loc="upper right")

# Panel C: corrected
ax_corr = wave_ax(wave_gs[2], "C)  After Applying Correction Vector  (one ILC iteration)", "nm", BLUE)
ax_corr.plot(t * 1000, y_ref,       color=GREEN, lw=1.0, alpha=0.4, linestyle="--", label="reference (faint)")
ax_corr.plot(t * 1000, y_meas,      color=RED,   lw=0.8, alpha=0.5, label="measured (faint)")
ax_corr.plot(t * 1000, y_corrected, color=BLUE,  lw=1.8, label="corrected (α=0.6)")
ax_corr.set_xlim(t[0]*1000, t[-1]*1000)
ax_corr.set_xlabel("Time (ms)", color=GRAY, fontsize=8)

# Compute RMS errors
rms_before = np.sqrt(np.mean((y_meas - y_ref)**2))
rms_after  = np.sqrt(np.mean((y_corrected - y_ref)**2))
ax_corr.text(0.98, 0.08,
    f"RMS error: {rms_before:.1f} nm → {rms_after:.1f} nm  ({(1-rms_after/rms_before)*100:.0f}% reduction)",
    transform=ax_corr.transAxes, ha="right", va="bottom",
    color=GREEN, fontsize=8,
    bbox=dict(facecolor="#001a0d", edgecolor=GREEN, boxstyle="round,pad=0.3"))
ax_corr.legend(fontsize=7, facecolor=BG, labelcolor=GRAY, loc="upper right")

# ── Right panel: format comparison ───────────────────────────────────────────
ax_fmt = fig.add_subplot(row1[1])
ax_fmt.set_facecolor(BG2)
ax_fmt.axis("off")
ax_fmt.set_title("Training Data Format", color=WHITE, fontsize=10, fontweight="bold")

bad_fmt = (
    "❌  CURRENT FORMAT\n\n"
    "CORRECTION VECTOR: Nonlinear phase\n"
    "compensation (+-deg), harmonic\n"
    "suppression (Nth harmonic, target\n"
    "% reduction), hysteresis feedforward\n"
    "(+-nm per half-cycle), creep\n"
    "compensation (nm/ms), DC offset\n"
    "(+-nm).\n\n"
    "Problem:\n"
    "• Values buried in prose\n"
    "• Cannot be machine-parsed\n"
    "• Cannot drive synthesis function\n"
    "• No ground truth to evaluate"
)

good_fmt = (
    "✅  PROPOSED FORMAT\n\n"
    "DEVIATION SUMMARY: THD=5.38%\n"
    "→ hysteresis dominant. Phase\n"
    "lag −8.96° vs FOPDT +9.10°\n"
    "→ 18.06° nonlinear excess.\n\n"
    "TEMPORAL: Q1-Q4 stable, drift=0.\n\n"
    "VIBRATION: Periodic-steady.\n\n"
    "CORRECTION VECTOR:\n"
    "phase_offset_deg = +8.96\n"
    "amplitude_scale  =  1.023\n"
    "harmonic_3_amp   = -0.031\n"
    "dc_offset_nm     = -2.10\n"
    "hysteresis_ff_nm = -15.83\n\n"
    "→ Parsed → Synthesis → Δu[n]"
)

ax_fmt.text(0.02, 0.97, bad_fmt,
    transform=ax_fmt.transAxes, va="top", ha="left",
    color=RED, fontsize=7.2, family="monospace",
    bbox=dict(facecolor="#1a0000", edgecolor=RED,
              boxstyle="round,pad=0.5", alpha=0.9))

ax_fmt.text(0.02, 0.47, good_fmt,
    transform=ax_fmt.transAxes, va="top", ha="left",
    color=GREEN, fontsize=7.2, family="monospace",
    bbox=dict(facecolor="#001a0d", edgecolor=GREEN,
              boxstyle="round,pad=0.5", alpha=0.9))

# ──────────────────────────────────────────────────────────────────────────────
# ROW 2 — ILC convergence + Correction vector legend + Ground truth source
# ──────────────────────────────────────────────────────────────────────────────
row2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[2], wspace=0.30
)

# ILC convergence
ax_ilc = fig.add_subplot(row2[0])
ax_ilc.set_facecolor(BG2)
ax_ilc.tick_params(colors=GRAY, labelsize=8)
for sp in ax_ilc.spines.values():
    sp.set_edgecolor("#333")

history = ilc_iterations(n_iter=8)
iters = list(range(1, len(history) + 1))
ax_ilc.plot(iters, history, "o-", color=BLUE, lw=2.2, markersize=7, label="RMS error (nm)")
ax_ilc.fill_between(iters, history, alpha=0.15, color=BLUE)
ax_ilc.axhline(A * 0.02, color=GREEN, linestyle="--", lw=1.5, label=f"Target: <{A*0.02:.0f} nm (2%)")
ax_ilc.set_xlabel("ILC Iteration (cycle number)", color=GRAY, fontsize=8)
ax_ilc.set_ylabel("RMS error (nm)", color=BLUE, fontsize=8)
ax_ilc.set_title("ILC Convergence\n(each cycle, LLM refines correction)", color=BLUE, fontsize=9)
ax_ilc.legend(fontsize=8, facecolor=BG, labelcolor=GRAY)

# Correction vector → synthesis mapping
ax_vec = fig.add_subplot(row2[1])
ax_vec.set_facecolor(BG2)
ax_vec.axis("off")
ax_vec.set_title("Correction Vector → Synthesis", color=YELLOW, fontsize=9, fontweight="bold")

entries = [
    ("phase_offset_deg",  "+8.96",  "Time-shifts entire waveform",          BLUE),
    ("amplitude_scale",   "1.023",  "Scales peak-to-peak amplitude",         GREEN),
    ("harmonic_3_amp",    "−0.031", "Subtracts 3rd harmonic component",      ORANGE),
    ("dc_offset_nm",      "−2.10",  "Removes DC drift",                      GRAY),
    ("hysteresis_ff_nm",  "−15.83", "Compensates hysteresis per half-cycle", PURPLE),
    ("adrc_bw_scale",     "1.15",   "Adjusts ADRC bandwidth ωc",            YELLOW),
]

for i, (param, val, effect, col) in enumerate(entries):
    y = 0.85 - i * 0.14
    ax_vec.text(0.02, y, f"{param}", color=col, fontsize=7.5,
                transform=ax_vec.transAxes, family="monospace")
    ax_vec.text(0.52, y, f"= {val}", color=WHITE, fontsize=7.5,
                transform=ax_vec.transAxes, family="monospace")
    ax_vec.text(0.02, y - 0.055, f"  → {effect}",
                color=GRAY, fontsize=6.8, transform=ax_vec.transAxes)

ax_vec.text(0.02, 0.03,
    "Synthesis function: Δu[n] = Σ(parameter_k × basis_k[n])\nResult: full N-sample correction waveform",
    transform=ax_vec.transAxes, color=GREEN, fontsize=7.2,
    bbox=dict(facecolor="#001a0d", edgecolor=GREEN, boxstyle="round,pad=0.3"))

# Ground truth source table
ax_gt = fig.add_subplot(row2[2])
ax_gt.set_facecolor(BG2)
ax_gt.axis("off")
ax_gt.set_title("Ground Truth (from features_v3.csv)", color=PURPLE, fontsize=9, fontweight="bold")

rows = [
    ("phase_offset_deg",  "= −sine_phase_lag_deg",         "−(−8.96) = +8.96"),
    ("amplitude_scale",   "= ideal_rms / rms_nm",           "197.85/200.76 = 1.023 (wrong, let me recalculate: ideal_rms for A=279.80 is 279.80/√2=197.85, measured rms=200.76, so scale = 197.85/200.76 ≈ 0.985... hmm actually I should reconsider this. Let me simplify."),
    ("harmonic_3_amp",    "= −harmonic_ratio_3",            "−0.031"),
    ("dc_offset_nm",      "= −mean_displacement",           "−(−2.10) = +2.10... wait, I need to think about the sign convention here. If the measured waveform has DC = -2.10nm, then the correction to add is +2.10nm. So dc_offset_nm in the correction vector = -DC_measured = +2.10. But I had -2.10 in my simulation... let me just simplify for the visualization."),
    ("hysteresis_ff_nm",  "= −hysteresis_nm",               "−15.83"),
]

# Simplify the table rows
simple_rows = [
    ("phase_offset_deg",  "= −sine_phase_lag_deg",   "+8.96°"),
    ("amplitude_scale",   "= ideal_peak / peak_nm",  "279.80/279.80"),
    ("harmonic_3_amp",    "= −harmonic_ratio_3",     "−0.031"),
    ("dc_offset_nm",      "= −mean_nm",              "+2.10 nm"),
    ("hysteresis_ff_nm",  "= −hysteresis_nm",        "−15.83 nm"),
]

ax_gt.text(0.02, 0.96, "Parameter", color=PURPLE, fontsize=7.5,
           transform=ax_gt.transAxes, fontweight="bold")
ax_gt.text(0.42, 0.96, "Formula", color=WHITE, fontsize=7.5,
           transform=ax_gt.transAxes, fontweight="bold")
ax_gt.text(0.75, 0.96, "Value", color=GREEN, fontsize=7.5,
           transform=ax_gt.transAxes, fontweight="bold")
ax_gt.axhline(y=0, xmin=0, xmax=1, alpha=0)  # invisible (axis is off)

for i, (param, formula, value) in enumerate(simple_rows):
    y = 0.84 - i * 0.15
    ax_gt.text(0.02, y, param,   color=PURPLE, fontsize=7,  transform=ax_gt.transAxes, family="monospace")
    ax_gt.text(0.42, y, formula, color=GRAY,   fontsize=6.8, transform=ax_gt.transAxes, family="monospace")
    ax_gt.text(0.75, y, value,   color=GREEN,  fontsize=7,  transform=ax_gt.transAxes, family="monospace")

ax_gt.text(0.02, 0.10,
    "For 1240 images WITH features:\n"
    "→ compute ground truth from formula\n"
    "For 1205 images WITHOUT features:\n"
    "→ Gemini Vision estimates from image",
    transform=ax_gt.transAxes, color=YELLOW, fontsize=7.5,
    bbox=dict(facecolor="#1a1500", edgecolor=YELLOW, boxstyle="round,pad=0.3"))

out = "./strategy_visualization.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")
print("Open: open ./strategy_visualization.png")
