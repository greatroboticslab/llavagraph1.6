"""
make_figures.py — run locally to generate README figures.
Combines a real waveform image (left) with actual LLM output text (right).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import textwrap
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

REAL_IMAGES = {
    "sine":   "../data/ori_plots/sine_Plots/sine_1000Hz_1000Hz_1_absolute_slice_50_200.png",
    "square": "../data/ori_plots/square_Plots/square_1000Hz_1000Hz_1_absolute_slice_50_200.png",
}

# Actual LLM output from eval run (job 53656)
ACTUAL_OUTPUTS = {
    "sine": {
        "input_summary": "COMMANDED WAVEFORM: sine at 20.1 Hz | Target peak: ±335.7 nm\n"
                         "Phase lag: 172.2° | THD: 22.01% | Crest factor: 1.628\n"
                         "2nd harmonic ratio: 0.1624 | 3rd harmonic ratio: 0.0992\n"
                         "Amplitude drift Q1→Q4: 18.87 nm",
        "diagnosis": (
            "The commanded sine at 20.1 Hz displays substantial nonlinearity and distortion. "
            "Despite achieving the desired peak displacement of 335.7 nm, the sine-fit residual "
            "is 99.98%, suggesting the output is far from sinusoidal. The phase lag of 172.2° "
            "greatly surpasses the expected 1.84°, indicating strong nonlinear effects. Resonance "
            "may also play a role given the observed oscillations around the peaks."
        ),
        "correction": (
            "To restore the commanded waveform, implement a combination of feedforward and feedback "
            "compensation strategies. Apply a feedforward correction to mitigate the excessive phase "
            "lag of 172.2° using a lead network tuned to the dominant frequency of 20.10 Hz. Use a "
            "feedback loop to dampen any resonances and ensure stability through proper gain adjustment."
        ),
        "cv": (
            "phase_offset_deg     = -172.2000\n"
            "amplitude_scale      = +1.1512\n"
            "harmonic_2_amp       = -0.1624\n"
            "harmonic_3_amp       = -0.0992\n"
            "dc_offset_nm         = +0.0000\n"
            "hysteresis_ff_nm     = +0.0000"
        ),
    },
    "square": {
        "input_summary": "COMMANDED WAVEFORM: square at 100 Hz | Target peak: ±521.6 nm\n"
                         "THD: 56.93% | Duty cycle: 0.4763 | Edge sharpness: 0.9166\n"
                         "2nd harmonic ratio: 0.0259 | Amplitude drift: 0 nm",
        "diagnosis": (
            "The measured output differs from the commanded square wave primarily because of "
            "nonlinear distortion caused by excessive phase lag. This results in a high total "
            "harmonic distortion (THD) of 56.93%. The crest factor of 1.608 indicates some "
            "overshooting on the peaks. The odd/even harmonic ratio of 150.7 suggests strong "
            "third harmonic content contributing to the distorted waveform."
        ),
        "correction": (
            "To improve the waveform fidelity, reducing the phase lag through adjustments in the "
            "control loop parameters will help minimize the nonlinear distortion. Advancing the "
            "phase angle by approximately 3.0 degrees should bring the observed phase lag closer "
            "to the predicted value. Dampening the oscillatory behavior may require fine-tuning "
            "the gain settings to stabilize the actuator response."
        ),
        "cv": (
            "duty_cycle_trim          = +0.0237\n"
            "edge_sharpness_deficit   = +0.0834\n"
            "harmonic_2_amp           = -0.0259\n"
            "dc_offset_nm             = +0.0000"
        ),
    },
}


def make_figure(wtype, out_path):
    info   = ACTUAL_OUTPUTS[wtype]
    color  = {"sine": "#1565C0", "square": "#2E7D32"}.get(wtype, "#333333")

    fig = plt.figure(figsize=(17, 7))
    fig.patch.set_facecolor("#F5F5F5")
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3],
                            left=0.02, right=0.98, top=0.88, bottom=0.04,
                            wspace=0.04)

    # ── left: real waveform image ─────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0])
    img_path = REAL_IMAGES.get(wtype)
    if img_path and Path(img_path).exists():
        img = imread(img_path)
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, "waveform image", ha="center", va="center",
                    fontsize=14, color="#999999")
    ax_img.axis("off")
    ax_img.set_title("Measured Waveform (real data)", fontsize=11,
                     fontweight="bold", pad=6, color="#222222")

    # input summary box
    ax_img.text(0.02, 0.02, info["input_summary"], transform=ax_img.transAxes,
                fontsize=7.5, verticalalignment="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFFCC",
                          edgecolor="#AAAAAA"))

    # ── right: LLM output ─────────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.set_facecolor("#FFFFFF")
    ax_txt.axis("off")

    y = 0.97
    def put(text, bold=False, color="#111111", size=9.5, indent=0.01, gap=0.038):
        nonlocal y
        ax_txt.text(indent, y, text, transform=ax_txt.transAxes,
                    fontsize=size, color=color,
                    fontweight="bold" if bold else "normal",
                    verticalalignment="top")
        y -= gap

    put("DIAGNOSIS", bold=True, color=color, size=11)
    y -= 0.006
    for line in textwrap.wrap(info["diagnosis"], 72):
        put(line, size=9)
    y -= 0.014

    put("CORRECTION", bold=True, color=color, size=11)
    y -= 0.006
    for line in textwrap.wrap(info["correction"], 72):
        put(line, size=9)
    y -= 0.014

    put("CORRECTION VECTOR  (analytically computed, 100% accurate)",
        bold=True, color="#555555", size=9.5)
    y -= 0.006
    for line in info["cv"].strip().split("\n"):
        put(line.strip(), size=9, color="#333333", indent=0.03,
            gap=0.032)

    fig.suptitle(
        f"Zamba2-7B-instruct  |  Waveform: {wtype.upper()}  |  Piezoelectric Actuator Diagnostic",
        fontsize=12, fontweight="bold", color="#111111", y=0.95
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


for wtype in ["sine", "square"]:
    make_figure(wtype, OUT / f"{wtype}_example.png")

print("Done.")
