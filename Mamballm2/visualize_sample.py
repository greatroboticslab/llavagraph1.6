"""
visualize_sample.py
===================
Diagnostic dashboard for one waveform sample.
Shows the PNG image alongside a visual breakdown of every feature metric
and how it deviates from the mathematical ideal.

Usage:
    python visualize_sample.py
    python visualize_sample.py --image ../vit_mamba/data/train/sine/sine-100Hz-10_absolute_aug_v3.png
    python visualize_sample.py --waveform square --index 0
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


FEATURES_CSV = "../../Desktop/mamballm/features_v3.csv"


# ── Ideal reference values per waveform ───────────────────────────────────────
#   (value, unit, label)
IDEALS = {
    "sine": [
        ("thd_pct",              0.0,   "%",      "THD"),
        ("sine_residual_pct",    0.0,   "%",      "Sine-fit residual"),
        ("crest_factor",         1.414, "",       "Crest factor"),
        ("harmonic_ratio_2",     0.0,   "",       "2nd harmonic / fund."),
        ("harmonic_ratio_3",     0.0,   "",       "3rd harmonic / fund."),
        ("hysteresis_nm",        0.0,   "nm",     "Hysteresis (pos/neg)"),
        ("half_cycle_asymmetry", 0.0,   "",       "Half-cycle asymmetry"),
    ],
    "square": [
        ("thd_pct",              0.0,   "%",      "THD"),
        ("square_duty_cycle",    0.5,   "",       "Duty cycle"),
        ("square_edge_sharpness",1.0,   "",       "Edge sharpness"),
        ("odd_even_harmonic_ratio", 10.0, "",     "Odd/even harmonic ratio"),
        ("harmonic_ratio_3",     0.333, "",       "3rd harmonic / fund."),
        ("crest_factor",         1.0,   "",       "Crest factor"),
    ],
    "noise": [
        ("spectral_flatness",    1.0,   "",       "Spectral flatness"),
        ("kurtosis",             3.0,   "",       "Kurtosis (Gaussian=3)"),
        ("noise_gaussianity_err",0.0,   "",       "Gaussianity error"),
        ("noise_autocorr_lag1",  0.0,   "",       "Autocorr lag-1"),
        ("crest_factor",         3.5,   "",       "Crest factor"),
        ("spectral_entropy",     None,  "",       "Spectral entropy (higher=better)"),
    ],
    "ramp": [
        ("ramp_rise_linearity",  1.0,   "",       "Rise linearity"),
        ("ramp_fall_linearity",  1.0,   "",       "Fall linearity"),
        ("ramp_asymmetry",       0.0,   "",       "Rise/fall asymmetry"),
        ("thd_pct",              0.0,   "%",      "THD"),
        ("harmonic_ratio_2",     0.5,   "",       "2nd harmonic / fund."),
        ("harmonic_ratio_3",     0.333, "",       "3rd harmonic / fund."),
    ],
    "pulse": [
        ("pulse_ringing_ratio",  0.0,   "",       "Post-pulse ringing ratio"),
        ("pulse_duty_cycle",     None,  "",       "Pulse duty cycle"),
        ("pulse_crest_factor",   None,  "",       "Pulse crest factor"),
        ("kurtosis",             None,  "",       "Kurtosis"),
        ("spectral_entropy",     None,  "",       "Spectral entropy"),
    ],
}

QUARTER_KEYS = [
    ("q1_peak_nm", "q1_mean_nm", "Q1\n480-545ms"),
    ("q2_peak_nm", "q2_mean_nm", "Q2\n545-610ms"),
    ("q3_peak_nm", "q3_mean_nm", "Q3\n610-675ms"),
    ("q4_peak_nm", "q4_mean_nm", "Q4\n675-740ms"),
]


# ── Feature matching ──────────────────────────────────────────────────────────

def _clean_stem(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r'_aug_v\d+$', '', s)
    s = re.sub(r'_orig$', '', s)
    # handle double-frequency prefix (e.g. square_100hz_100hz_8 -> square_100hz_8)
    s = re.sub(r'(\d+hz)[_-]\1', r'\1', s)
    # normalize separators
    s = s.replace('_', '-')
    return s


def load_features(features_csv: str) -> dict:
    df = pd.read_csv(features_csv)
    index = {}
    for _, row in df.iterrows():
        key = _clean_stem(Path(row["filename"]).stem)
        index[key] = row.to_dict()
    return index


def find_features(png_path: str, feat_index: dict):
    key = _clean_stem(Path(png_path).stem)
    return feat_index.get(key)


# ── Detect waveform from path ─────────────────────────────────────────────────

def detect_waveform(png_path: str) -> str:
    parts = Path(png_path).parts
    for part in parts:
        if part.lower() in ("sine", "square", "noise", "ramp", "pulse"):
            return part.lower()
    stem = Path(png_path).stem.lower()
    for w in ("sine", "square", "noise", "ramp", "pulse"):
        if w in stem:
            return w
    return "sine"


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_dashboard(png_path: str, feat_row: dict | None, waveform: str, description: str | None = None):
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#0f0f0f")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.38, wspace=0.35,
        left=0.05, right=0.97, top=0.93, bottom=0.06,
    )

    ax_img    = fig.add_subplot(gs[0, 0])
    ax_dev    = fig.add_subplot(gs[0, 1])
    ax_time   = fig.add_subplot(gs[0, 2])
    ax_harm   = fig.add_subplot(gs[1, 0])
    ax_fopdt  = fig.add_subplot(gs[1, 1])
    ax_desc   = fig.add_subplot(gs[1, 2])

    for ax in [ax_img, ax_dev, ax_time, ax_harm, ax_fopdt, ax_desc]:
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ACCENT  = "#00d4ff"
    RED     = "#ff4757"
    GREEN   = "#2ed573"
    YELLOW  = "#ffa502"
    GRAY    = "#aaaaaa"

    # ── Panel 1: Waveform image ──────────────────────────────────────────────
    img = mpimg.imread(png_path)
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"Measured Waveform\n{Path(png_path).name}", color=ACCENT, fontsize=9)

    has_feat = feat_row is not None

    # ── Panel 2: Key metrics vs ideal ────────────────────────────────────────
    ideal_specs = IDEALS.get(waveform, [])
    labels, actuals, ideals_v, colors = [], [], [], []
    for (key, ideal, unit, label) in ideal_specs:
        if ideal is None:
            continue
        val = feat_row.get(key) if has_feat else None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        labels.append(f"{label}\n(ideal={ideal}{unit})")
        actuals.append(float(val))
        ideals_v.append(float(ideal))
        pct_err = abs(float(val) - float(ideal)) / (abs(float(ideal)) + 1e-9)
        colors.append(RED if pct_err > 0.20 else YELLOW if pct_err > 0.05 else GREEN)

    if labels:
        x = np.arange(len(labels))
        w = 0.35
        ax_dev.bar(x - w/2, actuals, w, label="Measured", color=ACCENT, alpha=0.85)
        ax_dev.bar(x + w/2, ideals_v, w, label="Ideal",    color=GREEN,  alpha=0.60)
        for i, (a, c) in enumerate(zip(actuals, colors)):
            ax_dev.plot(x[i] - w/2, a, "o", color=c, markersize=8, zorder=5)
        ax_dev.set_xticks(x)
        ax_dev.set_xticklabels(labels, fontsize=6.5, color=GRAY, rotation=15, ha="right")
        ax_dev.tick_params(axis="y", colors=GRAY)
        ax_dev.legend(fontsize=7, facecolor="#222", labelcolor=GRAY)
    else:
        ax_dev.text(0.5, 0.5, "No feature data\navailable", transform=ax_dev.transAxes,
                    ha="center", va="center", color=GRAY, fontsize=11)
    ax_dev.set_title(f"Key Metrics vs Ideal  ({waveform.upper()})", color=ACCENT, fontsize=9)
    ax_dev.set_ylabel("Value", color=GRAY, fontsize=8)

    # ── Panel 3: Temporal Q1-Q4 ─────────────────────────────────────────────
    q_labels = [q[2] for q in QUARTER_KEYS]
    q_peaks  = [feat_row.get(q[0]) for q in QUARTER_KEYS] if has_feat else [None]*4
    q_means  = [feat_row.get(q[1]) for q in QUARTER_KEYS] if has_feat else [None]*4
    if any(v is not None for v in q_peaks):
        valid_peaks = [float(v) for v in q_peaks if v is not None and not np.isnan(float(v))]
        avg_peak = np.mean(valid_peaks) if valid_peaks else 0
        ax_time.bar(q_labels, [float(v) if v is not None else 0 for v in q_peaks],
                    color=ACCENT, alpha=0.80, label="Peak (nm)")
        ax_time.axhline(avg_peak, color=YELLOW, linestyle="--", linewidth=1.5, label=f"Avg={avg_peak:.1f} nm")
        ax_time.set_ylabel("Displacement (nm)", color=GRAY, fontsize=8)
        ax_time.tick_params(colors=GRAY)
        ax_time.legend(fontsize=7, facecolor="#222", labelcolor=GRAY)
        drift = feat_row.get("amplitude_drift_nm", None) if has_feat else None
        drift_str = f"  Drift Q1→Q4: {float(drift):.2f} nm" if drift is not None else ""
        ax_time.set_title(f"Temporal Deviation (Q1–Q4){drift_str}", color=ACCENT, fontsize=9)
    else:
        ax_time.text(0.5, 0.5, "No temporal data", transform=ax_time.transAxes,
                     ha="center", va="center", color=GRAY, fontsize=11)
        ax_time.set_title("Temporal Deviation (Q1–Q4)", color=ACCENT, fontsize=9)

    # ── Panel 4: Harmonic content ────────────────────────────────────────────
    harm_keys   = ["harmonic_ratio_2", "harmonic_ratio_3", "harmonic_ratio_4", "harmonic_ratio_5"]
    harm_labels = ["2nd", "3rd", "4th", "5th"]
    harm_ideal  = {"sine":   [0, 0, 0, 0],
                   "square": [0, 0.333, 0, 0.200],
                   "ramp":   [0.500, 0.333, 0.250, 0.200]}.get(waveform, [None]*4)
    harm_vals   = [float(feat_row.get(k, 0) or 0) for k in harm_keys] if has_feat else [0]*4

    x = np.arange(4)
    ax_harm.bar(x, harm_vals, color=ACCENT, alpha=0.85, label="Measured")
    if any(v is not None for v in harm_ideal):
        ax_harm.bar(x, [v if v is not None else 0 for v in harm_ideal],
                    alpha=0.4, color=GREEN, label="Ideal")
    ax_harm.set_xticks(x)
    ax_harm.set_xticklabels(harm_labels, color=GRAY)
    ax_harm.tick_params(axis="y", colors=GRAY)
    ax_harm.set_ylabel("Ratio to fundamental", color=GRAY, fontsize=8)
    ax_harm.set_title("Harmonic Content", color=ACCENT, fontsize=9)
    ax_harm.legend(fontsize=7, facecolor="#222", labelcolor=GRAY)

    # ── Panel 5: FOPDT phase/attenuation ────────────────────────────────────
    freqs   = np.logspace(1, 4, 300)
    tau, theta, K = 250e-6, 5e-6, 380.0
    phase_pred = -(np.arctan(2 * np.pi * freqs * tau) + 2 * np.pi * freqs * theta) * 180 / np.pi
    atten_pred = 1.0 / np.sqrt(1 + (2 * np.pi * freqs * tau) ** 2)

    color2 = "#ff6b81"
    ax_f2 = ax_fopdt.twinx()
    ax_fopdt.semilogx(freqs, phase_pred, color=ACCENT,  linewidth=1.8, label="Phase lag (deg)")
    ax_f2.semilogx(freqs, atten_pred * 100, color=color2, linewidth=1.8, linestyle="--", label="Attenuation (%)")

    if has_feat:
        freq = feat_row.get("dominant_freq_hz")
        meas_phase = feat_row.get("sine_phase_lag_deg")
        pred_phase = feat_row.get("fopdt_phase_lag_deg")
        pred_atten = feat_row.get("fopdt_attenuation")
        if freq and not np.isnan(float(freq)):
            f = float(freq)
            ax_fopdt.axvline(f, color=YELLOW, linestyle=":", linewidth=1.2, label=f"f={f:.0f} Hz")
        if meas_phase and not np.isnan(float(meas_phase)):
            ax_fopdt.axhline(float(meas_phase), color=GREEN,  linestyle="--", linewidth=1.0,
                             label=f"Measured phase={float(meas_phase):.1f}°")
        if pred_phase and not np.isnan(float(pred_phase)):
            ax_fopdt.axhline(float(pred_phase), color=RED,    linestyle="--", linewidth=1.0,
                             label=f"FOPDT pred={float(pred_phase):.1f}°")

    ax_fopdt.set_xlabel("Frequency (Hz)", color=GRAY, fontsize=8)
    ax_fopdt.set_ylabel("Phase lag (deg)", color=ACCENT, fontsize=8)
    ax_f2.set_ylabel("Attenuation (%)", color=color2, fontsize=8)
    ax_fopdt.tick_params(colors=GRAY)
    ax_f2.tick_params(colors=color2)
    ax_fopdt.set_title("FOPDT Actuator Model  (K=380 nm/V, τ=250μs, θ=5μs)", color=ACCENT, fontsize=9)
    lines1, labs1 = ax_fopdt.get_legend_handles_labels()
    lines2, labs2 = ax_f2.get_legend_handles_labels()
    ax_fopdt.legend(lines1 + lines2, labs1 + labs2, fontsize=6.5, facecolor="#222", labelcolor=GRAY)

    # ── Panel 6: Description text ────────────────────────────────────────────
    ax_desc.axis("off")
    if description:
        wrapped = description[:900] + ("..." if len(description) > 900 else "")
    else:
        wrapped = ("No description found in training_data.csv.\n"
                   "Run generate_training_data.py first.")
    ax_desc.text(0.03, 0.97, "Gemini Description", transform=ax_desc.transAxes,
                 color=ACCENT, fontsize=9, fontweight="bold", va="top")
    ax_desc.text(0.03, 0.88, wrapped, transform=ax_desc.transAxes,
                 color=GRAY, fontsize=6.5, va="top", wrap=True,
                 bbox=dict(facecolor="#111", edgecolor="#333", boxstyle="round,pad=0.4"))

    # ── Legend box: what each color means ───────────────────────────────────
    legend_txt = (
        "  Metric deviation guide:\n"
        "  ● Green  — within 5% of ideal\n"
        "  ● Yellow — 5–20% deviation\n"
        "  ● Red    — >20% deviation\n\n"
        "  FOPDT = First-Order Plus Dead-Time\n"
        "  actuator model (physics prior)\n\n"
        "  Ideal = mathematical reference\n"
        "  (perfect sine/square/ramp/etc.)"
    )
    fig.text(0.003, 0.01, legend_txt, color=GRAY, fontsize=7,
             va="bottom", ha="left",
             bbox=dict(facecolor="#111", edgecolor="#333", boxstyle="round,pad=0.5"))

    stem = Path(png_path).stem
    fig.suptitle(f"Diagnostic Dashboard — {stem}  |  waveform: {waveform.upper()}",
                 color="white", fontsize=12, fontweight="bold", y=0.97)

    out_path = f"./diagnostic_{stem[:50]}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    default=None, help="Path to a specific PNG")
    parser.add_argument("--waveform", default="sine",
                        choices=["sine", "square", "noise", "ramp", "pulse"])
    parser.add_argument("--index",    type=int, default=0,
                        help="Which image of that waveform type to use (0-based)")
    parser.add_argument("--features_csv", default=FEATURES_CSV)
    parser.add_argument("--training_csv", default="./training_data.csv")
    args = parser.parse_args()

    feat_index = load_features(args.features_csv)

    if args.image:
        png_path = args.image
        waveform = detect_waveform(png_path)
    else:
        data_root = Path("../vit_mamba/data")
        candidates = sorted((data_root).rglob(f"*{args.waveform}*/*.png"))
        if not candidates:
            candidates = []
            for split in ["train", "val", "test"]:
                candidates += sorted((data_root / split / args.waveform).glob("*.png"))
        if not candidates:
            raise SystemExit(f"No PNG files found for waveform={args.waveform}")
        idx = min(args.index, len(candidates) - 1)
        png_path = str(candidates[idx])
        waveform = args.waveform

    feat_row = find_features(png_path, feat_index)
    print(f"Image   : {png_path}")
    print(f"Waveform: {waveform}")
    print(f"Features: {'FOUND' if feat_row else 'NOT FOUND (nofeat)'}")

    # Try to load matching description from training_data.csv
    description = None
    try:
        td = pd.read_csv(args.training_csv)
        match = td[td["image_path"].str.endswith(Path(png_path).name)]
        if not match.empty:
            description = match.iloc[0]["description"]
            print(f"Description: found ({len(description)} chars)")
        else:
            print("Description: not found in training_data.csv")
    except Exception:
        pass

    out_path = plot_dashboard(png_path, feat_row, waveform, description)
    print(f"\nOpen the dashboard: open {out_path}")


if __name__ == "__main__":
    main()
