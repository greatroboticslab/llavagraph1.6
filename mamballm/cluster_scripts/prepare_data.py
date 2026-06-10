"""
prepare_data.py
===============
Converts training_data.csv → train.jsonl / val.jsonl / test.jsonl
in instruction-following format for Nemotron-H fine-tuning.

Run locally before uploading to cluster:
    python prepare_data.py --input training_data.csv --output_dir ./data
"""

import argparse
import json
import pandas as pd
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert in piezoelectric actuator closed-loop control and vibration analysis. "
    "Given a waveform type and its measured signal features, diagnose how the waveform "
    "deviates from its ideal reference and prescribe quantitative corrections. "
    "Structure your response as: "
    "(1) DEVIATION SUMMARY — actual vs ideal value with signed error for each metric; "
    "(2) TEMPORAL DEVIATION — which quarter (Q1-Q4, window 480-740 ms) is worst and by how much; "
    "(3) VIBRATION ANALYSIS — dominant distortion mechanism, resonant frequency if present, "
    "whether errors are steady or drifting; "
    "(4) CORRECTION VECTOR — specific magnitudes for phase compensation (°), amplitude "
    "rescaling (%), harmonic suppression, and any per-quarter feedforward adjustment."
)

# Human-readable labels for feature keys
FEATURE_LABELS = {
    # FFT features
    "dominant_freq_hz":        "Dominant frequency (Hz)",
    "dominant_amp_nm":         "Dominant amplitude (nm)",
    "thd_pct":                 "Total Harmonic Distortion (%)",
    "spectral_entropy":        "Spectral entropy",
    "spectral_flatness":       "Spectral flatness",
    "spectral_centroid_hz":    "Spectral centroid (Hz)",
    "band_energy_ratio":       "Band energy ratio (low/high)",
    "num_spectral_peaks":      "Number of spectral peaks",
    "harmonic_ratio_2":        "2nd harmonic ratio",
    "harmonic_ratio_3":        "3rd harmonic ratio",
    "harmonic_ratio_4":        "4th harmonic ratio",
    "harmonic_ratio_5":        "5th harmonic ratio",
    "odd_even_harmonic_ratio": "Odd/even harmonic ratio",
    # Time-domain features
    "rms_nm":                  "RMS displacement (nm)",
    "peak_nm":                 "Peak displacement (nm)",
    "crest_factor":            "Crest factor",
    "skewness":                "Skewness",
    "kurtosis":                "Kurtosis",
    "zero_cross_rate":         "Zero-crossing rate (Hz)",
    "peak_density_hz":         "Peak density (Hz)",
    # Waveform-specific features
    "sine_residual_pct":       "Sine-fit residual (%)",
    "sine_phase_lag_deg":      "Phase lag (°)",
    "square_duty_cycle":       "Duty cycle",
    "square_edge_sharpness":   "Edge sharpness",
    "noise_kurtosis":          "Noise kurtosis",
    "noise_gaussianity_err":   "Gaussianity error",
    "noise_autocorr_lag1":     "Autocorrelation lag-1",
    "ramp_rise_linearity":     "Rise linearity",
    "ramp_fall_linearity":     "Fall linearity",
    "ramp_asymmetry":          "Rise/fall asymmetry",
    "pulse_crest_factor":      "Pulse crest factor",
    "pulse_duty_cycle":        "Pulse duty cycle",
    "pulse_ringing_ratio":     "Post-pulse ringing ratio",
    # FOPDT model (meter repo, same device — K=380 nm/V, τ=250 µs, θ=5 µs)
    "fopdt_phase_lag_deg":     "FOPDT predicted phase lag (°)",
    "fopdt_attenuation":       "FOPDT amplitude attenuation factor",
    "v_drive_est_v":           "Estimated drive voltage (V)",
    # Hysteresis proxy (sine only)
    "hysteresis_nm":           "Hysteresis pos/neg asymmetry (nm)",
    "half_cycle_asymmetry":    "Half-cycle asymmetry fraction",
    # Per-quarter features (480–740 ms window, Q1–Q4 each 65 ms)
    "q1_peak_nm":              "Q1 (480-545 ms) peak displacement (nm)",
    "q1_mean_nm":              "Q1 (480-545 ms) mean displacement (nm)",
    "q1_rms_nm":               "Q1 (480-545 ms) RMS displacement (nm)",
    "q1_sine_residual_pct":    "Q1 (480-545 ms) sine-fit residual (%)",
    "q2_peak_nm":              "Q2 (545-610 ms) peak displacement (nm)",
    "q2_mean_nm":              "Q2 (545-610 ms) mean displacement (nm)",
    "q2_rms_nm":               "Q2 (545-610 ms) RMS displacement (nm)",
    "q2_sine_residual_pct":    "Q2 (545-610 ms) sine-fit residual (%)",
    "q3_peak_nm":              "Q3 (610-675 ms) peak displacement (nm)",
    "q3_mean_nm":              "Q3 (610-675 ms) mean displacement (nm)",
    "q3_rms_nm":               "Q3 (610-675 ms) RMS displacement (nm)",
    "q3_sine_residual_pct":    "Q3 (610-675 ms) sine-fit residual (%)",
    "q4_peak_nm":              "Q4 (675-740 ms) peak displacement (nm)",
    "q4_mean_nm":              "Q4 (675-740 ms) mean displacement (nm)",
    "q4_rms_nm":               "Q4 (675-740 ms) RMS displacement (nm)",
    "q4_sine_residual_pct":    "Q4 (675-740 ms) sine-fit residual (%)",
    "amplitude_drift_nm":      "Peak amplitude drift Q1→Q4 (nm)",
    "worst_quarter":           "Quarter with highest peak amplitude",
}


# Features that are only meaningful for specific waveforms.
# Excluding these per waveform prevents the model from inheriting wrong
# ideal values from other waveform templates (e.g. "odd/even ratio vs ideal ∞"
# leaking into sine outputs from square wave training examples).
_EXCLUDE_BY_WAVEFORM = {
    "sine":   {"odd_even_harmonic_ratio", "square_duty_cycle", "square_edge_sharpness",
               "noise_kurtosis", "noise_gaussianity_err", "noise_autocorr_lag1",
               "ramp_rise_linearity", "ramp_fall_linearity", "ramp_asymmetry",
               "pulse_crest_factor", "pulse_duty_cycle", "pulse_ringing_ratio"},
    "square": {"sine_residual_pct", "sine_phase_lag_deg",
               "noise_kurtosis", "noise_gaussianity_err", "noise_autocorr_lag1",
               "ramp_rise_linearity", "ramp_fall_linearity", "ramp_asymmetry",
               "pulse_crest_factor", "pulse_duty_cycle", "pulse_ringing_ratio",
               "hysteresis_nm", "half_cycle_asymmetry",
               "q1_sine_residual_pct", "q2_sine_residual_pct",
               "q3_sine_residual_pct", "q4_sine_residual_pct"},
    "noise":  {"sine_residual_pct", "sine_phase_lag_deg",
               "odd_even_harmonic_ratio", "square_duty_cycle", "square_edge_sharpness",
               "ramp_rise_linearity", "ramp_fall_linearity", "ramp_asymmetry",
               "pulse_crest_factor", "pulse_duty_cycle", "pulse_ringing_ratio",
               "hysteresis_nm", "half_cycle_asymmetry",
               "q1_sine_residual_pct", "q2_sine_residual_pct",
               "q3_sine_residual_pct", "q4_sine_residual_pct"},
    "ramp":   {"sine_residual_pct", "sine_phase_lag_deg",
               "odd_even_harmonic_ratio", "square_duty_cycle", "square_edge_sharpness",
               "noise_kurtosis", "noise_gaussianity_err", "noise_autocorr_lag1",
               "pulse_crest_factor", "pulse_duty_cycle", "pulse_ringing_ratio",
               "hysteresis_nm", "half_cycle_asymmetry",
               "q1_sine_residual_pct", "q2_sine_residual_pct",
               "q3_sine_residual_pct", "q4_sine_residual_pct"},
    "pulse":  {"sine_residual_pct", "sine_phase_lag_deg",
               "odd_even_harmonic_ratio", "square_duty_cycle", "square_edge_sharpness",
               "noise_kurtosis", "noise_gaussianity_err", "noise_autocorr_lag1",
               "ramp_rise_linearity", "ramp_fall_linearity", "ramp_asymmetry",
               "hysteresis_nm", "half_cycle_asymmetry",
               "q1_sine_residual_pct", "q2_sine_residual_pct",
               "q3_sine_residual_pct", "q4_sine_residual_pct"},
}


def format_features(features: dict, waveform: str) -> str:
    """Format features dict into a clean readable string, filtered per waveform type."""
    excluded = _EXCLUDE_BY_WAVEFORM.get(waveform, set())
    lines = []
    for key, label in FEATURE_LABELS.items():
        if key in excluded:
            continue
        val = features.get(key)
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            continue
        try:
            lines.append(f"  {label}: {float(val):.4g}")
        except (TypeError, ValueError):
            lines.append(f"  {label}: {val}")
    return "\n".join(lines)


def build_prompt(waveform: str, features: dict) -> str:
    feat_str = format_features(features, waveform)
    return (
        f"Waveform type: {waveform}\n"
        f"Measurement features:\n{feat_str}"
    )


def row_to_example(row) -> dict:
    features = json.loads(row["features_json"]) if row["features_json"] else {}
    prompt = build_prompt(row["waveform_label"], features)
    return {
        "system":   SYSTEM_PROMPT,
        "input":    prompt,
        "output":   row["description"].strip(),
        "waveform": row["waveform_label"],
        "split":    row["split"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="training_data_v2.csv")
    parser.add_argument("--output_dir", default="./data")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        out_file = out / f"{split}.jsonl"
        with open(out_file, "w") as f:
            for _, row in subset.iterrows():
                example = row_to_example(row)
                f.write(json.dumps(example) + "\n")
        print(f"{split}: {len(subset)} examples → {out_file}")

    # Print one example so you can verify
    sample = row_to_example(df.iloc[0])
    print("\n--- Sample training example ---")
    print(f"[system]  {sample['system'][:80]}...")
    print(f"[input]\n{sample['input']}")
    print(f"[output]  {sample['output'][:120]}...")


if __name__ == "__main__":
    main()
