"""
Batch Feature Extraction for Piezoelectric Actuator Displacement Data
======================================================================
Usage:
    python batch_feature_extraction.py --data_dir /path/to/piezo_data

Expected folder structure:
    data_dir/
        sine/      *.csv
        square/    *.csv
        noise/     *.csv
        ramp/      *.csv
        pulse/     *.csv

Each CSV must have:
    - A time column (e.g. Time_ms)
    - A displacement column (e.g. Absolute_Displacement_nm)
    No voltage column required.

Output:
    features.csv   -- one row per file, all extracted features + label
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import windows, find_peaks, welch
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────
WAVEFORM_FOLDERS = ["sine", "square", "noise", "ramp", "pulse"]
FS = 1000.0   # Sampling rate in Hz (Time_ms step = 1 ms)

# Image window — every PNG shows exactly this range
WINDOW_START_MS = 480
WINDOW_END_MS   = 740   # 260 ms total

# Quarter boundaries (ms) within the window
Q_BOUNDS_MS = [480, 545, 610, 675, 740]

# FOPDT model parameters calibrated from meter repo (same device)
# K from config.py default; τ estimated from observed ~9° phase lag at 100 Hz
# θ from simulate_config.json
FOPDT_K_NM_PER_V = 380.0     # nm/V  (midpoint of 350–410 from meter repo)
FOPDT_TAU_S      = 0.000250  # 250 µs (derived: arctan(2π·100·τ) ≈ 9° → τ≈250 µs)
FOPDT_THETA_S    = 0.000005  # 5 µs   (from meter simulate_config.json)


# ──────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────
def load_signal(filepath):
    """
    Load CSV and return (full_signal, window_signal) both zero-mean.
    full_signal  : entire recording (for FFT / spectral features)
    window_signal: 480–740 ms slice (matches what the PNG images show)
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Detect displacement column
    disp_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["disp", "displacement", "position", "nm", "um"]):
            disp_col = c
            break
    if disp_col is None:
        raise ValueError(f"Cannot find displacement column. Columns: {list(df.columns)}")

    # Detect time column
    time_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["time", "t_ms", "time_ms"]):
            time_col = c
            break

    full = df[disp_col].dropna().values.astype(float)
    full = full - np.mean(full)

    # Extract the image window (480–740 ms)
    if time_col is not None:
        mask = (df[time_col] >= WINDOW_START_MS) & (df[time_col] < WINDOW_END_MS)
        win_raw = df.loc[mask, disp_col].dropna().values.astype(float)
    else:
        # Fallback: use index arithmetic (1 kHz → 1 sample/ms)
        win_raw = full[WINDOW_START_MS:WINDOW_END_MS]

    window = win_raw - np.mean(win_raw)
    return full, window


# ──────────────────────────────────────────────────────
# UNIVERSAL FFT FEATURES (all waveforms)
# ──────────────────────────────────────────────────────
def fft_features(signal, fs=FS):
    """
    Compute FFT-based features that work for all waveform types.

    For NOISE: dominant_freq will be unreliable — rely on spectral_entropy and band_energy_ratio.
    For SINE:  low THD, low harmonic ratios, sharp dominant peak.
    For SQUARE: high odd harmonics, high THD.
    For RAMP:  all harmonics present with 1/n amplitude decay.
    For PULSE: broad spectrum, low spectral entropy relative to noise.
    """
    N = len(signal)
    if N < 32:
        return None

    win = windows.hann(N)
    fft_vals = np.fft.rfft(signal * win)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    amps = (2.0 / win.sum()) * np.abs(fft_vals)

    # --- Dominant frequency (skip DC)
    min_idx = max(1, int(2.0 * N / fs))  # ignore below 2 Hz
    search = amps.copy()
    search[:min_idx] = 0
    dom_idx = int(np.argmax(search))
    dom_freq = float(freqs[dom_idx])
    dom_amp = float(amps[dom_idx])

    bin_res = float(freqs[1]) if len(freqs) > 1 else 1.0

    def harmonic_amp(n):
        target = n * dom_freq
        if target >= fs / 2 or dom_freq < 1.0:
            return 0.0
        lo = max(0, int((target - 3 * bin_res) / bin_res))
        hi = min(len(freqs) - 1, int((target + 3 * bin_res) / bin_res))
        return float(np.max(amps[lo:hi+1])) if lo < hi else 0.0

    h2 = harmonic_amp(2)
    h3 = harmonic_amp(3)
    h4 = harmonic_amp(4)
    h5 = harmonic_amp(5)

    # THD — high for square/ramp, low for sine, meaningless for noise
    thd = np.sqrt((h2**2 + h3**2 + h4**2 + h5**2) / max(dom_amp**2, 1e-30)) * 100.0

    # Odd/even harmonic ratio — key discriminator for square vs ramp
    # Square: strong odd (h3, h5), weak even (h2, h4)
    # Ramp:   both odd and even present
    odd_energy = h3**2 + h5**2
    even_energy = h2**2 + h4**2
    odd_even_ratio = odd_energy / max(even_energy, 1e-30)

    # Spectral entropy — HIGH for noise, LOW for sine/pulse
    psd = amps[min_idx:]**2
    psd_norm = psd / (psd.sum() + 1e-30)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-30)))

    # Spectral centroid
    spectral_centroid = float(
        np.sum(freqs[min_idx:] * amps[min_idx:]**2) /
        max(np.sum(amps[min_idx:]**2), 1e-30)
    )

    # Band energy ratio: energy in [1-100 Hz] vs [100-500 Hz]
    # Noise: more spread; Sine: concentrated at low freq
    low_mask  = (freqs >= 1)   & (freqs < 100)
    high_mask = (freqs >= 100) & (freqs < 500)
    low_energy  = float(np.sum(amps[low_mask]**2))
    high_energy = float(np.sum(amps[high_mask]**2))
    band_energy_ratio = low_energy / max(high_energy, 1e-30)

    # Spectral flatness (Wiener entropy) — close to 1 for noise, near 0 for tonal signals
    geom_mean = np.exp(np.mean(np.log(amps[min_idx:] + 1e-30)))
    arith_mean = np.mean(amps[min_idx:]) + 1e-30
    spectral_flatness = float(geom_mean / arith_mean)

    # Number of significant peaks in spectrum
    peak_threshold = 0.05 * dom_amp
    spectral_peaks, _ = find_peaks(amps[min_idx:], height=peak_threshold, distance=5)
    num_spectral_peaks = len(spectral_peaks)

    return {
        "dominant_freq_hz":     round(dom_freq, 4),
        "dominant_amp_nm":      round(dom_amp, 6),
        "harmonic_ratio_2":     round(h2 / max(dom_amp, 1e-30), 6),
        "harmonic_ratio_3":     round(h3 / max(dom_amp, 1e-30), 6),
        "harmonic_ratio_4":     round(h4 / max(dom_amp, 1e-30), 6),
        "harmonic_ratio_5":     round(h5 / max(dom_amp, 1e-30), 6),
        "odd_even_harmonic_ratio": round(float(odd_even_ratio), 6),
        "thd_pct":              round(float(thd), 4),
        "spectral_entropy":     round(float(spectral_entropy), 4),
        "spectral_centroid_hz": round(spectral_centroid, 4),
        "spectral_flatness":    round(float(spectral_flatness), 6),
        "band_energy_ratio":    round(float(band_energy_ratio), 4),
        "num_spectral_peaks":   int(num_spectral_peaks),
    }


# ──────────────────────────────────────────────────────
# WAVEFORM-SPECIFIC FEATURES
# ──────────────────────────────────────────────────────

def sine_specific_features(signal, dom_freq, fs=FS):
    """
    Sine: measure how much the actual waveform deviates from a pure sine.
    Key insight: hysteresis causes phase lag and amplitude distortion.
    """
    N = len(signal)
    t = np.arange(N) / fs

    if dom_freq < 0.5:
        return {"sine_residual_pct": np.nan, "sine_phase_lag_deg": np.nan}

    # Fit a pure sine at dominant frequency
    A = np.column_stack([
        np.sin(2 * np.pi * dom_freq * t),
        np.cos(2 * np.pi * dom_freq * t),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    fitted = A @ coeffs
    residual = signal - fitted
    residual_pct = float(np.sqrt(np.mean(residual**2)) / max(np.std(signal), 1e-30) * 100)

    # Phase lag from fitted coefficients
    phase_lag_deg = float(np.degrees(np.arctan2(coeffs[1], coeffs[0])))

    return {
        "sine_residual_pct":   round(residual_pct, 4),
        "sine_phase_lag_deg":  round(phase_lag_deg, 4),
    }


def square_specific_features(signal, fs=FS):
    """
    Square: measure rise/fall time and how 'square' the waveform actually is.
    At high freq, hysteresis rounds the edges — the waveform becomes sine-like.
    """
    if np.std(signal) < 1e-30:
        return {"square_duty_cycle": np.nan, "square_edge_sharpness": np.nan}

    # Duty cycle: fraction of time signal is above zero
    duty_cycle = float(np.mean(signal > 0))

    # Edge sharpness: variance of first derivative (high = sharp edges)
    diff_signal = np.diff(signal)
    edge_sharpness = float(np.var(diff_signal) / max(np.var(signal), 1e-30))

    return {
        "square_duty_cycle":      round(duty_cycle, 4),
        "square_edge_sharpness":  round(edge_sharpness, 6),
    }


def noise_specific_features(signal, fs=FS):
    """
    Noise: characterize the randomness and frequency distribution.
    True broadband noise has high entropy, near-Gaussian distribution.
    """
    # Kolmogorov-Smirnov-like Gaussianity: kurtosis of Gaussian = 3
    std = np.std(signal)
    if std < 1e-30:
        return {"noise_kurtosis": np.nan, "noise_gaussianity_err": np.nan}

    normalized = (signal - np.mean(signal)) / std
    kurtosis = float(np.mean(normalized**4))
    # Gaussianity error: how far kurtosis is from 3.0
    gaussianity_err = abs(kurtosis - 3.0)

    # Autocorrelation at lag 1 (should be ~0 for white noise)
    ac = float(np.corrcoef(signal[:-1], signal[1:])[0, 1])

    return {
        "noise_kurtosis":        round(kurtosis, 4),
        "noise_gaussianity_err": round(gaussianity_err, 4),
        "noise_autocorr_lag1":   round(ac, 6),
    }


def ramp_specific_features(signal, fs=FS):
    """
    Ramp (sawtooth): measure linearity of the rising/falling segments.
    Hysteresis causes the ramp to be nonlinear — S-shaped instead of straight.
    """
    # Split into ascending and descending segments by sign of derivative
    diff = np.diff(signal)
    ascending = diff[diff > 0]
    descending = diff[diff < 0]

    if len(ascending) < 5 or len(descending) < 5:
        return {"ramp_rise_linearity": np.nan, "ramp_fall_linearity": np.nan, "ramp_asymmetry": np.nan}

    # Linearity of rising segments: std of derivative should be low if linear
    rise_linearity = float(1.0 - np.std(ascending) / max(np.mean(np.abs(ascending)), 1e-30))
    fall_linearity = float(1.0 - np.std(np.abs(descending)) / max(np.mean(np.abs(descending)), 1e-30))

    # Asymmetry: are rise and fall slopes similar?
    ramp_asymmetry = float(abs(np.mean(ascending) - np.mean(np.abs(descending))) /
                           max(np.mean(np.abs(ascending)) + np.mean(np.abs(descending)), 1e-30))

    return {
        "ramp_rise_linearity": round(rise_linearity, 4),
        "ramp_fall_linearity": round(fall_linearity, 4),
        "ramp_asymmetry":      round(ramp_asymmetry, 4),
    }


def pulse_specific_features(signal, fs=FS):
    """
    Pulse: measure pulse sharpness and duty cycle.
    High crest factor and low duty cycle are key identifiers.
    Hysteresis causes pulse tails and pre-ringing.
    """
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    crest_factor = peak / max(rms, 1e-30)

    # Pulse duty cycle: fraction of time signal exceeds 50% of peak
    threshold = 0.5 * peak
    duty_cycle = float(np.mean(np.abs(signal) > threshold))

    # Pulse ringing: energy in the tail after main pulse
    # Simple proxy: ratio of tail RMS to peak RMS
    peak_idx = int(np.argmax(np.abs(signal)))
    tail_start = min(peak_idx + int(0.01 * fs), len(signal) - 1)  # 10ms after peak
    if tail_start < len(signal) - 10:
        tail_rms = float(np.sqrt(np.mean(signal[tail_start:]**2)))
        ringing_ratio = tail_rms / max(rms, 1e-30)
    else:
        ringing_ratio = float("nan")

    return {
        "pulse_crest_factor":  round(float(crest_factor), 4),
        "pulse_duty_cycle":    round(duty_cycle, 4),
        "pulse_ringing_ratio": round(float(ringing_ratio), 4) if not np.isnan(ringing_ratio) else float("nan"),
    }


# ──────────────────────────────────────────────────────
# UNIVERSAL TIME-DOMAIN FEATURES (all waveforms)
# ──────────────────────────────────────────────────────
def time_domain_features(signal):
    """
    Basic time-domain statistics — valid for all waveform types.
    """
    std = float(np.std(signal))
    if std < 1e-30:
        return {k: float("nan") for k in
                ["rms_nm", "peak_nm", "crest_factor", "skewness", "kurtosis",
                 "zero_cross_rate", "peak_density_hz"]}

    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    normalized = (signal - np.mean(signal)) / std

    peaks, _ = find_peaks(signal, height=0.1 * peak, distance=int(FS * 0.005))
    peak_density = float(len(peaks) / len(signal) * FS)

    zc = float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal) * FS)

    return {
        "rms_nm":           round(rms, 6),
        "peak_nm":          round(peak, 6),
        "crest_factor":     round(peak / max(rms, 1e-30), 4),
        "skewness":         round(float(np.mean(normalized**3)), 4),
        "kurtosis":         round(float(np.mean(normalized**4)), 4),
        "zero_cross_rate":  round(zc, 4),
        "peak_density_hz":  round(peak_density, 4),
    }


# ──────────────────────────────────────────────────────
# PER-QUARTER FEATURES  (480–740 ms window, split into Q1–Q4)
# ──────────────────────────────────────────────────────
def quarter_features(window, waveform, dom_freq, fs=FS):
    """
    Compute amplitude, RMS, and mean for each 65 ms quarter of the image window.
    For sine waveforms also compute per-quarter sine-fit residual.
    Quarter sizes in samples at 1 kHz:
      Q1: 0–64   (480–544 ms)
      Q2: 65–129 (545–609 ms)
      Q3: 130–194 (610–674 ms)
      Q4: 195–259 (675–739 ms)
    """
    n = len(window)
    if n < 4:
        return {}

    # Quarter boundaries relative to window start (65 ms each)
    boundaries = [0, 65, 130, 195, n]
    feat = {}
    q_peaks = []

    for i, qname in enumerate(["q1", "q2", "q3", "q4"]):
        s, e = boundaries[i], boundaries[i + 1]
        q = window[s:e]
        if len(q) == 0:
            continue
        q_peak = float(np.max(np.abs(q)))
        q_peaks.append(q_peak)
        feat[f"{qname}_peak_nm"]   = round(q_peak, 2)
        feat[f"{qname}_mean_nm"]   = round(float(np.mean(q)), 2)
        feat[f"{qname}_rms_nm"]    = round(float(np.sqrt(np.mean(q ** 2))), 2)

        # Per-quarter sine-fit residual (sine waveforms only)
        if waveform == "sine" and dom_freq >= 0.5 and len(q) >= 4:
            q_std = float(np.std(q))
            if q_std < 1.0:
                feat[f"{qname}_sine_residual_pct"] = float("nan")
            else:
                t_q = np.arange(len(q)) / fs
                A_q = np.column_stack([
                    np.sin(2 * np.pi * dom_freq * t_q),
                    np.cos(2 * np.pi * dom_freq * t_q),
                ])
                try:
                    c, _, _, _ = np.linalg.lstsq(A_q, q, rcond=None)
                    res = q - A_q @ c
                    res_pct = float(np.sqrt(np.mean(res ** 2)) / q_std * 100)
                    feat[f"{qname}_sine_residual_pct"] = round(res_pct, 2)
                except Exception:
                    feat[f"{qname}_sine_residual_pct"] = float("nan")

    if q_peaks:
        feat["amplitude_drift_nm"] = round(float(max(q_peaks) - min(q_peaks)), 2)
        feat["worst_quarter"]      = ["q1", "q2", "q3", "q4"][int(np.argmax(q_peaks))]

    return feat


# ──────────────────────────────────────────────────────
# FOPDT-DERIVED FEATURES  (meter repo model, same device)
# ──────────────────────────────────────────────────────
def fopdt_features(dom_freq, peak_nm):
    """
    Compute FOPDT model predictions using calibrated meter repo parameters.
    These give a physics-based reference to separate linear dynamics from
    nonlinear distortion (hysteresis, creep).

    Returns:
      fopdt_phase_lag_deg  : phase lag the linear model predicts at dom_freq
      fopdt_attenuation    : amplitude attenuation factor at dom_freq
      v_drive_est_v        : estimated drive voltage (peak_nm / effective gain)
    """
    if dom_freq < 0.5 or np.isnan(peak_nm):
        return {
            "fopdt_phase_lag_deg": float("nan"),
            "fopdt_attenuation":   float("nan"),
            "v_drive_est_v":       float("nan"),
        }

    omega = 2 * np.pi * dom_freq
    tau   = FOPDT_TAU_S
    theta = FOPDT_THETA_S
    K     = FOPDT_K_NM_PER_V

    atten        = 1.0 / np.sqrt(1.0 + (omega * tau) ** 2)
    phase_deg    = np.degrees(np.arctan(omega * tau)) + np.degrees(omega * theta)
    v_drive_est  = peak_nm / (K * atten) if atten > 1e-12 else float("nan")

    return {
        "fopdt_phase_lag_deg": round(float(phase_deg), 3),
        "fopdt_attenuation":   round(float(atten), 6),
        "v_drive_est_v":       round(float(v_drive_est), 4),
    }


# ──────────────────────────────────────────────────────
# HYSTERESIS PROXY  (sine waveforms)
# ──────────────────────────────────────────────────────
def hysteresis_proxy(window, dom_freq, fs=FS):
    """
    For sine waveforms, hysteresis causes asymmetry between positive and
    negative half-cycles. Measure the mean positive peak vs mean negative peak.

    hysteresis_nm         : |mean_pos_peaks - |mean_neg_peaks||
    half_cycle_asymmetry  : hysteresis_nm / overall_peak_nm (fraction)
    """
    if dom_freq < 0.5 or len(window) < 10:
        return {"hysteresis_nm": float("nan"), "half_cycle_asymmetry": float("nan")}

    # Find positive and negative peaks
    min_dist = max(3, int(fs / (2.0 * dom_freq)))  # half-period distance
    pos_peaks, _ = find_peaks( window, height=0.1 * np.max(np.abs(window)), distance=min_dist)
    neg_peaks, _ = find_peaks(-window, height=0.1 * np.max(np.abs(window)), distance=min_dist)

    if len(pos_peaks) == 0 or len(neg_peaks) == 0:
        return {"hysteresis_nm": float("nan"), "half_cycle_asymmetry": float("nan")}

    mean_pos =  float(np.mean(window[pos_peaks]))
    mean_neg = -float(np.mean(window[neg_peaks]))   # negate to get magnitude
    hyst_nm  = abs(mean_pos - mean_neg)
    overall_peak = float(np.max(np.abs(window)))
    asymmetry = hyst_nm / max(overall_peak, 1e-30)

    return {
        "hysteresis_nm":        round(hyst_nm, 2),
        "half_cycle_asymmetry": round(asymmetry, 4),
    }


# ──────────────────────────────────────────────────────
# PROCESS ONE FILE
# ──────────────────────────────────────────────────────
def process_file(filepath, waveform_label):
    try:
        signal, window = load_signal(filepath)

        fft_feat = fft_features(signal)
        if fft_feat is None:
            print(f"  SKIP (too short): {os.path.basename(filepath)}")
            return None

        td_feat  = time_domain_features(signal)
        dom_freq = fft_feat["dominant_freq_hz"]
        peak_nm  = td_feat.get("peak_nm", float("nan"))

        # Waveform-specific features (full signal)
        if waveform_label == "sine":
            specific = sine_specific_features(signal, dom_freq)
        elif waveform_label == "square":
            specific = square_specific_features(signal)
        elif waveform_label == "noise":
            specific = noise_specific_features(signal)
        elif waveform_label == "ramp":
            specific = ramp_specific_features(signal)
        elif waveform_label == "pulse":
            specific = pulse_specific_features(signal)
        else:
            specific = {}

        # New: per-quarter features from image window (480–740 ms)
        q_feat = quarter_features(window, waveform_label, dom_freq)

        # New: FOPDT model context (meter repo, same device)
        fopdt_feat = fopdt_features(dom_freq, peak_nm)

        # New: hysteresis proxy (sine only)
        hyst_feat = hysteresis_proxy(window, dom_freq) if waveform_label == "sine" else {}

        row = {
            "filename":  os.path.basename(filepath),
            "waveform":  waveform_label,
            "n_samples": len(signal),
        }
        row.update(fft_feat)
        row.update(td_feat)
        row.update(specific)
        row.update(q_feat)
        row.update(fopdt_feat)
        row.update(hyst_feat)
        return row

    except Exception as e:
        print(f"  ERROR {os.path.basename(filepath)}: {e}")
        return None


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch piezoelectric feature extraction")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory with sine/square/noise/ramp/pulse subfolders")
    parser.add_argument("--output", type=str, default="features.csv",
                        help="Output CSV path (default: features.csv)")
    args = parser.parse_args()

    all_rows = []
    total, ok = 0, 0

    for waveform in WAVEFORM_FOLDERS:
        folder = os.path.join(args.data_dir, waveform)
        if not os.path.isdir(folder):
            print(f"WARNING: folder not found, skipping: {folder}")
            continue

        csv_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
        print(f"\n[{waveform.upper()}] {len(csv_files)} files found")

        for fname in csv_files:
            total += 1
            row = process_file(os.path.join(folder, fname), waveform)
            if row:
                all_rows.append(row)
                ok += 1
                print(f"  OK  {fname:<35}  dom_freq={row['dominant_freq_hz']:>8.2f} Hz"
                      f"  THD={row['thd_pct']:>7.2f}%"
                      f"  entropy={row['spectral_entropy']:>6.3f}")

    if not all_rows:
        print("\nERROR: No files processed. Check your data_dir path and CSV format.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"Done. {ok}/{total} files processed successfully.")
    print(f"Feature matrix: {len(df)} rows x {len(df.columns)} columns")
    print(f"Saved to: {args.output}")
    print(f"{'='*60}")

    print("\nSample count per waveform:")
    print(df["waveform"].value_counts().to_string())

    print("\nMean values of key features per waveform:")
    key_cols = ["dominant_freq_hz", "thd_pct", "spectral_entropy",
                "spectral_flatness", "crest_factor", "kurtosis"]
    available = [c for c in key_cols if c in df.columns]
    print(df.groupby("waveform")[available].mean().round(3).to_string())


if __name__ == "__main__":
    main()
