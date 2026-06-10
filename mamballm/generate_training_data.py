"""
generate_training_data.py  (v2 — segmented temporal descriptions)
==================================================================
Generate Gemini vision descriptions for piezoelectric waveform images.

v2 changes vs v1:
  - Segmented temporal analysis: each description covers all 4 quarters
    of the visible 480–740 ms window (Q1 480-545ms, Q2 545-610ms,
    Q3 610-675ms, Q4 675-740ms)
  - Amplitude-quantified and deviation-focused output
  - Ends with a closed-loop control implication sentence
  - Case-insensitive feature matching (fixes the 90% empty-features bug)
  - Longer output budget (650 tokens) for richer descriptions

Usage:
    python generate_training_data.py --image_dir ./piezo_data --features_csv ./features.csv
    python generate_training_data.py --limit 5   # dry run / test
    python generate_training_data.py --output training_data_v2.csv

Outputs CSV with columns:
    image_path, waveform_label, split, features_json, description
"""

import os
import re
import json
import time
import base64
import argparse
import requests
import pandas as pd
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ─────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in piezoelectric actuator closed-loop control and vibration analysis. \
Your task is to diagnose how a measured displacement waveform deviates from its ideal \
reference waveform, and to prescribe quantitative corrections.

Rules:
- Frame every observation as: actual value vs ideal value → signed deviation. \
For example: "THD = 5.38% vs ideal 0% → +5.38% excess distortion."
- Use the provided numerical metrics directly — do not guess values from the image alone.
- The CORRECTION VECTOR must list specific magnitudes (°, %, nm, dB) for every axis.
- Focus on vibration-relevant quantities: harmonic content, phase lag, amplitude error, \
resonance signatures, and temporal drift — not visual appearance.
- Do NOT describe what the waveform looks like. Diagnose what is wrong and what must change.
- Target 200–250 words. No padding or repetition.
- Write exclusively in English. Do not use any other language."""

# The visible window in every image is always 480–740 ms (260 ms total).
# Divide into four equal quarters:
#   Q1: 480–545 ms   Q2: 545–610 ms   Q3: 610–675 ms   Q4: 675–740 ms
_SEGMENT_HEADER = """\
The visible time window spans 480–740 ms (260 ms total).
Divide your analysis into four equal quarters:
  Q1: 480–545 ms | Q2: 545–610 ms | Q3: 610–675 ms | Q4: 675–740 ms\
"""

PROMPT_TEMPLATES = {

    # ── SINE ──────────────────────────────────────────────────────
    "sine": """\
A piezoelectric actuator was commanded to produce a sinusoidal displacement at \
{dominant_freq_hz} Hz. The measured waveform (heterodyne interferometry, 480–740 ms) \
is shown above.

IDEAL REFERENCE: perfect sine at {dominant_freq_hz} Hz, {peak_nm} nm peak, \
zero DC offset, THD = 0%, all harmonics = 0.
FOPDT linear model (meter repo, same device — K={fopdt_K} nm/V, τ=250 µs, θ=5 µs) \
predicts {fopdt_phase_lag_deg}° phase lag and {fopdt_atten_pct}% amplitude attenuation \
at this frequency. Estimated drive voltage: {v_drive_est_v} V.

Full-window deviation metrics:
  Peak displacement        : {peak_nm} nm   (ideal: {peak_nm} nm — shape deviation only)
  RMS displacement         : {rms_nm} nm    (ideal sine RMS = peak/√2 ≈ {ideal_rms} nm)
  Crest factor             : {crest_factor} (ideal sine = 1.414)
  THD                      : {thd_pct}%     (ideal: 0%)
  Sine-fit residual        : {sine_residual_pct}%  (0% = perfect sine)
  Measured phase           : {sine_phase_lag_deg}° (FOPDT linear prediction: {fopdt_phase_lag_deg}°)
  2nd harmonic / fund.     : {harmonic_ratio_2}    (ideal: 0)
  3rd harmonic / fund.     : {harmonic_ratio_3}    (ideal: 0)
  Hysteresis (pos/neg asymmetry): {hysteresis_nm} nm  (ideal: 0 nm)
  Half-cycle asymmetry     : {half_cycle_asymmetry} (ideal: 0)

Per-quarter measured peaks (480–740 ms window):
  Q1 (480–545 ms): peak = {q1_peak_nm} nm, mean = {q1_mean_nm} nm, \
sine residual = {q1_sine_residual_pct}%
  Q2 (545–610 ms): peak = {q2_peak_nm} nm, mean = {q2_mean_nm} nm, \
sine residual = {q2_sine_residual_pct}%
  Q3 (610–675 ms): peak = {q3_peak_nm} nm, mean = {q3_mean_nm} nm, \
sine residual = {q3_sine_residual_pct}%
  Q4 (675–740 ms): peak = {q4_peak_nm} nm, mean = {q4_mean_nm} nm, \
sine residual = {q4_sine_residual_pct}%
  Amplitude drift across window: {amplitude_drift_nm} nm

Write a deviation analysis in exactly these four parts — no bullet points, continuous prose:

1. DEVIATION SUMMARY: For each full-window metric, state "actual vs ideal → signed \
deviation." Compare measured phase to FOPDT linear prediction: the difference is the \
nonlinear phase excess due to hysteresis/creep. Identify the dominant error.

2. TEMPORAL DEVIATION (Q1–Q4): Use the per-quarter peak and sine residual values \
above. Identify which quarter deviates most from the full-window average peak \
({peak_nm} nm) and by how many nm. State whether distortion increases, decreases, \
or is stable across the window (use amplitude_drift_nm as evidence of creep or drift).

3. VIBRATION ANALYSIS: Attribute distortion to hysteresis (half-cycle asymmetry \
{hysteresis_nm} nm, nonlinear phase excess beyond FOPDT), creep (amplitude drift \
{amplitude_drift_nm} nm across window), or bandwidth saturation (crest factor \
{crest_factor} vs ideal 1.414). State dominant harmonic frequency and fraction. \
Classify vibration as periodic-steady, quasi-periodic, or drifting.

4. CORRECTION VECTOR: List every required closed-loop correction with exact magnitude — \
nonlinear phase compensation beyond FOPDT (±°), harmonic suppression (Nth harmonic, \
target % reduction), hysteresis feedforward (±nm per half-cycle), creep compensation \
(nm/ms drift rate if applicable), DC offset correction (±nm).""",

    # ── SQUARE ────────────────────────────────────────────────────
    "square": """\
A piezoelectric actuator was commanded to produce a square-wave displacement at \
{dominant_freq_hz} Hz. The measured waveform (heterodyne interferometry, 480–740 ms) \
is shown above.

IDEAL REFERENCE: perfect square wave at {dominant_freq_hz} Hz, {peak_nm} nm peak, \
50% duty cycle, instantaneous edges, no ringing, only odd harmonics at theoretical \
ratios (3rd = 33.3%, 5th = 20.0%, 7th = 14.3% of fundamental).

Measured deviation metrics:
  Peak displacement        : {peak_nm} nm
  RMS displacement         : {rms_nm} nm
  THD                      : {thd_pct}%
  Odd/even harmonic ratio  : {odd_even_harmonic_ratio} (ideal: ∞ — even harmonics absent)
  3rd harmonic / fund.     : {harmonic_ratio_3}  (ideal: 0.333)
  Duty cycle               : {square_duty_cycle}  (ideal: 0.500)
  Edge sharpness           : {square_edge_sharpness} (ideal: 1.000 — instantaneous)
  Kurtosis                 : {kurtosis}

Time window: 480–740 ms → Q1: 480–545 ms | Q2: 545–610 ms | Q3: 610–675 ms | Q4: 675–740 ms

Write a deviation analysis in exactly these four parts — no bullet points, continuous prose:

1. DEVIATION SUMMARY: For each metric, state "actual vs ideal → signed deviation." \
Compute duty-cycle error (actual − 0.500), edge-sharpness deficit (1.000 − actual), \
and odd/even harmonic contamination. Identify the dominant error.

2. TEMPORAL DEVIATION (Q1–Q4): For each quarter state whether edge rounding, \
ringing, or amplitude asymmetry is worse or better than the full-cycle average. \
Identify which transition (rising or falling edge) carries more distortion.

3. VIBRATION ANALYSIS: Attribute edge rounding to actuator bandwidth or hysteresis; \
attribute ringing to mechanical resonance. State the resonant frequency if detectable \
from harmonic content. Describe whether errors are consistent or drifting across cycles.

4. CORRECTION VECTOR: Phase pre-emphasis (±°), bandwidth extension requirement, \
duty-cycle trim (target ± Δ from 0.500), resonance notch frequency (Hz) and depth (dB), \
and which edge (rising/falling) needs more aggressive compensation.""",

    # ── NOISE ─────────────────────────────────────────────────────
    "noise": """\
A piezoelectric actuator was driven by broadband random (noise) voltage for system \
identification. The measured displacement (heterodyne interferometry, 480–740 ms) \
is shown above.

IDEAL REFERENCE: stationary, zero-mean Gaussian white noise with uniform spectral \
density, crest factor ≈ 3–4, spectral flatness = 1.0, autocorrelation lag-1 ≈ 0, \
Gaussianity error = 0.

Measured deviation metrics:
  RMS displacement         : {rms_nm} nm
  Peak displacement        : {peak_nm} nm
  Crest factor             : {crest_factor} (ideal broadband ≈ 3–4)
  Spectral entropy         : {spectral_entropy}  (ideal white noise = maximum)
  Spectral flatness        : {spectral_flatness}  (ideal: 1.0)
  Band energy ratio        : {band_energy_ratio}
  Kurtosis                 : {kurtosis}  (ideal Gaussian = 3.0)
  Gaussianity error        : {noise_gaussianity_err} (ideal: 0)
  Autocorrelation lag-1    : {noise_autocorr_lag1}  (ideal: 0)

Time window: 480–740 ms → Q1: 480–545 ms | Q2: 545–610 ms | Q3: 610–675 ms | Q4: 675–740 ms

Write a deviation analysis in exactly these four parts — no bullet points, continuous prose:

1. DEVIATION SUMMARY: For each metric, state "actual vs ideal → signed deviation." \
Determine whether the excitation is bandwidth-limited, non-Gaussian, non-stationary, \
or colored (correlated). Identify the dominant quality failure.

2. TEMPORAL DEVIATION (Q1–Q4): State whether amplitude envelope or spectral density \
appears to shift between quarters. If Q1–Q4 are statistically identical, say so in \
one sentence. Otherwise identify which quarter shows a burst, dropout, or level shift.

3. VIBRATION ANALYSIS: State whether spectral coloring is due to actuator mechanical \
resonance (narrow peak in spectral entropy), amplifier bandwidth roll-off (low spectral \
flatness), or sensor noise floor clipping. Estimate the usable identification bandwidth \
in Hz from these metrics.

4. CORRECTION VECTOR: Whitening filter order and cutoff (Hz), re-scaling of drive \
amplitude (±% to hit target RMS), any notch needed to suppress resonance peak (Hz, dB), \
and whether a non-Gaussian correction (clipping, kurtosis target) is required.""",

    # ── RAMP ──────────────────────────────────────────────────────
    "ramp": """\
A piezoelectric actuator was commanded to produce a sawtooth (ramp) displacement at \
{dominant_freq_hz} Hz. The measured waveform (heterodyne interferometry, 480–740 ms) \
is shown above.

IDEAL REFERENCE: perfectly linear sawtooth at {dominant_freq_hz} Hz, {peak_nm} nm \
peak-to-peak, rise linearity = 1.000, fall linearity = 1.000, zero asymmetry, \
instantaneous flyback.

Measured deviation metrics:
  Peak displacement        : {peak_nm} nm
  RMS displacement         : {rms_nm} nm
  THD                      : {thd_pct}%
  2nd harmonic / fund.     : {harmonic_ratio_2}  (ramp ideal: 0.500)
  3rd harmonic / fund.     : {harmonic_ratio_3}  (ramp ideal: 0.333)
  Rise linearity           : {ramp_rise_linearity}  (ideal: 1.000)
  Fall linearity           : {ramp_fall_linearity}  (ideal: 1.000)
  Rise/fall asymmetry      : {ramp_asymmetry}   (ideal: 0)
  Kurtosis                 : {kurtosis}

Time window: 480–740 ms → Q1: 480–545 ms | Q2: 545–610 ms | Q3: 610–675 ms | Q4: 675–740 ms

Write a deviation analysis in exactly these four parts — no bullet points, continuous prose:

1. DEVIATION SUMMARY: For each metric, state "actual vs ideal → signed deviation." \
Compute rise-linearity deficit (1.000 − actual), fall-linearity deficit, and asymmetry \
magnitude. Identify whether nonlinearity is concentrated in the rising slope, \
falling slope, or flyback.

2. TEMPORAL DEVIATION (Q1–Q4): For each quarter state whether slope curvature \
(S-shape, concavity) is larger or smaller than the full-cycle average. Identify \
the quarter with maximum deviation from linearity and estimate the nm error \
at mid-ramp.

3. VIBRATION ANALYSIS: Attribute rise nonlinearity to piezoelectric creep \
(progressive concavity) or hysteresis (asymmetric rise vs fall). State whether \
the flyback is abrupt (bandwidth-limited by amplifier) or gradual (creep recovery). \
Describe whether errors accumulate or are consistent cycle-to-cycle.

4. CORRECTION VECTOR: Feedforward nonlinearity pre-compensation (rising slope \
correction in nm at quarter-span), flyback timing adjustment (ms), asymmetry \
correction (±% amplitude on rise vs fall), and which quarter requires the largest \
iterative-learning-control update.""",

    # ── PULSE ─────────────────────────────────────────────────────
    "pulse": """\
A piezoelectric actuator was driven by impulse/pulse voltage signals. \
The measured displacement (heterodyne interferometry, 480–740 ms) is shown above.

IDEAL REFERENCE: clean rectangular pulses with zero post-pulse ringing, flat \
baseline between pulses, symmetric positive/negative excursions if bipolar, \
pulse crest factor matching drive amplitude, ringing ratio = 0.

Measured deviation metrics:
  Peak displacement        : {peak_nm} nm
  RMS displacement         : {rms_nm} nm
  Crest factor             : {crest_factor}
  Pulse crest factor       : {pulse_crest_factor}  (ideal = drive-defined target)
  Pulse duty cycle         : {pulse_duty_cycle}    (ideal = commanded duty cycle)
  Post-pulse ringing ratio : {pulse_ringing_ratio} (ideal: 0)
  Spectral entropy         : {spectral_entropy}
  Kurtosis                 : {kurtosis}

Time window: 480–740 ms → Q1: 480–545 ms | Q2: 545–610 ms | Q3: 610–675 ms | Q4: 675–740 ms

Write a deviation analysis in exactly these four parts — no bullet points, continuous prose:

1. DEVIATION SUMMARY: For each metric, state "actual vs ideal → signed deviation." \
Quantify ringing as % of pulse peak and estimate its decay time constant. Identify \
whether the dominant error is ringing, amplitude undershoot, duty-cycle error, \
or baseline drift.

2. TEMPORAL DEVIATION (Q1–Q4): For each quarter state whether pulses are present, \
whether ringing amplitude is larger or smaller than the full-window average, \
and whether the baseline drifts between pulses. Identify the quarter with worst \
post-pulse oscillation.

3. VIBRATION ANALYSIS: Attribute ringing to mechanical resonance of the \
actuator-load system (identify resonant frequency in Hz from ringing period) or \
electrical ringing in the drive amplifier. State whether the ringing is underdamped \
(oscillatory) or overdamped (exponential). Describe whether pulse amplitude is \
consistent or modulated across the window.

4. CORRECTION VECTOR: Input-shaping notch frequency (Hz) and damping ratio, \
active damping gain required (estimate in dB reduction of ringing), duty-cycle \
trim (±% from commanded), baseline offset compensation (nm), and which quarter \
needs per-pulse feedforward suppression.""",
}


# ─────────────────────────────────────────────────────────────────
# FEATURE MATCHING  (case-insensitive, v2 fix)
# ─────────────────────────────────────────────────────────────────

def _clean_stem(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r'_aug_v\d+$', '', s)
    s = re.sub(r'_orig$', '', s)
    # collapse duplicate freq tokens like "100hz_100hz" → "100hz"
    s = re.sub(r'(\d+hz)[_-]\1', r'\1', s)
    return s


def build_feat_index(features_df: pd.DataFrame) -> dict:
    """Build a case-insensitive lookup: cleaned_stem → feature dict."""
    index = {}
    for _, row in features_df.iterrows():
        raw_name = row["filename"]          # e.g. "sine-100Hz-10_absolute.csv"
        stem = Path(raw_name).stem          # e.g. "sine-100Hz-10_absolute"
        key  = _clean_stem(stem)            # lowercase + stripped
        index[key] = row.to_dict()
    return index


def find_features(png_path: str, feat_index: dict) -> dict | None:
    base = _clean_stem(Path(png_path).stem)
    # Try several normalisation variants
    candidates = [
        base,
        base.replace('_', '-'),
        re.sub(r'[-_]+', '-', base),
        re.sub(r'[-_]+', '_', base),
    ]
    for c in candidates:
        if c in feat_index:
            return feat_index[c]
    return None


# ─────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 3) -> str:
    if val is None:
        return "N/A"
    try:
        if pd.isna(val):
            return "N/A"
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def build_prompt(waveform: str, feat_row: dict | None) -> str:
    row = feat_row or {}

    # Derived: ideal RMS for a pure sine
    try:
        ideal_rms = f"{float(row['peak_nm']) / (2 ** 0.5):.2f}"
    except (KeyError, TypeError, ValueError):
        ideal_rms = "N/A"

    # FOPDT attenuation as percentage loss
    try:
        fopdt_atten_pct = f"{(1.0 - float(row['fopdt_attenuation'])) * 100:.2f}"
    except (KeyError, TypeError, ValueError):
        fopdt_atten_pct = "N/A"

    fmt = {
        "ideal_rms":               ideal_rms,
        "fopdt_K":                 "380",
        "fopdt_atten_pct":         fopdt_atten_pct,
        "dominant_freq_hz":        _fmt(row.get("dominant_freq_hz"), 2),
        "rms_nm":                  _fmt(row.get("rms_nm"), 2),
        "peak_nm":                 _fmt(row.get("peak_nm"), 2),
        "thd_pct":                 _fmt(row.get("thd_pct"), 2),
        "sine_residual_pct":       _fmt(row.get("sine_residual_pct"), 2),
        "sine_phase_lag_deg":      _fmt(row.get("sine_phase_lag_deg"), 2),
        "harmonic_ratio_2":        _fmt(row.get("harmonic_ratio_2"), 4),
        "harmonic_ratio_3":        _fmt(row.get("harmonic_ratio_3"), 4),
        "odd_even_harmonic_ratio": _fmt(row.get("odd_even_harmonic_ratio"), 3),
        "square_duty_cycle":       _fmt(row.get("square_duty_cycle"), 3),
        "square_edge_sharpness":   _fmt(row.get("square_edge_sharpness"), 5),
        "spectral_entropy":        _fmt(row.get("spectral_entropy"), 3),
        "spectral_flatness":       _fmt(row.get("spectral_flatness"), 4),
        "band_energy_ratio":       _fmt(row.get("band_energy_ratio"), 2),
        "kurtosis":                _fmt(row.get("kurtosis"), 3),
        "crest_factor":            _fmt(row.get("crest_factor"), 3),
        "noise_gaussianity_err":   _fmt(row.get("noise_gaussianity_err"), 3),
        "noise_autocorr_lag1":     _fmt(row.get("noise_autocorr_lag1"), 4),
        "ramp_rise_linearity":     _fmt(row.get("ramp_rise_linearity"), 4),
        "ramp_fall_linearity":     _fmt(row.get("ramp_fall_linearity"), 4),
        "ramp_asymmetry":          _fmt(row.get("ramp_asymmetry"), 4),
        "pulse_crest_factor":      _fmt(row.get("pulse_crest_factor"), 3),
        "pulse_duty_cycle":        _fmt(row.get("pulse_duty_cycle"), 4),
        "pulse_ringing_ratio":     _fmt(row.get("pulse_ringing_ratio"), 4),
        # FOPDT model features (meter repo, same device)
        "fopdt_phase_lag_deg":     _fmt(row.get("fopdt_phase_lag_deg"), 2),
        "fopdt_attenuation":       _fmt(row.get("fopdt_attenuation"), 4),
        "v_drive_est_v":           _fmt(row.get("v_drive_est_v"), 3),
        # Hysteresis proxy
        "hysteresis_nm":           _fmt(row.get("hysteresis_nm"), 2),
        "half_cycle_asymmetry":    _fmt(row.get("half_cycle_asymmetry"), 4),
        # Per-quarter features (480–740 ms window)
        "q1_peak_nm":              _fmt(row.get("q1_peak_nm"), 2),
        "q2_peak_nm":              _fmt(row.get("q2_peak_nm"), 2),
        "q3_peak_nm":              _fmt(row.get("q3_peak_nm"), 2),
        "q4_peak_nm":              _fmt(row.get("q4_peak_nm"), 2),
        "q1_mean_nm":              _fmt(row.get("q1_mean_nm"), 2),
        "q2_mean_nm":              _fmt(row.get("q2_mean_nm"), 2),
        "q3_mean_nm":              _fmt(row.get("q3_mean_nm"), 2),
        "q4_mean_nm":              _fmt(row.get("q4_mean_nm"), 2),
        "q1_rms_nm":               _fmt(row.get("q1_rms_nm"), 2),
        "q2_rms_nm":               _fmt(row.get("q2_rms_nm"), 2),
        "q3_rms_nm":               _fmt(row.get("q3_rms_nm"), 2),
        "q4_rms_nm":               _fmt(row.get("q4_rms_nm"), 2),
        "q1_sine_residual_pct":    _fmt(row.get("q1_sine_residual_pct"), 2),
        "q2_sine_residual_pct":    _fmt(row.get("q2_sine_residual_pct"), 2),
        "q3_sine_residual_pct":    _fmt(row.get("q3_sine_residual_pct"), 2),
        "q4_sine_residual_pct":    _fmt(row.get("q4_sine_residual_pct"), 2),
        "amplitude_drift_nm":      _fmt(row.get("amplitude_drift_nm"), 2),
    }
    body = PROMPT_TEMPLATES[waveform].format(**fmt)
    return SYSTEM_PROMPT + "\n\n" + body


# ─────────────────────────────────────────────────────────────────
# GEMINI API
# ─────────────────────────────────────────────────────────────────

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


def call_gemini(
    api_key: str,
    waveform: str,
    feat_row: dict | None,
    image_path: str,
    max_retries: int = 4,
) -> str | None:
    prompt_text = build_prompt(waveform, feat_row)
    image_b64   = base64.b64encode(Path(image_path).read_bytes()).decode()

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                {"text": prompt_text},
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 800,
            "temperature":     0.30,
            "thinkingConfig":  {"thinkingBudget": 0},
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=40,
            )
            if resp.status_code == 429:
                wait = 12 * (2 ** attempt)
                print(f"\n    [rate-limit] sleeping {wait}s …", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except requests.Timeout:
            print(f"\n    [timeout {attempt+1}/{max_retries}] retrying …", flush=True)
            time.sleep(5)
        except Exception as exc:
            print(f"\n    [error {attempt+1}/{max_retries}] {type(exc).__name__}: {exc}",
                  flush=True)
            if attempt == max_retries - 1:
                return None
            time.sleep(5)

    return None


# ─────────────────────────────────────────────────────────────────
# CLEAN OUTPUT
# ─────────────────────────────────────────────────────────────────

_PREAMBLE_RE = re.compile(
    r"^(Here'?s?( a| my| the)? (technical )?description[^:]*:\s*"
    r"|Here'?s?( a| my)? analysis[^:]*:\s*"
    r"|The following (is|describes)[^:]*:\s*"
    r"|Output:\s*)",
    re.IGNORECASE,
)


def clean_description(text: str) -> str:
    return _PREAMBLE_RE.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

WAVEFORMS = ["sine", "square", "noise", "ramp", "pulse"]


def main():
    parser = argparse.ArgumentParser(description="Generate segmented waveform descriptions (v2)")
    parser.add_argument("--image_dir",    default="./piezo_data")
    parser.add_argument("--features_csv", default="./features.csv")
    parser.add_argument("--output",       default="training_data_v2.csv")
    parser.add_argument("--splits",       nargs="+", default=["train", "val", "test"])
    parser.add_argument("--limit",        type=int, default=None,
                        help="Cap total images for a dry run")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY not set in environment or .env file.")

    # ── Load features (case-insensitive index) ─────────────────
    features_df = pd.read_csv(args.features_csv)
    valid = features_df["rms_nm"].notna()
    if (~valid).sum():
        print(f"[features] Dropping {(~valid).sum()} corrupt row(s).")
    features_df  = features_df[valid]
    feat_index   = build_feat_index(features_df)
    print(f"[features] Loaded {len(feat_index)} valid entries.")

    # ── Collect PNGs ───────────────────────────────────────────
    all_images: list[tuple[str, str, str]] = []
    for split in args.splits:
        for waveform in WAVEFORMS:
            folder = Path(args.image_dir) / split / waveform
            if folder.is_dir():
                for png in sorted(folder.glob("*.png")):
                    all_images.append((str(png), waveform, split))

    print(f"[images]   Found {len(all_images)} PNGs.")
    if args.limit:
        all_images = all_images[:args.limit]
        print(f"[images]   Capped at {args.limit}.")

    # ── Resume from existing output ────────────────────────────
    done_paths: set[str] = set()
    results:    list[dict] = []
    out_path = Path(args.output)
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing    = pd.read_csv(out_path)
            results     = existing.to_dict("records")
            done_paths  = set(existing["image_path"].tolist())
            print(f"[resume]   {len(done_paths)} already done — skipping.")
        except Exception:
            print("[resume]   Existing output unreadable — starting fresh.")

    # ── Process ────────────────────────────────────────────────
    total  = len(all_images)
    failed = 0
    feat_hit = 0

    for i, (png_path, waveform, split) in enumerate(all_images, 1):
        if png_path in done_paths:
            continue

        feat_row  = find_features(png_path, feat_index)
        feat_dict = (
            {k: v for k, v in feat_row.items() if k not in ("filename", "waveform")}
            if feat_row else {}
        )
        if feat_row:
            feat_hit += 1

        label = Path(png_path).name[:45].ljust(45)
        feat_tag = "[feat]" if feat_row else "[nofeat]"
        print(f"[{i:4d}/{total}] {split}/{waveform}  {label}  {feat_tag}", end="  ", flush=True)

        description = call_gemini(api_key, waveform, feat_row, png_path)
        if description is None:
            failed += 1
            print("FAILED")
            continue

        description = clean_description(description)

        results.append({
            "image_path":     png_path,
            "waveform_label": waveform,
            "split":          split,
            "features_json":  json.dumps(feat_dict),
            "description":    description,
        })
        done_paths.add(png_path)
        print(f"OK  ({len(description)} chars)")

        # Free-tier rate limit: 15 req/min → 1 req/4 s
        time.sleep(4)

        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(args.output, index=False)

    pd.DataFrame(results).to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"Done.  {len(results)} descriptions saved → {args.output}")
    print(f"Feature match rate : {feat_hit}/{total - len(done_paths) + feat_hit} images had features")
    print(f"Failed             : {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
