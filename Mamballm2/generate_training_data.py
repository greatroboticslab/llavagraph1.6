"""
generate_training_data.py
=========================
Generate training data for MambaLLM piezo waveform correction.

Pipeline per image:
  1. Load PNG + look up features from features_full.csv
  2. Compute CORRECTION VECTOR from features (signal processing, not AI)
  3. Send PNG + features to Gemini → get 3-section text description
  4. Concatenate description + correction vector → save to training_data.csv

Output format per sample:
  DEVIATION SUMMARY: ...
  TEMPORAL DEVIATION: ...
  VIBRATION ANALYSIS: ...

  CORRECTION VECTOR:
  phase_offset_deg = +8.96
  amplitude_scale  = 1.023
  ...

Usage:
    python generate_training_data.py --limit 5    # test first
    python generate_training_data.py              # full run (~2.7h free tier)
"""

import os, re, json, time, base64, argparse, requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

WAVEFORMS = ["sine", "square", "noise", "ramp", "pulse"]

# ── Name matching ─────────────────────────────────────────────────────────────

def _clean_stem(stem: str) -> str:
    s = re.sub(r'_aug_v\d+$', '', stem, flags=re.IGNORECASE)
    s = re.sub(r'_orig$',     '', s,    flags=re.IGNORECASE)
    s = s.lower().replace('_', '-')
    # Note: do NOT remove duplicate frequency (100hz-100hz) —
    # that would cause collisions between old and new naming conventions.
    return s


def build_feat_index(features_csv: str) -> dict:
    df = pd.read_csv(features_csv)
    index = {}
    for _, row in df.iterrows():
        key = _clean_stem(Path(row["filename"]).stem)
        index[key] = row.to_dict()
    return index


def find_features(png_path: str, feat_index: dict):
    key = _clean_stem(Path(png_path).stem)
    return feat_index.get(key)


# ── Correction vector (computed from features, not guessed) ───────────────────

def _f(val, default=0.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def compute_correction_vector(waveform: str, feat: dict) -> dict:
    """
    Derive correction parameters directly from measured features.
    Each value = mathematical inverse of the measured error.
    """
    cv = {}

    if waveform == "sine":
        # Phase: negate measured lag to get required advance
        phase_lag = _f(feat.get("sine_phase_lag_deg"), 0.0)
        cv["phase_offset_deg"] = round(-phase_lag, 3)

        # Amplitude: ratio of ideal RMS to measured RMS
        peak = _f(feat.get("peak_nm"), 0.0)
        rms  = _f(feat.get("rms_nm"),  1.0)
        ideal_rms = peak / (2 ** 0.5) if peak > 0 else rms
        cv["amplitude_scale"] = round(ideal_rms / rms, 4) if rms > 0 else 1.0

        # Harmonic cancellation (negate to cancel)
        h2 = _f(feat.get("harmonic_ratio_2"), 0.0)
        h3 = _f(feat.get("harmonic_ratio_3"), 0.0)
        cv["harmonic_2_amp"] = round(-h2, 5) if abs(h2) > 0.005 else 0.0
        cv["harmonic_3_amp"] = round(-h3, 5) if abs(h3) > 0.005 else 0.0

        # DC offset: negate quarter mean average
        means = [_f(feat.get(f"q{i}_mean_nm"), 0.0) for i in range(1, 5)]
        dc = float(np.mean([m for m in means if m != 0.0])) if any(m != 0 for m in means) else 0.0
        cv["dc_offset_nm"] = round(-dc, 3)

        # Hysteresis feedforward: negate measured asymmetry
        hyst = _f(feat.get("hysteresis_nm"), 0.0)
        cv["hysteresis_ff_nm"] = round(-hyst, 3)

    elif waveform == "square":
        # Duty cycle trim: difference from ideal 0.5
        duty = _f(feat.get("square_duty_cycle"), 0.5)
        cv["duty_cycle_trim"] = round(0.5 - duty, 4)

        # Edge sharpness deficit
        edge = _f(feat.get("square_edge_sharpness"), 1.0)
        cv["edge_sharpness_deficit"] = round(1.0 - edge, 5)

        # Even harmonic cancellation (square should have none)
        h2 = _f(feat.get("harmonic_ratio_2"), 0.0)
        cv["harmonic_2_amp"] = round(-h2, 5) if abs(h2) > 0.01 else 0.0

        # DC offset
        means = [_f(feat.get(f"q{i}_mean_nm"), 0.0) for i in range(1, 5)]
        dc = float(np.mean([m for m in means if m != 0.0])) if any(m != 0 for m in means) else 0.0
        cv["dc_offset_nm"] = round(-dc, 3)

    elif waveform == "ramp":
        # Linearity corrections
        rise = _f(feat.get("ramp_rise_linearity"), 1.0)
        fall = _f(feat.get("ramp_fall_linearity"), 1.0)
        asym = _f(feat.get("ramp_asymmetry"),      0.0)
        cv["rise_linearity_correction"] = round(1.0 - rise, 4)
        cv["fall_linearity_correction"] = round(1.0 - fall, 4)
        cv["asymmetry_correction"]      = round(-asym, 4)

        # Amplitude
        peak = _f(feat.get("peak_nm"), 0.0)
        rms  = _f(feat.get("rms_nm"),  1.0)
        ideal_rms_ramp = peak / (3 ** 0.5)   # sawtooth: RMS = peak/√3
        cv["amplitude_scale"] = round(ideal_rms_ramp / rms, 4) if rms > 0 else 1.0

        means = [_f(feat.get(f"q{i}_mean_nm"), 0.0) for i in range(1, 5)]
        dc = float(np.mean([m for m in means if m != 0.0])) if any(m != 0 for m in means) else 0.0
        cv["dc_offset_nm"] = round(-dc, 3)

    elif waveform == "pulse":
        # Ringing suppression target
        ring = _f(feat.get("pulse_ringing_ratio"), 0.0)
        cv["ringing_suppression_target"] = round(ring, 4)

        # Duty cycle
        duty = _f(feat.get("pulse_duty_cycle"), 0.0)
        cv["pulse_duty_cycle_measured"] = round(duty, 4)

        # Amplitude
        peak = _f(feat.get("peak_nm"), 0.0)
        rms  = _f(feat.get("rms_nm"),  1.0)
        cv["amplitude_scale"] = round(peak / rms, 4) if rms > 0 else 1.0

        means = [_f(feat.get(f"q{i}_mean_nm"), 0.0) for i in range(1, 5)]
        dc = float(np.mean([m for m in means if m != 0.0])) if any(m != 0 for m in means) else 0.0
        cv["dc_offset_nm"] = round(-dc, 3)

    elif waveform == "noise":
        # Spectral whitening: how far from flat spectrum
        flatness = _f(feat.get("spectral_flatness"), 1.0)
        cv["whitening_required"] = round(1.0 - flatness, 4)

        # Amplitude rescaling
        rms = _f(feat.get("rms_nm"), 1.0)
        cv["rms_measured_nm"] = round(rms, 3)

        # Gaussianity error
        gauss_err = _f(feat.get("noise_gaussianity_err"), 0.0)
        cv["gaussianity_error"] = round(gauss_err, 4)

        autocorr = _f(feat.get("noise_autocorr_lag1"), 0.0)
        cv["autocorr_lag1"] = round(autocorr, 4)

    return cv


def format_correction_vector(cv: dict) -> str:
    lines = ["CORRECTION VECTOR:"]
    for k, v in cv.items():
        if isinstance(v, float) and v == 0.0:
            lines.append(f"  {k} = 0.0")
        else:
            sign = "+" if isinstance(v, (int, float)) and v > 0 else ""
            lines.append(f"  {k} = {sign}{v}")
    return "\n".join(lines)


# ── Gemini prompts (text description only — 3 sections) ──────────────────────

SYSTEM_PROMPT = (
    "You are a piezoelectric actuator control expert. "
    "The piezo was COMMANDED to produce the waveform stated below. "
    "Your job: write two short paragraphs that a non-expert engineer can understand.\n\n"
    "FORMAT (use these exact headers):\n\n"
    "DIAGNOSIS: [2-3 sentences] Explain in plain English WHY the measured output "
    "deviates from the command. Name the dominant physical mechanism "
    "(e.g. hysteresis saturation, bandwidth limitation, creep, resonance). "
    "Describe what physically happens to the signal — do not just list numbers. "
    "Include 3-4 key measured values WITH their meaning "
    "(e.g. 'phase lag reaches 172.2° — the linear model predicts only 1.84°, "
    "so 170° of excess lag is pure nonlinear distortion'). "
    "End with temporal behaviour: which quarter is worst and whether drift is "
    "stable, growing, or quasi-periodic.\n\n"
    "CORRECTION: [1-2 sentences] Explain in plain English WHAT must be done to "
    "restore the commanded waveform. Connect each correction action to the physical "
    "cause (e.g. 'advance phase by X° to counteract the lag', "
    "'cancel the Nth harmonic that creates the square-like shape'). "
    "Use the correction vector values as the specific numbers.\n\n"
    "RULES:\n"
    "1. Write for a reader who understands engineering but not piezo control.\n"
    "2. Never say 'the waveform looks like X' — say 'the commanded sine shows X'.\n"
    "3. Every number must have a unit and a brief explanation of what it means.\n"
    "4. Total length: 60-100 words across both paragraphs. Be concise.\n"
    "5. English only. No bullet points. No other sections."
)

# Ideal reference values per waveform type
_IDEALS = {
    "sine":   {"THD": "0%", "sine-fit residual": "0%",
               "crest factor": "1.414", "even harmonics": "0"},
    "square": {"THD (theoretical)": "48.3%", "duty cycle": "0.500",
               "edge sharpness": "1.000", "even harmonics": "0"},
    "noise":  {"spectral flatness": "1.000", "kurtosis": "3.0 (Gaussian)",
               "autocorr lag-1": "0", "gaussianity error": "0"},
    "ramp":   {"rise linearity": "1.000", "fall linearity": "1.000",
               "rise/fall asymmetry": "0"},
    "pulse":  {"post-pulse ringing": "0", "baseline between pulses": "flat"},
}

_FEAT_LABELS = {
    "dominant_freq_hz":   "Dominant frequency (Hz)",
    "rms_nm":             "RMS displacement (nm)",
    "peak_nm":            "Peak displacement (nm)",
    "thd_pct":            "THD (%)",
    "crest_factor":       "Crest factor",
    "harmonic_ratio_2":   "2nd harmonic ratio",
    "harmonic_ratio_3":   "3rd harmonic ratio",
    "odd_even_harmonic_ratio": "Odd/even harmonic ratio",
    "sine_residual_pct":  "Sine-fit residual (%)",
    "sine_phase_lag_deg": "Phase lag (deg)",
    "square_duty_cycle":  "Duty cycle",
    "square_edge_sharpness": "Edge sharpness",
    "noise_gaussianity_err": "Gaussianity error",
    "noise_autocorr_lag1":   "Autocorrelation lag-1",
    "spectral_flatness":  "Spectral flatness",
    "ramp_rise_linearity":"Rise linearity",
    "ramp_fall_linearity":"Fall linearity",
    "ramp_asymmetry":     "Rise/fall asymmetry",
    "pulse_ringing_ratio":"Post-pulse ringing ratio",
    "pulse_duty_cycle":   "Pulse duty cycle",
    "hysteresis_nm":      "Hysteresis (nm)",
    "half_cycle_asymmetry": "Half-cycle asymmetry",
    "fopdt_phase_lag_deg":"FOPDT predicted phase lag (deg)",
    "fopdt_attenuation":  "FOPDT attenuation factor",
    "amplitude_drift_nm": "Amplitude drift Q1→Q4 (nm)",
    "q1_peak_nm": "Q1 peak (nm)", "q2_peak_nm": "Q2 peak (nm)",
    "q3_peak_nm": "Q3 peak (nm)", "q4_peak_nm": "Q4 peak (nm)",
    "q1_mean_nm": "Q1 mean (nm)", "q2_mean_nm": "Q2 mean (nm)",
    "q3_mean_nm": "Q3 mean (nm)", "q4_mean_nm": "Q4 mean (nm)",
}

_EXCLUDE = {
    "sine":   {"odd_even_harmonic_ratio","square_duty_cycle","square_edge_sharpness",
               "noise_gaussianity_err","noise_autocorr_lag1","spectral_flatness",
               "ramp_rise_linearity","ramp_fall_linearity","ramp_asymmetry",
               "pulse_ringing_ratio","pulse_duty_cycle"},
    "square": {"sine_residual_pct","sine_phase_lag_deg","noise_gaussianity_err",
               "noise_autocorr_lag1","spectral_flatness","ramp_rise_linearity",
               "ramp_fall_linearity","ramp_asymmetry","pulse_ringing_ratio",
               "pulse_duty_cycle","hysteresis_nm","half_cycle_asymmetry"},
    "noise":  {"sine_residual_pct","sine_phase_lag_deg","odd_even_harmonic_ratio",
               "square_duty_cycle","square_edge_sharpness","ramp_rise_linearity",
               "ramp_fall_linearity","ramp_asymmetry","pulse_ringing_ratio",
               "pulse_duty_cycle","hysteresis_nm","half_cycle_asymmetry"},
    "ramp":   {"sine_residual_pct","sine_phase_lag_deg","odd_even_harmonic_ratio",
               "square_duty_cycle","square_edge_sharpness","noise_gaussianity_err",
               "noise_autocorr_lag1","spectral_flatness","pulse_ringing_ratio",
               "pulse_duty_cycle","hysteresis_nm","half_cycle_asymmetry"},
    "pulse":  {"sine_residual_pct","sine_phase_lag_deg","odd_even_harmonic_ratio",
               "square_duty_cycle","square_edge_sharpness","noise_gaussianity_err",
               "noise_autocorr_lag1","spectral_flatness","ramp_rise_linearity",
               "ramp_fall_linearity","ramp_asymmetry","hysteresis_nm",
               "half_cycle_asymmetry"},
}


def build_feature_text(waveform: str, feat: dict) -> str:
    exclude = _EXCLUDE.get(waveform, set())

    # Header: what was commanded + ideal reference
    freq = feat.get("dominant_freq_hz")
    peak = feat.get("peak_nm")
    freq_str = f"{float(freq):.1f} Hz" if freq and not np.isnan(float(freq)) else "unknown Hz"
    peak_str = f"±{float(peak):.1f} nm" if peak and not np.isnan(float(peak)) else "unknown nm"

    lines = [
        f"COMMANDED WAVEFORM: {waveform} at {freq_str}, target peak {peak_str}",
        "",
        "IDEAL REFERENCE for this waveform type:",
    ]
    for label, ideal_val in _IDEALS.get(waveform, {}).items():
        lines.append(f"  {label}: {ideal_val}")

    # FOPDT prediction (linear model baseline — shows how much is nonlinear)
    fopdt_phase = feat.get("fopdt_phase_lag_deg")
    fopdt_atten = feat.get("fopdt_attenuation")
    if fopdt_phase and not np.isnan(float(fopdt_phase)):
        lines.append(f"  FOPDT predicted phase lag: {float(fopdt_phase):.2f}° "
                     f"(any excess is nonlinear distortion)")
    if fopdt_atten and not np.isnan(float(fopdt_atten)):
        lines.append(f"  FOPDT predicted attenuation: {float(fopdt_atten):.4f}")

    lines += ["", "MEASURED FEATURES (480-740 ms window):"]

    for k, label in _FEAT_LABELS.items():
        if k in exclude:
            continue
        v = feat.get(k)
        if v is None:
            continue
        try:
            if np.isnan(float(v)):
                continue
            lines.append(f"  {label}: {float(v):.4g}")
        except (TypeError, ValueError):
            lines.append(f"  {label}: {v}")

    return "\n".join(lines)


# ── Gemini API ────────────────────────────────────────────────────────────────

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


def call_gemini(api_key: str, png_path: str, feat_text: str,
                max_retries: int = 4) -> str | None:
    image_b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{feat_text}\n\n"
        "Write the DIAGNOSIS and CORRECTION paragraphs now. "
        "Use the measured numbers above. Connect every number to its physical meaning. "
        "Keep DIAGNOSIS to 3 sentences and CORRECTION to 2 sentences. "
        "Always write complete sentences — never leave a sentence unfinished."
    )
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature":     0.20,
            "thinkingConfig":  {"thinkingBudget": 0},
        },
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(GEMINI_URL, params={"key": api_key},
                              json=payload, timeout=45)
            if r.status_code == 429:
                wait = 15 * (2 ** attempt)
                print(f"\n    [rate-limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.Timeout:
            print(f"\n    [timeout {attempt+1}/{max_retries}]", flush=True)
            time.sleep(5)
        except Exception as e:
            print(f"\n    [error {attempt+1}/{max_retries}] {e}", flush=True)
            if attempt == max_retries - 1:
                return None
            time.sleep(5)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(
        description="Generate MambaLLM training data using Gemini vision API."
    )
    parser.add_argument("--image_dir",    default="../vit_mamba/data",
                        help="Root folder containing train/val/test/<waveform>/*.png")
    parser.add_argument("--features_csv", default="./features_full.csv",
                        help="CSV with extracted signal features (493 rows)")
    parser.add_argument("--output",       default="./training_data.csv",
                        help="Output CSV (auto-resumed if it already exists)")
    parser.add_argument("--splits",       nargs="+",
                        default=["train", "val", "test"],
                        help="Which splits to process")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Process only the first N images (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY not set in .env")

    # ── Load features ──────────────────────────────────────────────────────────
    feat_index = build_feat_index(args.features_csv)
    print(f"\n  Features loaded : {len(feat_index)} entries")

    # ── Collect images ─────────────────────────────────────────────────────────
    all_images = []
    wf_counts: dict[str, int] = {}
    for split in args.splits:
        for wf in WAVEFORMS:
            folder = Path(args.image_dir) / split / wf
            if folder.is_dir():
                imgs = sorted(folder.glob("*.png"))
                all_images.extend((str(p), wf, split) for p in imgs)
                wf_counts[wf] = wf_counts.get(wf, 0) + len(imgs)

    print(f"  Images found    : {len(all_images)}  "
          + "  ".join(f"{wf}={n}" for wf, n in wf_counts.items()))

    if args.limit:
        all_images = all_images[:args.limit]
        print(f"  Limit active    : processing first {args.limit} images only")

    # ── Resume ─────────────────────────────────────────────────────────────────
    done: set[str] = set()
    results: list[dict] = []
    out_path = Path(args.output)
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_path)
            results  = existing.to_dict("records")
            done     = set(existing["image_path"].tolist())
            print(f"  Resuming        : {len(done)} already done, "
                  f"{len(all_images) - len(done)} remaining")
        except Exception:
            print("  Resume          : existing file unreadable — starting fresh")
    else:
        print(f"  Output          : {args.output}  (new file)")

    remaining = [x for x in all_images if x[0] not in done]
    total     = len(all_images)
    todo      = len(remaining)

    if todo == 0:
        print("\n  Nothing to do — all images already processed.\n")
        return

    print(f"\n  Starting generation — {todo} images to process\n")
    print(f"  {'#':>5}  {'pct':>4}  {'elapsed':>7}  {'ETA':>7}  "
          f"{'split/wf':<14}  {'file':<40}  status")
    print(f"  {'-'*5}  {'-'*4}  {'-'*7}  {'-'*7}  "
          f"{'-'*14}  {'-'*40}  ------")

    failed        = 0
    nofeat_count  = 0
    processed     = 0
    initial_done  = len(done)
    start_time    = time.time()

    for png_path, wf, split in remaining:
        feat = find_features(png_path, feat_index)
        if feat is None:
            nofeat_count += 1
            feat = {}

        feat_text = build_feature_text(wf, feat)
        corr_vec  = compute_correction_vector(wf, feat)
        corr_str  = format_correction_vector(corr_vec)
        feat_tag  = "" if feat else " [no feat]"

        processed += 1
        done_total = initial_done + processed
        pct        = done_total / total * 100
        elapsed    = time.time() - start_time
        rate       = processed / elapsed if elapsed > 0 else 0
        eta        = (todo - processed) / rate if rate > 0 else 0

        label = Path(png_path).name[:40].ljust(40)
        prefix = (f"  {done_total:>5}  {pct:>3.0f}%  "
                  f"{_fmt_duration(elapsed):>7}  {_fmt_duration(eta):>7}  "
                  f"{split+'/'+wf:<14}  {label}  ")

        print(prefix, end="", flush=True)

        description = call_gemini(api_key, png_path, feat_text)
        if description is None:
            failed += 1
            print(f"FAILED{feat_tag}")
            continue

        full_output = description.strip() + "\n\n" + corr_str
        results.append({
            "image_path":             png_path,
            "waveform_label":         wf,
            "split":                  split,
            "features_json":          json.dumps({k: v for k, v in feat.items()
                                                  if k not in ("filename", "waveform")}),
            "correction_vector_json": json.dumps(corr_vec),
            "description":            full_output,
        })
        done.add(png_path)
        print(f"OK  {len(full_output)} chars{feat_tag}")

        time.sleep(4)   # free tier: 15 req/min

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(args.output, index=False)
            elapsed_ck = time.time() - start_time
            print(f"\n  ── checkpoint: {len(results)} total saved  "
                  f"({_fmt_duration(elapsed_ck)} elapsed) ──\n")

    pd.DataFrame(results).to_csv(args.output, index=False)

    total_time   = time.time() - start_time
    success      = processed - failed
    success_rate = success / processed * 100 if processed > 0 else 0

    print(f"\n  {'='*62}")
    print(f"  Done!")
    print(f"    Total saved   : {len(results)} samples  →  {args.output}")
    print(f"    Processed     : {processed}  (success {success_rate:.1f}%,  "
          f"failed {failed})")
    print(f"    No features   : {nofeat_count}")
    print(f"    Total time    : {_fmt_duration(total_time)}")
    print(f"    Avg rate      : {processed/total_time*60:.1f} images/min")
    print(f"  {'='*62}\n")


if __name__ == "__main__":
    main()
