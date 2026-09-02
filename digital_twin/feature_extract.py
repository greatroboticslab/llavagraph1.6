"""
feature_extract.py
===================
Real-time physical-property monitor for the digital twin.

Deliberately does NOT reimplement feature formulas. It imports
mamballm/batch_feature_extraction.py directly and calls the same
functions used to build features_full.csv from real hardware
recordings. This is the load-bearing design choice: if simulated and
real features were computed by two separately-written formulas, any
sim-vs-real comparison would be partly measuring "did I write the two
formulas the same way" rather than "does the physics model match the
real device". Importing the same module for both removes that risk.

Also loads the real feature distribution (Mamballm2/features_full.csv)
so simulated features can be shown next to the real per-waveform
mean +/- std, as a running sanity check on whether the physics-informed
simulator is actually landing in physically realistic territory.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAMBALLM_DIR = _REPO_ROOT / "mamballm"
_REAL_FEATURES_CSV = _REPO_ROOT / "Mamballm2" / "features_full.csv"

if str(_MAMBALLM_DIR) not in sys.path:
    sys.path.insert(0, str(_MAMBALLM_DIR))

import batch_feature_extraction as bfe  # noqa: E402  (path set up above)

FS = bfe.FS
assert FS == 1000.0, "digital_twin/physics_model.py FS_OUT must match batch_feature_extraction.FS"

# Mamballm2/features_full.csv has 493 raw rows, but the leakage audit
# (newpaper_draft/docs/github_data_audit_openloop.md,
# submission_truth_audit_20260820.md) found these 4 pulse rows unused /
# not part of the leakage-safe classifier data (mamba2_fast/
# training_data_short.csv) -- the paper's official, audited count is 489
# usable open-loop measurements, not 493. Excluded here everywhere so the
# demo's "real" numbers match what the paper is allowed to claim.
_EXCLUDED_MEASUREMENTS = {
    "Pulse-1_absolute.csv", "Pulse-2_absolute.csv",
    "Pulse-3_absolute.csv", "Pulse-29_absolute.csv",
}

# Separate provenance from the leakage-audit exclusions above: these are
# raw measurements flagged by audit_raw_data.py as physically-impossible
# sensor/interferometer glitches (e.g. Pulse-9_absolute.csv, whose peak
# displacement of ~456,000 nm cannot be real for this 25mm/0.5mm disc
# PZT -- confirmed both by inspecting its raw trace and by the
# audit's physically-derived amplitude ceiling). Loaded from the audit's
# own output so this list stays in sync with re-runs of that script;
# falls back to empty if the audit hasn't been run yet in this checkout.
_THIS_DIR = Path(__file__).resolve().parent
_artifact_json = _THIS_DIR / "artifact_measurements.json"
if _artifact_json.exists():
    with open(_artifact_json) as _f:
        _ARTIFACT_MEASUREMENTS = set(json.load(_f))
else:
    _ARTIFACT_MEASUREMENTS = set()

# The 493 real measurements' raw per-sample time-series CSVs are not all
# in one folder on this machine -- they're split across these locations.
# Indexed once, by filename, across all of them.
_RAW_DATA_ROOTS = [
    _REPO_ROOT / "data" / "output_file" / "csv",
    _REPO_ROOT / "data" / "Archive",
    Path("/Users/ilminurablikim/Desktop/Lab/mamballm/data_csv"),
]

_RAW_FILE_INDEX_CACHE = None


def _raw_file_index() -> dict:
    global _RAW_FILE_INDEX_CACHE
    if _RAW_FILE_INDEX_CACHE is None:
        index = {}
        for root in _RAW_DATA_ROOTS:
            if not root.is_dir():
                continue
            for p in root.rglob("*.csv"):
                index.setdefault(p.name, p)
        _RAW_FILE_INDEX_CACHE = index
    return _RAW_FILE_INDEX_CACHE


def real_measurements(waveform: str) -> pd.DataFrame:
    """The audited-usable real measurements for one waveform type:
    filename, its dominant_freq_hz (reused from features_full.csv, not
    recomputed -- same fft_features() call that would be run on the raw
    file anyway), and the resolved local file path if found. Used to
    frequency-match a real measurement against the simulator's current
    command frequency, so waveform overlays compare like with like."""
    df = pd.read_csv(_REAL_FEATURES_CSV)
    df = df[df["waveform"] == waveform]
    df = df[~df["filename"].isin(_EXCLUDED_MEASUREMENTS | _ARTIFACT_MEASUREMENTS)]
    idx = _raw_file_index()
    df = df.copy()
    df["path"] = df["filename"].map(lambda f: idx.get(f))
    return df[["filename", "dominant_freq_hz", "path"]].reset_index(drop=True)


def extract_features(signal_full: np.ndarray, window: np.ndarray, waveform: str) -> dict:
    """Same computation as batch_feature_extraction.process_file(), minus
    the CSV/file-loading step -- takes arrays (as produced by
    physics_model.simulate(), or loaded from a real measurement CSV)
    directly."""
    fft_feat = bfe.fft_features(signal_full)
    if fft_feat is None:
        return {}

    td_feat = bfe.time_domain_features(signal_full)
    dom_freq = fft_feat["dominant_freq_hz"]
    peak_nm = td_feat.get("peak_nm", float("nan"))

    if waveform == "sine":
        specific = bfe.sine_specific_features(signal_full, dom_freq)
    elif waveform == "square":
        specific = bfe.square_specific_features(signal_full)
    elif waveform == "noise":
        specific = bfe.noise_specific_features(signal_full)
    elif waveform == "ramp":
        specific = bfe.ramp_specific_features(signal_full)
    elif waveform == "pulse":
        specific = bfe.pulse_specific_features(signal_full)
    else:
        specific = {}

    q_feat = bfe.quarter_features(window, waveform, dom_freq)
    fopdt_feat = bfe.fopdt_features(dom_freq, peak_nm)
    hyst_feat = bfe.hysteresis_proxy(window, dom_freq) if waveform == "sine" else {}

    row = {"waveform": waveform, "n_samples": len(signal_full)}
    row.update(fft_feat)
    row.update(td_feat)
    row.update(specific)
    row.update(q_feat)
    row.update(fopdt_feat)
    row.update(hyst_feat)
    return row


def window_slice(signal: np.ndarray, fs: float = FS,
                  start_ms: float = None, end_ms: float = None) -> np.ndarray:
    """Match the same 480-740ms analysis window batch_feature_extraction.py
    uses for real recordings, but scaled to whatever length `signal` has
    (the simulator's default duration is shorter than a full real
    recording, so this takes a proportional slice by default)."""
    n = len(signal)
    if start_ms is None or end_ms is None:
        # proportional fallback: middle 260ms-equivalent fraction (matches
        # the real window's 260ms out of a typical ~1s+ recording)
        frac = 260.0 / 1000.0
        s = int(n * 0.1)
        e = min(n, s + int(fs * frac))
        window = signal[s:e]
    else:
        s = int(start_ms / 1000.0 * fs)
        e = int(end_ms / 1000.0 * fs)
        window = signal[s:e]
    return window - np.mean(window) if len(window) else window


_REAL_STATS_CACHE = None


def real_feature_stats() -> pd.DataFrame:
    """Per-waveform mean/std of every numeric column in the real,
    audited-usable 489-measurement feature table (Mamballm2/
    features_full.csv minus the 4 excluded pulse rows) -- the reference
    distribution simulated features get compared against."""
    global _REAL_STATS_CACHE
    if _REAL_STATS_CACHE is None:
        df = pd.read_csv(_REAL_FEATURES_CSV)
        df = df[~df["filename"].isin(_EXCLUDED_MEASUREMENTS | _ARTIFACT_MEASUREMENTS)]
        numeric = df.select_dtypes(include=[np.number])
        numeric["waveform"] = df["waveform"]
        _REAL_STATS_CACHE = numeric.groupby("waveform").agg(["mean", "std"])
    return _REAL_STATS_CACHE


def real_measurement_count() -> int:
    df = pd.read_csv(_REAL_FEATURES_CSV)
    return int((~df["filename"].isin(_EXCLUDED_MEASUREMENTS | _ARTIFACT_MEASUREMENTS)).sum())
