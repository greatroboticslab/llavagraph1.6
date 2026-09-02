"""
build_pairs.py
================
Builds the (command, real-response) training pairs for T2, the Mamba
learned forward twin.

Data-availability constraint this works around: the real hardware CSVs
(Time_ms, Absolute_Displacement_nm) record the MEASURED response only --
there is no synchronized recorded voltage channel. So the "command"
signal u(t) used here is the NOMINAL command reconstructed from the
experiment's known settings (waveform type + the measured dominant
frequency + duty cycle where applicable), normalized to unit amplitude,
not a hardware-logged voltage trace. This is standard practice for
system identification from single-channel response data (you trust the
AWG did what it was told), and it's already implicitly what
fopdt_features()'s v_drive_est_v does elsewhere in this codebase (it
also has to *infer* drive voltage rather than read it). Using a
normalized (unit-amplitude) command is a deliberate simplification: the
model has to learn the effective gain (K) itself from the data, rather
than depend on an uncertain voltage estimate.

noise is excluded: "command -> response" isn't a meaningful framing for
a waveform whose command IS randomness -- there's nothing to hand the
model as u(t) that would be more than a different noise realization.

Split: plain stratified-by-waveform split, NOT group-based. Unlike
mamba2_fast's training_data_short.csv (2445 rows = 489 measurements x 5
augmented copies each, which is why THAT split has to be group-aware),
Mamballm2/features_full.csv has exactly one row per physical
measurement -- no augmentation-created siblings to leak across splits.
"""

import json
from pathlib import Path

import numpy as np
from scipy import signal as sps

from feature_extract import real_measurements, window_slice
import feature_extract as fe
from signal_align import align_phase

FS_OUT = 1000.0
WINDOW_S = 0.26  # matches the 260ms analysis window used everywhere else
N_SAMPLES = int(WINDOW_S * FS_OUT)

_INCLUDED_WAVEFORMS = ["sine", "square", "ramp", "pulse"]  # noise excluded, see module docstring


def _nominal_command(waveform: str, freq_hz: float, duty: float, n: int, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    if waveform == "sine":
        return np.sin(2 * np.pi * freq_hz * t)
    if waveform == "square":
        return sps.square(2 * np.pi * freq_hz * t, duty=duty)
    if waveform == "ramp":
        return sps.sawtooth(2 * np.pi * freq_hz * t, width=1.0)
    if waveform == "pulse":
        sq = sps.square(2 * np.pi * freq_hz * t, duty=duty)
        return (sq + 1.0) / 2.0
    raise ValueError(waveform)


def build_dataset(seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    examples = []  # each: dict(waveform, freq_hz, u, y, filename)

    for waveform in _INCLUDED_WAVEFORMS:
        df = real_measurements(waveform)
        df = df[df["path"].notna()]
        for row in df.itertuples():
            sig, win = fe.bfe.load_signal(str(row.path))
            if len(win) < N_SAMPLES // 2:
                continue  # too short to be usable
            y = win[:N_SAMPLES]
            if len(y) < N_SAMPLES:
                y = np.pad(y, (0, N_SAMPLES - len(y)), mode="edge")

            duty = 0.5
            feat = fe.extract_features(sig, win, waveform)
            if waveform == "square" and feat.get("square_duty_cycle") is not None \
                    and not np.isnan(feat["square_duty_cycle"]):
                duty = float(np.clip(feat["square_duty_cycle"], 0.05, 0.95))
            elif waveform == "pulse" and feat.get("pulse_duty_cycle") is not None \
                    and not np.isnan(feat["pulse_duty_cycle"]):
                duty = float(np.clip(feat["pulse_duty_cycle"], 0.05, 0.95))

            freq_hz = float(row.dominant_freq_hz) if row.dominant_freq_hz and row.dominant_freq_hz > 0 else 1.0
            u = _nominal_command(waveform, freq_hz, duty, N_SAMPLES, FS_OUT)
            u = align_phase(u, y, freq_hz, FS_OUT)

            examples.append({
                "waveform": waveform, "freq_hz": freq_hz, "filename": row.filename,
                "u": u.astype(np.float32), "y": y.astype(np.float32),
            })

    # stratified split by waveform type
    by_wf = {}
    for ex in examples:
        by_wf.setdefault(ex["waveform"], []).append(ex)

    splits = {"train": [], "val": [], "test": []}
    for wf, items in by_wf.items():
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_frac))
        n_test = max(1, int(len(items) * test_frac))
        val_idx = set(idx[:n_val])
        test_idx = set(idx[n_val:n_val + n_test])
        for i, ex in enumerate(items):
            split = "val" if i in val_idx else ("test" if i in test_idx else "train")
            splits[split].append(ex)

    return splits


def save_dataset(splits: dict, out_dir: str = "pairs_data"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    meta = {}
    for split_name, items in splits.items():
        U = np.stack([e["u"] for e in items])
        Y = np.stack([e["y"] for e in items])
        np.savez(out / f"{split_name}.npz", u=U, y=Y)
        meta[split_name] = [
            {"waveform": e["waveform"], "freq_hz": e["freq_hz"], "filename": e["filename"]}
            for e in items
        ]
        counts = {wf: sum(1 for e in items if e["waveform"] == wf) for wf in _INCLUDED_WAVEFORMS}
        counts_str = ", ".join(f"{wf}={n}" for wf, n in counts.items())
        print(f"{split_name}: {len(items)} examples ({counts_str})")
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    splits = build_dataset()
    save_dataset(splits)
