"""
make_split.py
==============
Canonical train/val/test split for all downstream Stage 1+ scripts
(stage2_calibrate_shared.py, build_pairs_residual.py, ...), built fresh
from the CURRENT (Stage-0-cleaned) real measurement set. Replaces the
earlier ad hoc arrangement where build_pairs.py generated a split and
everything else silently reused its (now stale, pre-cleaning,
Pulse-9-contaminated) meta.json.

Stratified by waveform type, not group-based: Mamballm2/features_full.csv
has one row per physical measurement (no augmentation-created siblings
to leak across splits, unlike mamba2_fast's training_data_short.csv --
see feature_extract.py's docstring).
"""

import json

import numpy as np

from feature_extract import real_measurements

WAVEFORMS = ["sine", "square", "ramp", "pulse"]  # noise excluded, see build_pairs*.py docstrings


def make_split(seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for wf in WAVEFORMS:
        df = real_measurements(wf)
        df = df[df["path"].notna()]
        items = [{"waveform": wf, "freq_hz": float(r.dominant_freq_hz), "filename": r.filename}
                 for r in df.itertuples()]
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_frac))
        n_test = max(1, int(len(items) * test_frac))
        val_idx = set(idx[:n_val])
        test_idx = set(idx[n_val:n_val + n_test])
        for i, it in enumerate(items):
            split = "val" if i in val_idx else ("test" if i in test_idx else "train")
            splits[split].append(it)
    return splits


if __name__ == "__main__":
    splits = make_split()
    for name, items in splits.items():
        counts = {wf: sum(1 for it in items if it["waveform"] == wf) for wf in WAVEFORMS}
        print(f"{name}: {len(items)} ({', '.join(f'{k}={v}' for k, v in counts.items())})")
    with open("stage_split.json", "w") as f:
        json.dump(splits, f, indent=2)
    print("saved stage_split.json")
