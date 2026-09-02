"""
stage2_calibrate_shared.py
============================
Stage 2 of the physics-deepening plan: fit ONE shared parameter set
(K, tau, Prandtl-Ishlinskii hysteresis thresholds/weights, saturation)
across sine/square/ramp/pulse real measurements simultaneously, using
only the TRAIN split (stage_split.json) so the fitted model can still be
evaluated on a genuinely held-out test set later.

Physical rationale for sharing (not fitting per waveform type, as the
original T1 did): K, tau, and hysteresis are properties of the PZT
device itself, not of which command shape you happen to be driving it
with. A single shared fit that works reasonably across all four waveform
types is much stronger evidence of getting the physics right than four
independently-tuned parameter sets, which could each overfit their own
type without describing anything physically unified.

This is a long-running optimization (9 parameters x N examples x
multiple types) -- expect it to take minutes, not seconds.

Usage:
    python3 make_split.py              # once
    python3 stage2_calibrate_shared.py
"""

import json

import numpy as np

from physics_model import TwinParams
from feature_extract import extract_features, real_feature_stats, real_measurements
import feature_extract as fe
from calibrate import calibrate_shared

WAVEFORMS = ["sine", "square", "ramp", "pulse"]
N_PER_TYPE = 10  # real measurements per waveform type used in the shared fit


def main():
    split = json.load(open("stage_split.json"))
    train_items = split["train"]

    rng = np.random.default_rng(0)
    stats = real_feature_stats()

    def make_std_lookup(waveform):
        def f(k):
            try:
                return stats.loc[waveform, (k, "std")]
            except KeyError:
                return None
        return f

    std_lookup_by_waveform = {wf: make_std_lookup(wf) for wf in WAVEFORMS}
    base_params_by_waveform = {wf: TwinParams(waveform=wf) for wf in WAVEFORMS}

    real_batch = []
    for wf in WAVEFORMS:
        items = [it for it in train_items if it["waveform"] == wf]
        chosen = rng.choice(len(items), size=min(N_PER_TYPE, len(items)), replace=False)
        real_df = real_measurements(wf)
        for i in chosen:
            m = items[i]
            path = real_df.loc[real_df["filename"] == m["filename"], "path"].iloc[0]
            sig, win = fe.bfe.load_signal(str(path))
            feat = extract_features(sig, win, wf)
            real_batch.append((wf, m["freq_hz"], feat))

    print(f"Calibrating shared parameters on {len(real_batch)} real measurements "
          f"({N_PER_TYPE} per waveform type x {len(WAVEFORMS)} types)...")

    fitted, gap_before, gap_after, history = calibrate_shared(
        base_params_by_waveform, real_batch, std_lookup_by_waveform,
        calib_duration_s=0.15, maxfev=500, verbose=True)

    print(f"\ngap before: {gap_before:.3f}  ->  gap after: {gap_after:.3f}")
    print("fitted shared parameters:")
    for k, v in fitted.items():
        print(f"  {k}: {v:.5g}")

    with open("stage2_fitted_params.json", "w") as f:
        json.dump({
            "fitted": fitted, "gap_before": gap_before, "gap_after": gap_after,
            "n_per_type": N_PER_TYPE, "waveforms": WAVEFORMS,
        }, f, indent=2)
    print("\nsaved stage2_fitted_params.json")


if __name__ == "__main__":
    main()
