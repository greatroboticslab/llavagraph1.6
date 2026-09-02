"""
audit_raw_data.py
===================
Stage 0 of the physics-deepening plan: flag real measurement files whose
raw displacement trace is physically impossible for this device, before
any further physics fitting happens downstream.

Two statistical approaches were tried first and both failed, which is
worth recording rather than hiding:

  - Cross-file MAD-based robust z-score on max single-sample jump: broke
    down because many files (especially low-frequency square/pulse
    recordings) have >50% of their sample-to-sample jumps at the ADC's
    quantization floor, making the median absolute deviation exactly 0
    for that waveform type and blowing the z-score up to meaningless
    values for EVERY file, including the genuine outlier.
  - Intra-file ratio (own max jump / own median jump): broke down for
    the same underlying reason -- a slow square/pulse wave spends most
    of its 1kHz-sampled duration flat between edges, so the median jump
    is legitimately 0, making the ratio blow up for ordinary, correct
    step-edge recordings, not just glitches.

What actually works, and is simpler: a hard, physically-derived ceiling
on peak absolute displacement. This device is a 25mm-diameter,
0.5mm-thick disc PZT with calibrated gain K~380 nm/V (see
mamballm/batch_feature_extraction.py's FOPDT_K_NM_PER_V) driven by a
Moku:Go AWG at up to +/-5V -- so even generously allowing for
saturation/hysteresis overshoot, no real measurement should exceed
roughly 10 microns (10,000 nm). Checking the full real dataset confirms
this cleanly separates one genuine glitch from everything else: the
single highest peak amplitude across all 489 measurements is
Pulse-9_absolute.csv at 456,213 nm (a physically impossible >450
micron jump -- confirmed by inspecting its raw trace directly, not
just this statistic), while the second-highest across the ENTIRE rest
of the dataset is 2,570 nm -- almost three orders of magnitude lower,
and well within plausible range. There is no ambiguous middle ground:
any ceiling between ~3,000nm and ~450,000nm gives the identical result.
"""

import json

import numpy as np
import pandas as pd

from feature_extract import real_measurements
import feature_extract as fe

WAVEFORMS = ["sine", "square", "ramp", "pulse", "noise"]
PEAK_CEILING_NM = 10_000.0  # 10 microns; see module docstring for derivation


def main():
    rows = []
    for waveform in WAVEFORMS:
        df = real_measurements(waveform)
        df = df[df["path"].notna()]
        for row in df.itertuples():
            sig, _ = fe.bfe.load_signal(str(row.path))
            peak_abs = float(np.max(np.abs(sig)))
            rows.append({"waveform": waveform, "filename": row.filename, "peak_abs_nm": peak_abs})

    audit = pd.DataFrame(rows)
    audit["flagged"] = audit["peak_abs_nm"] > PEAK_CEILING_NM
    audit.to_csv("raw_data_audit.csv", index=False)

    flagged = audit[audit["flagged"]].sort_values("peak_abs_nm", ascending=False)
    print(f"Audited {len(audit)} real measurements across {len(WAVEFORMS)} waveform types.")
    print(f"Physical ceiling: {PEAK_CEILING_NM:,.0f} nm (see module docstring for derivation).")
    print(f"Flagged {len(flagged)} measurement(s) as physically impossible:\n")
    if len(flagged):
        print(flagged[["waveform", "filename", "peak_abs_nm"]].to_string(index=False))
    else:
        print("(none)")

    runner_up = audit[~audit["flagged"]]["peak_abs_nm"].max()
    print(f"\nFor context, the highest peak amplitude among NOT-flagged measurements "
          f"is {runner_up:,.1f} nm -- {PEAK_CEILING_NM / runner_up:.1f}x below the ceiling, "
          f"confirming the ceiling isn't cutting close to any legitimate measurement.")

    with open("artifact_measurements.json", "w") as f:
        json.dump(sorted(flagged["filename"].tolist()), f, indent=2)
    print("\nSaved full audit to raw_data_audit.csv, flagged list to artifact_measurements.json")


if __name__ == "__main__":
    main()
