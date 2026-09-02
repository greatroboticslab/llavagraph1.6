"""
signal_align.py
================
Shared phase-alignment helper. Used by both build_pairs.py (aligning a
reconstructed command against a real measurement) and
build_pairs_residual.py (aligning a T0-simulated waveform against a real
measurement) -- same underlying problem both times: the real measurement
window (feature_extract.window_slice()) is a fixed-offset slice of a
longer recording, starting at an arbitrary phase of the commanded cycle,
not at the synthetic signal's t=0.
"""

import numpy as np


def align_phase(candidate: np.ndarray, target: np.ndarray, freq_hz: float, fs: float) -> np.ndarray:
    """Shift `candidate` (rolling, so no data is discarded) by whichever
    offset within one period best cross-correlates it with `target`."""
    period = max(2, int(round(fs / max(freq_hz, 1e-6))))
    period = min(period, len(candidate))
    t_n = (target - target.mean()) / (target.std() + 1e-9)
    best_shift, best_score = 0, -np.inf
    for shift in range(period):
        c_shifted = np.roll(candidate, shift)
        c_n = (c_shifted - c_shifted.mean()) / (c_shifted.std() + 1e-9)
        score = float(np.dot(c_n, t_n))
        if score > best_score:
            best_score, best_shift = score, shift
    return np.roll(candidate, best_shift)
