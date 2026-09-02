"""
stage4_probe_corrector.py
===========================
Stage 4 of the physics-deepening plan: treat the trained Mamba corrector
as an unknown system and characterize it the same way you'd characterize
the real device -- feed it controlled, in-distribution synthetic inputs
(T1-v2 simulations, not real data) and look at what the correction
itself does, as a function of frequency and around step transitions.

Two probes:

1. Frequency sweep (sine, T1-v2, 10-450 Hz): for each frequency, measure
   the correction's own gain (RMS of the correction / RMS of the ideal
   signal) and its phase shift relative to the ideal (via cross-
   correlation lag). A resonance the physics model doesn't capture would
   show up as a GAIN PEAK at some frequency, with a phase transition
   around it -- the classic second-order-system signature. A flat gain
   and phase across frequency would instead suggest the correction is
   doing something closer to a fixed, frequency-independent adjustment
   (e.g. fixing a static amplitude/offset error), not compensating for
   an unmodeled dynamic mode.

2. Step transients (pulse, T1-v2, several frequencies): visual check of
   the correction's shape right at each step edge -- oscillatory
   (ringing, i.e. resonance-like) vs. smooth/monotonic vs. unstructured.

Both probes are run on SIMULATED (T1-v2) inputs only, matching what the
corrector actually saw during training (real per-sample paired command
data isn't available -- see build_pairs_residual.py docstring) -- this
is deliberately an in-distribution probe of the trained function itself,
not a new claim about the real device.
"""

import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from mamba_twin_model import MambaTwin
from physics_model import TwinParams, simulate
from signal_align import align_phase

CKPT_PATH = "mamba_correction.pt"
WINDOW_S = 0.26
N_SAMPLES = int(WINDOW_S * 1000.0)


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = MambaTwin(d_model=ckpt["d_model"], d_state=ckpt["d_state"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["y_scale"]


def base_params(waveform):
    fitted = json.load(open("stage2_fitted_params.json"))["fitted"]
    return TwinParams(waveform=waveform, seed=0, **fitted)


def correction_for(model, y_scale, waveform, freq_hz):
    p = base_params(waveform)
    p.freq_hz = freq_hz
    p.duration_s = WINDOW_S
    r = simulate(p)
    y_ideal = r["y_nm"][:N_SAMPLES]
    if len(y_ideal) < N_SAMPLES:
        y_ideal = np.pad(y_ideal, (0, N_SAMPLES - len(y_ideal)), mode="edge")
    with torch.no_grad():
        u_t = torch.from_numpy(y_ideal / y_scale).float().unsqueeze(0)
        delta = model(u_t).numpy()[0] * y_scale
    return y_ideal, delta


def _fit_at_freq(signal: np.ndarray, freq_hz: float, fs: float = 1000.0):
    """Amplitude and phase of `signal` at EXACTLY freq_hz, via a
    least-squares sin/cos fit (same method sine_specific_features() in
    batch_feature_extraction.py uses for sine_phase_lag_deg) -- not an
    FFT bin (which would only land on freq_hz for specific window
    lengths) and not a discrete-shift search (see module docstring:
    that approach's resolution is limited to 360/samples_per_period
    degrees, which is unusably coarse above ~100Hz at 1kHz sampling --
    e.g. only 2-3 samples/period at 420Hz, so a shift-search could only
    ever report phase in ~120-180 degree steps, which is why the
    original version of this function flatlined to exactly 0 degrees
    for every frequency above 100Hz: not a real finding, a resolution
    floor). This algebraic fit has no such resolution limit -- it only
    needs enough total samples for a well-conditioned 2-parameter
    regression (trivially satisfied by all 260 samples here, regardless
    of how many fall within one cycle)."""
    n = len(signal)
    t = np.arange(n) / fs
    A = np.column_stack([np.sin(2 * np.pi * freq_hz * t), np.cos(2 * np.pi * freq_hz * t)])
    coeffs, *_ = np.linalg.lstsq(A, signal, rcond=None)
    amp = float(np.hypot(coeffs[0], coeffs[1]))
    phase_deg = float(np.degrees(np.arctan2(coeffs[1], coeffs[0])))
    return amp, phase_deg


def freq_sweep(model, y_scale):
    freqs = np.concatenate([np.arange(10, 100, 10), np.arange(100, 460, 40)])
    gains, phases, fund_gains = [], [], []
    for f in freqs:
        y_ideal, delta = correction_for(model, y_scale, "sine", f)
        rms_ideal = np.sqrt(np.mean(y_ideal ** 2)) + 1e-9
        rms_delta = np.sqrt(np.mean(delta ** 2))
        gains.append(rms_delta / rms_ideal)

        amp_ideal, phase_ideal = _fit_at_freq(y_ideal, f)
        amp_delta, phase_delta = _fit_at_freq(delta, f)
        fund_gains.append(amp_delta / (amp_ideal + 1e-9))
        phase_diff = (phase_delta - phase_ideal + 180.0) % 360.0 - 180.0
        phases.append(phase_diff)
    return freqs, np.array(gains), np.array(phases), np.array(fund_gains)


def main():
    model, y_scale = load_model()

    print("=== Frequency sweep: correction's own gain and phase vs. frequency (sine, T1-v2) ===")
    print("(phase now via least-squares sin/cos fit at the exact drive frequency -- see "
          "_fit_at_freq() docstring for why the old discrete-shift-search version was unreliable "
          "above ~100Hz)")
    freqs, gains, phases, fund_gains = freq_sweep(model, y_scale)
    for f, g, ph, fg in zip(freqs, gains, phases, fund_gains):
        print(f"  {f:5.0f} Hz   RMS ratio = {g:.4f}   fundamental-only ratio = {fg:.4f}   "
              f"phase = {ph:7.1f} deg")

    peak_idx = int(np.argmax(gains))
    print(f"\nPeak correction gain at {freqs[peak_idx]:.0f} Hz "
          f"(RMS ratio {gains[peak_idx]:.4f}); "
          f"gain range [{gains.min():.4f}, {gains.max():.4f}], "
          f"ratio max/min = {gains.max()/max(gains.min(),1e-9):.2f}x")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(freqs, gains, "o-", label="RMS ratio (broadband)")
    ax1.plot(freqs, fund_gains, "s--", label="fundamental-only ratio", alpha=0.7)
    ax1.set_ylabel("correction / ideal amplitude ratio")
    ax1.set_title("Correction's own frequency response (probe, sine T1-v2 inputs)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(freqs, phases, "o-", color="tab:orange")
    ax2.set_ylabel("phase of correction rel. to ideal (deg)")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylim(-190, 190)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("corrector_freq_response.png", dpi=130)
    print("saved corrector_freq_response.png")

    print("\n=== Step transients: pulse, T1-v2, a few frequencies ===")
    fig2, axes = plt.subplots(3, 1, figsize=(8, 7))
    for ax, f in zip(axes, [2.0, 20.0, 100.0]):
        y_ideal, delta = correction_for(model, y_scale, "pulse", f)
        t_ms = np.arange(len(y_ideal)) / 1000.0 * 1000
        ax.plot(t_ms, y_ideal - y_ideal.mean(), label="ideal (T1-v2)", lw=1, alpha=0.7)
        ax.plot(t_ms, delta, label="correction (delta_y)", lw=1.3, color="tab:red")
        ax.set_title(f"pulse @ {f:.0f} Hz")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("nm")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("corrector_step_transients.png", dpi=130)
    print("saved corrector_step_transients.png")


if __name__ == "__main__":
    main()
