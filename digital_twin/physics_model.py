"""
physics_model.py
=================
Forward physical simulator ("digital twin" v1) for the open-loop PZT
actuator described in draft_v2.tex.

Model structure (Hammerstein form, standard in PZT modeling literature —
Adriaens, de Koning & Banning 2000, "Modeling Piezoelectric Actuators",
already cited in draft_v2.tex as adriaens2000modeling):

    command voltage u(t)
        -> hysteresis   (Prandtl-Ishlinskii: weighted sum of 3 play
                          operators -- see _prandtl_ishlinskii() docstring)
        -> FOPDT linear dynamics           (same eq. as draft_v2.tex Sec 3,
                                             zero-order-hold discretisation,
                                             Eq. (zoh) in the paper)
        -> soft amplitude saturation       (dielectric/mechanical limit)
        -> additive measurement noise
        -> displacement y(t)  [nm]

Stage 1 revision note: the hysteresis stage used to be a single play
operator with a fixed width, which (a) cannot produce amplitude-dependent
hysteresis by construction -- real sine data shows hysteresis_nm
correlating with drive amplitude at r=0.59 within a controlled frequency
band, which a single fixed-width operator has no way to reproduce -- and
(b) had a genuine discretization bug: its window was centered on the
PREVIOUS OUTPUT (`clip(u[n], y[n-1]-w, y[n-1]+w)`), which converges to a
no-op as the simulation timestep gets finer (verified: at FS_SIM=200kHz
the old operator was numerically indistinguishable from the identity
function for realistic command frequencies). The corrected play operator
below centers the window on the CURRENT INPUT instead
(`max(u[n]-w, min(u[n]+w, y[n-1]))`), which is the standard,
discretization-independent formulation and was verified to produce a
constant 2w loop height regardless of simulation timestep or amplitude
(as expected for an ideal play operator). Summing three of these at
different thresholds (Prandtl-Ishlinskii) was verified separately to
reproduce the empirically-observed amplitude-dependence: loop height
grows as amplitude crosses each threshold, then saturates -- exactly the
"grows then plateaus" shape a global linear fit of hysteresis-vs-amplitude
failed to capture (R^2=0.024) but a controlled-frequency-band correlation
did detect (r=0.59).

K/tau/theta defaults are copied from mamballm/batch_feature_extraction.py
(FOPDT_K_NM_PER_V, FOPDT_TAU_S, FOPDT_THETA_S) — those were calibrated
against the real device (same 25mm/0.5mm disc PZT, same interferometer
setup). Using the same numbers here means "default simulator" and
"what the real-data pipeline calls typical" are the same claim, not two
independently guessed ones.

Internally simulated at FS_SIM (200 kHz) so tau=250us and theta=5us are
both resolved with multiple samples (a 1 kHz step would give dt/tau=4,
far too coarse for a stable discrete first-order recursion), then
resampled down to FS_OUT=1000 Hz to match the real hardware's ADC rate
(SG-uMD2, 1 kHz) — the same FS batch_feature_extraction.py assumes.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

FS_SIM = 200_000.0  # internal simulation rate, Hz (resolves theta=5us as 1 sample)
FS_OUT = 1_000.0    # output rate, Hz -- must match batch_feature_extraction.FS

# Calibrated FOPDT constants, copied from mamballm/batch_feature_extraction.py
# so simulator defaults and the real-data feature pipeline agree.
DEFAULT_K_NM_PER_V = 380.0
DEFAULT_TAU_S = 0.000250
DEFAULT_THETA_S = 0.000005

WAVEFORM_TYPES = ["sine", "square", "ramp", "pulse", "noise"]


@dataclass
class TwinParams:
    """Every physically-interpretable knob the simulator exposes."""

    waveform: str = "sine"
    freq_hz: float = 100.0
    amp_v: float = 1.0            # commanded drive voltage amplitude
    duration_s: float = 0.3       # matches the paper's 260ms analysis window + margin
    duty: float = 0.5             # square/pulse duty cycle

    # Linear dynamics (FOPDT) -- see draft_v2.tex Sec 3, Eq. (fopdt_step)
    K_nm_per_v: float = DEFAULT_K_NM_PER_V
    tau_s: float = DEFAULT_TAU_S
    theta_s: float = DEFAULT_THETA_S

    # Nonlinearities -- hysteresis is a Prandtl-Ishlinskii sum of 3
    # play operators (thresholds in volts, command-side; weights are
    # normalized to sum to 1 inside _prandtl_ishlinskii(), so K alone
    # keeps its meaning as the overall gain -- the weights only shape
    # how hysteresis grows with amplitude, not the overall output scale).
    # Defaults verified (synthetically, not yet against real data) to
    # reproduce a "grows then saturates with amplitude" loop shape.
    hyst_r1_v: float = 0.02
    hyst_r2_v: float = 0.08
    hyst_r3_v: float = 0.25
    hyst_w1: float = 0.5
    hyst_w2: float = 0.3
    hyst_w3: float = 0.2
    sat_nm: float = 2000.0        # soft saturation limit; large = effectively linear
    noise_nm: float = 3.0         # additive measurement noise, RMS nm

    seed: int = None


def _command_signal(p: TwinParams, t: np.ndarray) -> np.ndarray:
    w = p.waveform
    if w == "sine":
        return p.amp_v * np.sin(2 * np.pi * p.freq_hz * t)
    if w == "square":
        return p.amp_v * sps.square(2 * np.pi * p.freq_hz * t, duty=p.duty)
    if w == "ramp":
        # width=1.0 -> rising sawtooth (matches the ramp waveform in the real dataset)
        return p.amp_v * sps.sawtooth(2 * np.pi * p.freq_hz * t, width=1.0)
    if w == "pulse":
        sq = sps.square(2 * np.pi * p.freq_hz * t, duty=p.duty)
        return p.amp_v * (sq + 1.0) / 2.0  # unipolar pulse train, 0..amp_v
    if w == "noise":
        rng = np.random.default_rng(p.seed)
        raw = rng.standard_normal(len(t))
        # band-limit to roughly the same spectral character as the other
        # waveform types' commanded frequency, so "noise" isn't literally
        # infinite-bandwidth
        sos = sps.butter(4, min(p.freq_hz * 5, FS_SIM / 2 - 1), fs=FS_SIM, output="sos")
        return p.amp_v * sps.sosfiltfilt(sos, raw)
    raise ValueError(f"unknown waveform type: {w!r}")


def _play_operator(u: np.ndarray, width: float) -> np.ndarray:
    """Rate-independent backlash/play hysteresis operator, ONE elementary
    "hysteron" -- the building block _prandtl_ishlinskii() sums several
    of.

    y[0] = u[0]; y[n] = max(u[n]-width, min(u[n]+width, y[n-1])).

    The window is centered on the CURRENT INPUT u[n] (not the previous
    output y[n-1] -- see module docstring for why that version was wrong).
    This formulation is exact regardless of how finely the input is
    sampled: during any monotonic run of u, y tracks u offset by exactly
    -width (rising) or +width (falling) once past the initial transient,
    giving a loop of height 2*width in the (u, y) plane independent of
    amplitude, frequency, or timestep -- verified numerically (constant
    2*width loop height for amplitude 0.2 through 4.0, at FS_SIM).
    """
    if width <= 0:
        return u.copy()
    y = np.empty_like(u)
    y[0] = u[0]
    for n in range(1, len(u)):
        y[n] = max(u[n] - width, min(u[n] + width, y[n - 1]))
    return y


def _prandtl_ishlinskii(u: np.ndarray, thresholds, weights) -> np.ndarray:
    """Weighted sum of play operators at different thresholds -- the
    standard way (Prandtl-Ishlinskii model) to build a richer hysteresis
    operator out of the simplest possible elementary one.

    Physical intuition: a small-amplitude drive only ever moves within
    the smallest thresholds' windows, so only the small-threshold
    hysterons ever engage; a larger-amplitude drive additionally engages
    the larger-threshold hysterons, adding their contribution to the
    aggregate loop. This makes the AGGREGATE loop height grow with drive
    amplitude (something a single play operator cannot do by
    construction) and then saturate once amplitude exceeds every
    threshold -- verified numerically for the default thresholds/weights
    below: loop height 0.02 -> 0.07 -> 0.128 -> 0.168 (saturated) as
    amplitude sweeps 0.01 -> 0.05 -> 0.15 -> 0.4+.

    Weights are normalized to sum to 1 here (not required to sum to 1 by
    the caller) so K (physics_model's overall static gain) keeps its
    existing meaning as the sole control of overall output scale --
    otherwise K and sum(weights) would be redundant, non-identifiable
    parameters when fitting to data.
    """
    total_w = sum(weights)
    if total_w <= 0:
        return u.copy()
    out = np.zeros_like(u)
    for r, w in zip(thresholds, weights):
        if w <= 0:
            continue
        out += (w / total_w) * _play_operator(u, max(r, 0.0))
    return out


def _fopdt_response(u: np.ndarray, K: float, tau: float, theta: float, fs: float) -> np.ndarray:
    """Exact zero-order-hold discretisation of the FOPDT linear dynamics,
    same equation as draft_v2.tex Eq. (zoh): Abar = exp(-dt/tau),
    Bbar = K*(1-exp(-dt/tau)). Dead time applied as an integer-sample
    delay (exact at FS_SIM since theta*FS_SIM = 1 sample for the
    default 5us/200kHz combination; rounds sensibly for other values).
    """
    dt = 1.0 / fs
    delay_samples = max(0, int(round(theta * fs)))
    if delay_samples > 0:
        u_delayed = np.concatenate([np.full(delay_samples, u[0]), u[:-delay_samples]])
    else:
        u_delayed = u

    Abar = np.exp(-dt / tau)
    Bbar = K * (1.0 - Abar)
    # h[n] = Abar*h[n-1] + Bbar*u_delayed[n-1], h[0] = Bbar*u_delayed[0]
    b = [0.0, Bbar]
    a = [1.0, -Abar]
    y = sps.lfilter(b, a, u_delayed)
    return y


def simulate(p: TwinParams) -> dict:
    """Run the forward digital twin and return time-domain signals at
    FS_OUT (1 kHz, matching real hardware), plus the FS_SIM intermediate
    stages for inspection/plotting."""
    n_sim = int(round(p.duration_s * FS_SIM))
    t_sim = np.arange(n_sim) / FS_SIM

    u_cmd = _command_signal(p, t_sim)
    u_hyst = _prandtl_ishlinskii(
        u_cmd,
        [p.hyst_r1_v, p.hyst_r2_v, p.hyst_r3_v],
        [p.hyst_w1, p.hyst_w2, p.hyst_w3],
    )
    y_lin = _fopdt_response(u_hyst, p.K_nm_per_v, p.tau_s, p.theta_s, FS_SIM)

    if p.sat_nm and p.sat_nm > 0:
        y_sat = p.sat_nm * np.tanh(y_lin / p.sat_nm)
    else:
        y_sat = y_lin

    rng = np.random.default_rng(p.seed)
    y_noisy = y_sat + p.noise_nm * rng.standard_normal(n_sim)

    n_out = int(round(p.duration_s * FS_OUT))
    y_out = sps.resample_poly(y_noisy, up=1, down=int(FS_SIM / FS_OUT))[:n_out]
    u_out = sps.resample_poly(u_cmd, up=1, down=int(FS_SIM / FS_OUT))[:n_out]
    t_out = np.arange(len(y_out)) / FS_OUT

    return {
        "t": t_out,
        "y_nm": y_out - np.mean(y_out),  # zero-mean, matches load_signal() convention
        "u_v": u_out,
        "fs": FS_OUT,
        # intermediate stages, for a "what did each physical effect do" view
        "t_sim": t_sim,
        "u_cmd_sim": u_cmd,
        "u_hyst_sim": u_hyst,
        "y_lin_sim": y_lin,
    }
