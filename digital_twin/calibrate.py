"""
calibrate.py
=============
T1 baseline: classical (gradient-free) optimizer calibration of the
digital twin's physical parameters against real measurements.

Why this exists: it answers "how much of the sim-to-real gap is fixable
by just picking better physical parameters, with no learning at all?"
That's the baseline the later Mamba learned twin (T2) has to beat to be
worth the extra complexity -- see digital_twin/README.md roadmap.

Two entry points:
  - calibrate(): fits one waveform type at a time (used by app.py's
    interactive panel).
  - calibrate_shared(): fits ONE parameter set across multiple waveform
    types at once (Stage 2 of the physics-deepening plan) -- physically
    correct since K/tau/hysteresis are properties of the device, not of
    which command shape you happen to be driving it with. Reuses the
    same objective machinery.

Nelder-Mead is used (not a gradient method) because the objective goes
through FFT peak-picking and argmax operations inside
feature_extract.extract_features(), which aren't differentiable.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from physics_model import TwinParams, simulate
from feature_extract import extract_features, window_slice

# theta (dead time) is intentionally excluded from calibration: at
# FS_OUT=1kHz its effect on the *output* signal is far below one sample
# period (5us vs 1ms), so it isn't identifiable from this data -- fitting
# it would just chase optimizer noise.
#
# Bounds are kept identical to the sidebar sliders' [min, max] in app.py --
# a fitted value outside its slider's range would crash Streamlit when
# "Apply fitted parameters" writes it into that slider's session_state key.
# Hysteresis is now 6 params (Prandtl-Ishlinskii: 3 thresholds + 3
# weights, see physics_model.py Stage 1 revision note) instead of the
# old single hysteresis_v.
_PARAM_NAMES = [
    "K_nm_per_v", "tau_s",
    "hyst_r1_v", "hyst_r2_v", "hyst_r3_v",
    "hyst_w1", "hyst_w2", "hyst_w3",
    "sat_nm",
]
_BOUNDS = [
    (100.0, 600.0), (50e-6, 600e-6),
    (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    (200.0, 5000.0),
]

_Z_CAP = 10.0  # see feature_gap() docstring


def feature_gap(sim_feat: dict, real_feat: dict, std_lookup) -> float:
    """Mean squared z-difference (using the real-device population std as
    the normalizer) over every numeric feature both dicts have. Same idea
    as the app's per-feature "check" column, collapsed to one number so
    an optimizer (and a human) can compare two parameter sets at a
    glance.

    Each feature's z-difference is capped at _Z_CAP before squaring.
    Without this, ratio-type features (band_energy_ratio,
    odd_even_harmonic_ratio) can have a near-zero denominator for an
    unusually clean signal and blow up to z~10^4 -- found empirically via
    eval_mamba_twin.py, where one such feature alone inflated a whole
    waveform type's average gap into the millions and made the T0-vs-T2
    comparison meaningless. Capping keeps one numerically unstable
    feature from swamping the other ~40 well-behaved ones; it does not
    hide real mismatches on any bounded feature (those never approach the
    cap)."""
    total, n = 0.0, 0
    for k, v in sim_feat.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or np.isnan(v):
            # NaN on the sim side happens legitimately for some feature/
            # frequency/window-length combinations -- e.g. ramp_rise_linearity
            # is NaN whenever the window is short enough that a ramp cycle
            # doesn't complete (see ramp_specific_features()'s own NaN
            # fallback). Skipping just this one feature (not the whole
            # example) is correct; previously this wasn't checked at all,
            # so one such NaN silently poisoned the entire batch mean into
            # NaN -- caught via calibrate() printing "gap nan -> nan" for
            # ramp/pulse, which made the Nelder-Mead objective flat (nan
            # everywhere) and it never moved the parameters from x0.
            continue
        if k not in real_feat:
            continue
        rv = real_feat[k]
        if not isinstance(rv, (int, float)) or isinstance(rv, bool) or np.isnan(rv):
            continue
        std = std_lookup(k)
        if std is None or not np.isfinite(std) or std <= 0:
            continue
        z = min(abs((v - rv) / std), _Z_CAP)
        total += z ** 2
        n += 1
    return total / n if n else np.inf


def _params_from_vector(base_params: TwinParams, x, **overrides) -> TwinParams:
    kwargs = dict(zip(_PARAM_NAMES, x))
    kwargs.update(overrides)
    return replace(base_params, **kwargs)


def calibrate(waveform: str, base_params: TwinParams, real_batch: list,
              std_lookup, calib_duration_s: float = 0.15, maxfev: int = 400):
    """real_batch: list of (freq_hz, real_feat_dict), one per real
    measurement to calibrate against (features already extracted by the
    caller). Returns (fitted_TwinParams, gap_before, gap_after, history).

    maxfev default raised from the original 120 to 400 now that there
    are 9 fitted parameters instead of 4 -- Nelder-Mead needs more
    function evaluations per added dimension to converge properly.
    """

    def objective(x):
        gaps = []
        for freq_hz, real_feat in real_batch:
            p = _params_from_vector(base_params, x, freq_hz=freq_hz, duration_s=calib_duration_s)
            r = simulate(p)
            win = window_slice(r["y_nm"])
            sim_feat = extract_features(r["y_nm"], win, waveform)
            gaps.append(feature_gap(sim_feat, real_feat, std_lookup))
        return float(np.mean(gaps))

    x0 = np.array([getattr(base_params, name) for name in _PARAM_NAMES])
    gap_before = objective(x0)
    history = [gap_before]

    res = minimize(objective, x0, method="Nelder-Mead", bounds=_BOUNDS,
                    callback=lambda xk: history.append(objective(xk)),
                    options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-3})

    fitted = _params_from_vector(base_params, res.x)
    return fitted, gap_before, float(res.fun), history


def calibrate_shared(base_params_by_waveform: dict, real_batch: list,
                      std_lookup_by_waveform: dict,
                      calib_duration_s: float = 0.15, maxfev: int = 800,
                      verbose: bool = False):
    """Fits ONE shared parameter set across real measurements spanning
    multiple waveform types -- Stage 2 of the physics-deepening plan.

    real_batch: list of (waveform, freq_hz, real_feat_dict).
    base_params_by_waveform: {waveform: TwinParams} -- supplies each
      waveform's non-fitted fields (duty cycle, etc.); the fitted fields
      (_PARAM_NAMES) come from x/x0, identical across every waveform.
    std_lookup_by_waveform: {waveform: std_lookup_fn}.

    Returns (fitted_param_dict, gap_before, gap_after, history).
    fitted_param_dict maps _PARAM_NAMES -> value; build a per-waveform
    TwinParams with e.g. replace(base_params_by_waveform[wf], **fitted).
    """
    waveforms = sorted(base_params_by_waveform.keys())
    x0 = np.array([getattr(base_params_by_waveform[waveforms[0]], name) for name in _PARAM_NAMES])

    def objective(x):
        gaps = []
        for wf, freq_hz, real_feat in real_batch:
            p = _params_from_vector(base_params_by_waveform[wf], x,
                                     freq_hz=freq_hz, duration_s=calib_duration_s)
            r = simulate(p)
            win = window_slice(r["y_nm"])
            sim_feat = extract_features(r["y_nm"], win, wf)
            gaps.append(feature_gap(sim_feat, real_feat, std_lookup_by_waveform[wf]))
        return float(np.mean(gaps))

    gap_before = objective(x0)
    history = [gap_before]

    def callback(xk):
        g = objective(xk)
        history.append(g)
        if verbose and len(history) % 20 == 0:
            print(f"  [calibrate_shared] eval {len(history)}: gap={g:.3f}")

    res = minimize(objective, x0, method="Nelder-Mead", bounds=_BOUNDS,
                    callback=callback,
                    options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-3})

    fitted = dict(zip(_PARAM_NAMES, res.x))
    return fitted, gap_before, float(res.fun), history
