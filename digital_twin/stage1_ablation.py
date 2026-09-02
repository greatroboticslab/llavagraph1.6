"""
stage1_ablation.py
====================
Task 1: decompose the 23.6% physics-only gap reduction (old buggy T0 ->
T1-v2) into what each individual change actually contributed. The
combined number alone doesn't say whether the bug fix, the PI model
structure, or the shared calibration did the work.

Variants (each simulated with its own local copy of simulate(), so this
file never touches physics_model.py's production code):

  B0  old buggy single play operator (window centered on previous
      output -- the bug), default/uncalibrated params. Reconstructs
      what the demo actually was before Stage 1.
  B1  bug FIXED, but still a single operator (no PI), same
      default/uncalibrated params. Isolates the bug fix alone.
  B2  PI model (3 hysterons), default/uncalibrated (synthetic-guess)
      params -- this is what physics_model.py's TwinParams() defaults
      already are. Isolates the PI structure alone, on top of B1.
  B3  PI + shared calibration = T1-v2 (stage2_fitted_params.json).
      Isolates the calibration step, on top of B2. This is the number
      already reported as the Stage 1+2 result.
  B4  single operator (bug fixed), but SHARED-CALIBRATED like B3 (not
      just left at defaults like B1) -- forces the PI weights w2, w3 to
      0 during calibration (locked via optimizer bounds) so only one
      hysteron is ever active. Answers the specific question "would a
      single calibrated operator have been just as good as PI, without
      needing the multi-hysteron structure at all?"

All five evaluated on the SAME held-out test split (stage_split.json)
with the same feature-based gap score used everywhere else.
"""

import json

import numpy as np
from scipy import signal as sps
from scipy.optimize import minimize

from physics_model import TwinParams, FS_SIM, FS_OUT, _command_signal, _fopdt_response
from feature_extract import extract_features, window_slice, real_feature_stats
from calibrate import feature_gap, calibrate_shared, _PARAM_NAMES as _FULL_PARAM_NAMES, _BOUNDS as _FULL_BOUNDS

WAVEFORMS = ["sine", "square", "ramp", "pulse"]


# ---------------------------------------------------------------- local, self-contained simulate() variants --
def _old_buggy_play(u, width):
    """The ORIGINAL (buggy) operator: window centered on the PREVIOUS
    OUTPUT. Converges to a no-op as timestep -> 0. Reconstructed here
    only for this ablation; physics_model.py no longer contains this."""
    if width <= 0:
        return u.copy()
    y = np.empty_like(u)
    y[0] = u[0]
    lo, hi = y[0] - width, y[0] + width
    for n in range(1, len(u)):
        y[n] = min(max(u[n], lo), hi)
        lo, hi = y[n] - width, y[n] + width
    return y


def _fixed_play(u, width):
    """The CORRECTED single operator: window centered on current input."""
    if width <= 0:
        return u.copy()
    y = np.empty_like(u)
    y[0] = u[0]
    for n in range(1, len(u)):
        y[n] = max(u[n] - width, min(u[n] + width, y[n - 1]))
    return y


def _pi(u, thresholds, weights):
    total_w = sum(weights)
    if total_w <= 0:
        return u.copy()
    out = np.zeros_like(u)
    for r, w in zip(thresholds, weights):
        if w > 0:
            out += (w / total_w) * _fixed_play(u, max(r, 0.0))
    return out


def simulate_variant(p: TwinParams, hyst_fn) -> np.ndarray:
    """Mirrors physics_model.simulate(), but with a pluggable hysteresis
    function -- hyst_fn(u_cmd) -> u_hyst."""
    n_sim = int(round(p.duration_s * FS_SIM))
    t_sim = np.arange(n_sim) / FS_SIM
    u_cmd = _command_signal(p, t_sim)
    u_hyst = hyst_fn(u_cmd)
    y_lin = _fopdt_response(u_hyst, p.K_nm_per_v, p.tau_s, p.theta_s, FS_SIM)
    y_sat = p.sat_nm * np.tanh(y_lin / p.sat_nm) if p.sat_nm > 0 else y_lin
    rng = np.random.default_rng(p.seed)
    y_noisy = y_sat + p.noise_nm * rng.standard_normal(n_sim)
    n_out = int(round(p.duration_s * FS_OUT))
    y_out = sps.resample_poly(y_noisy, up=1, down=int(FS_SIM / FS_OUT))[:n_out]
    return y_out - np.mean(y_out)


# ---------------------------------------------------------------- evaluation --
def evaluate(label, hyst_fn, params_by_waveform, test_items, stats):
    def mean_lookup(wf):
        def f(k):
            try:
                return stats.loc[wf, (k, "mean")]
            except KeyError:
                return None
        return f

    def std_lookup(wf):
        def f(k):
            try:
                return stats.loc[wf, (k, "std")]
            except KeyError:
                return None
        return f

    from feature_extract import real_measurements
    import feature_extract as fe

    gaps = {wf: [] for wf in WAVEFORMS}
    for wf in WAVEFORMS:
        real_df = real_measurements(wf)
        items = [it for it in test_items if it["waveform"] == wf]
        for it in items:
            row = real_df[real_df["filename"] == it["filename"]]
            if row.empty or row["path"].isna().all():
                continue
            path = row["path"].iloc[0]
            sig, win = fe.bfe.load_signal(str(path))
            real_feat = extract_features(sig, win, wf)
            real_dict = {k: mean_lookup(wf)(k) for k in real_feat}

            p = params_by_waveform[wf]
            p.freq_hz = it["freq_hz"]
            p.duration_s = 0.26
            y_sim = simulate_variant(p, hyst_fn)
            sim_feat = extract_features(y_sim, window_slice(y_sim), wf)
            gaps[wf].append(feature_gap(sim_feat, real_dict, std_lookup(wf)))

    row = {wf: float(np.nanmean(gaps[wf])) if gaps[wf] else float("nan") for wf in WAVEFORMS}
    row["overall"] = float(np.nanmean([g for wf in WAVEFORMS for g in gaps[wf]]))
    print(f"{label:<45}" + "".join(f"{row[wf]:>10.3f}" for wf in WAVEFORMS) + f"{row['overall']:>10.3f}")
    return row


def calibrate_single_operator_shared(base_params_by_waveform, real_batch, std_lookup_by_waveform):
    """Same as calibrate_shared(), but with w2 and w3 locked to 0 via
    bounds -- forces a single-hysteron-equivalent PI model, so
    calibration can only ever move r1/w1 (plus K, tau, sat), never
    engage a second or third threshold. Reuses calibrate.py's objective
    machinery directly rather than duplicating it."""
    import calibrate as cal
    locked_bounds = list(_FULL_BOUNDS)
    w2_idx = _FULL_PARAM_NAMES.index("hyst_w2")
    w3_idx = _FULL_PARAM_NAMES.index("hyst_w3")
    locked_bounds[w2_idx] = (0.0, 0.0)
    locked_bounds[w3_idx] = (0.0, 0.0)
    orig_bounds = cal._BOUNDS
    cal._BOUNDS = locked_bounds
    try:
        return calibrate_shared(base_params_by_waveform, real_batch, std_lookup_by_waveform,
                                 calib_duration_s=0.15, maxfev=400, verbose=True)
    finally:
        cal._BOUNDS = orig_bounds


def main():
    split = json.load(open("stage_split.json"))
    test_items = split["test"]
    stats = real_feature_stats()
    print(f"{'variant':<45}" + "".join(f"{wf:>10}" for wf in WAVEFORMS) + f"{'overall':>10}")

    default_params = lambda: {wf: TwinParams(waveform=wf) for wf in WAVEFORMS}

    # B0: old buggy operator, uncalibrated defaults
    evaluate("B0 old buggy op, uncalibrated",
              lambda u: _old_buggy_play(u, 0.05), default_params(), test_items, stats)

    # B1: bug fixed, single operator, uncalibrated defaults (same width as before)
    evaluate("B1 bugfix only (single op), uncalibrated",
              lambda u: _fixed_play(u, 0.05), default_params(), test_items, stats)

    # B2: PI structure, uncalibrated (synthetic-guess) defaults
    pi_default = lambda u: _pi(u, [0.02, 0.08, 0.25], [0.5, 0.3, 0.2])
    evaluate("B2 bugfix+PI structure, uncalibrated",
              pi_default, default_params(), test_items, stats)

    # B3: PI + shared calibration = T1-v2 (already computed in Stage 2)
    fitted = json.load(open("stage2_fitted_params.json"))["fitted"]
    b3_params = {wf: TwinParams(waveform=wf, seed=0, **fitted) for wf in WAVEFORMS}
    pi_fitted = lambda u: _pi(u, [fitted["hyst_r1_v"], fitted["hyst_r2_v"], fitted["hyst_r3_v"]],
                              [fitted["hyst_w1"], fitted["hyst_w2"], fitted["hyst_w3"]])
    evaluate("B3 bugfix+PI+shared calibration (T1-v2)",
              pi_fitted, b3_params, test_items, stats)

    # B4: single operator, but shared-calibrated (w2=w3=0 locked)
    print("\ncalibrating B4 (single-hysteron-equivalent, shared)...")
    rng = np.random.default_rng(1)

    def make_std_lookup(waveform):
        def f(k):
            try:
                return stats.loc[waveform, (k, "std")]
            except KeyError:
                return None
        return f

    import feature_extract as fe
    from feature_extract import real_measurements
    train_items = split["train"]
    real_batch = []
    for wf in WAVEFORMS:
        items = [it for it in train_items if it["waveform"] == wf]
        chosen = rng.choice(len(items), size=min(10, len(items)), replace=False)
        real_df = real_measurements(wf)
        for i in chosen:
            m = items[i]
            path = real_df.loc[real_df["filename"] == m["filename"], "path"].iloc[0]
            sig, win = fe.bfe.load_signal(str(path))
            feat = extract_features(sig, win, wf)
            real_batch.append((wf, m["freq_hz"], feat))

    base_params_by_waveform = {wf: TwinParams(waveform=wf) for wf in WAVEFORMS}
    std_lookup_by_waveform = {wf: make_std_lookup(wf) for wf in WAVEFORMS}
    fitted_b4, gap_before_b4, gap_after_b4, _ = calibrate_single_operator_shared(
        base_params_by_waveform, real_batch, std_lookup_by_waveform)
    print(f"B4 calibration: gap {gap_before_b4:.3f} -> {gap_after_b4:.3f}")
    print("B4 fitted:", {k: round(v, 4) for k, v in fitted_b4.items()})

    b4_params = {wf: TwinParams(waveform=wf, seed=0, **fitted_b4) for wf in WAVEFORMS}
    pi_b4 = lambda u: _pi(u, [fitted_b4["hyst_r1_v"], fitted_b4["hyst_r2_v"], fitted_b4["hyst_r3_v"]],
                          [fitted_b4["hyst_w1"], fitted_b4["hyst_w2"], fitted_b4["hyst_w3"]])
    evaluate("B4 bugfix, single op, shared calibration",
              pi_b4, b4_params, test_items, stats)

    with open("stage1_ablation_b4.json", "w") as f:
        json.dump({"fitted": fitted_b4, "gap_before": gap_before_b4, "gap_after": gap_after_b4}, f, indent=2)


if __name__ == "__main__":
    main()
