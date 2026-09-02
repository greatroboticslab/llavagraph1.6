"""
app.py -- Digital Twin demo v1 (Streamlit)
===========================================
Open-loop PZT actuator digital twin: simulate a waveform from physical
parameters, monitor its physical properties in real time, compare it
against real hardware measurements of the SAME waveform type AND
frequency (using the exact same feature-extraction code the paper's
real-data pipeline uses), and calibrate the physical parameters against
a batch of real measurements (T1 baseline).

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from physics_model import TwinParams, WAVEFORM_TYPES, simulate, FS_OUT
from feature_extract import extract_features, window_slice, real_feature_stats, \
    real_measurements, real_measurement_count
import feature_extract as fe
import feature_docs
from calibrate import calibrate as run_calibration, feature_gap

_CORRECTOR_WAVEFORMS = ["sine", "square", "ramp", "pulse"]  # matches build_pairs_residual.py scope


@st.cache_resource
def load_corrector():
    """Trained Mamba residual corrector + Stage-2 shared-calibrated
    parameters (T1-v2), if both exist. The corrector was only ever
    trained on T1-v2-based ideal signals (train_mamba_correction.py) --
    feeding it a signal built from arbitrary hand-tuned slider values
    would be out-of-distribution, so this is applied on top of T1-v2's
    fixed parameters, not whatever the sidebar sliders currently say."""
    this_dir = Path(__file__).resolve().parent
    ckpt_path, fitted_path = this_dir / "mamba_correction.pt", this_dir / "stage2_fitted_params.json"
    if not (ckpt_path.exists() and fitted_path.exists()):
        return None, None, None
    import torch
    from mamba_twin_model import MambaTwin
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MambaTwin(d_model=ckpt["d_model"], d_state=ckpt["d_state"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    fitted = json.load(open(fitted_path))["fitted"]
    return model, ckpt["y_scale"], fitted

st.set_page_config(page_title="PZT Digital Twin -- Demo v1", layout="wide")
st.title("Open-Loop PZT Digital Twin -- Demo v1")
st.caption(
    f"Physics simulator (hysteresis -> FOPDT -> saturation -> noise) + real-time "
    f"property monitor. Feature formulas are imported directly from "
    f"mamballm/batch_feature_extraction.py -- the same code that built the "
    f"paper's real-data feature table. Real-data reference: "
    f"{real_measurement_count()} audited-usable open-loop measurements "
    f"(Mamballm2/features_full.csv, 493 raw rows minus 4 excluded per the "
    f"leakage audit [docs/github_data_audit_openloop.md] minus 1 more excluded "
    f"as a sensor/interferometer glitch [audit_raw_data.py]). No closed-loop "
    f"data anywhere in this app."
)

# ---------------------------------------------------------------- defaults --
DEFAULTS = {
    "waveform": "sine", "freq_hz": 100.0, "amp_v": 1.0, "duration_s": 0.3,
    "duty": 0.5, "K": 380.0, "tau_us": 250.0, "theta_us": 5.0,
    "hyst_r1": 0.02, "hyst_r2": 0.08, "hyst_r3": 0.25,
    "hyst_w1": 0.5, "hyst_w2": 0.3, "hyst_w3": 0.2,
    "sat_nm": 2000.0, "noise_nm": 3.0, "seed": 42,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Command waveform")
    waveform = st.selectbox("Waveform type", WAVEFORM_TYPES, key="waveform")
    freq_hz = st.slider("Frequency (Hz)", 1.0, 500.0, key="freq_hz")
    amp_v = st.slider("Drive amplitude (V)", 0.1, 5.0, key="amp_v")
    duty = 0.5
    if waveform in ("square", "pulse"):
        duty = st.slider("Duty cycle", 0.05, 0.95, key="duty")
    duration_s = st.slider("Duration (s)", 0.1, 1.0, key="duration_s")

    st.header("Linear dynamics (FOPDT)")
    st.caption("Defaults match the calibrated values already used in "
               "batch_feature_extraction.py for this device.")
    K = st.slider("K -- static gain (nm/V)", 100.0, 600.0, key="K")
    tau_us = st.slider("tau -- time constant (us)", 50.0, 600.0, key="tau_us")
    theta_us = st.slider("theta -- dead time (us)", 0.0, 50.0, key="theta_us")

    st.header("Nonlinearities")
    with st.expander("Hysteresis (Prandtl-Ishlinskii, 3 hysterons)", expanded=False):
        st.caption(
            "Weighted sum of 3 play operators at different thresholds. A single "
            "operator can't produce amplitude-dependent hysteresis loop growth; "
            "summing several at different thresholds does -- small drive only "
            "engages the small-threshold hysteron(s), larger drive engages more."
        )
        hyst_r1 = st.slider("Threshold r1 (V)", 0.0, 0.5, key="hyst_r1")
        hyst_w1 = st.slider("Weight w1", 0.0, 1.0, key="hyst_w1")
        hyst_r2 = st.slider("Threshold r2 (V)", 0.0, 0.5, key="hyst_r2")
        hyst_w2 = st.slider("Weight w2", 0.0, 1.0, key="hyst_w2")
        hyst_r3 = st.slider("Threshold r3 (V)", 0.0, 0.5, key="hyst_r3")
        hyst_w3 = st.slider("Weight w3", 0.0, 1.0, key="hyst_w3")
        st.caption("Weights are normalized to sum to 1 internally -- only their "
                   "*relative* size matters, K keeps sole control of overall scale.")
    sat_nm = st.slider("Saturation limit (nm)", 200.0, 5000.0, key="sat_nm")
    noise_nm = st.slider("Measurement noise (nm RMS)", 0.0, 20.0, key="noise_nm")

    seed = st.number_input("Random seed", step=1, key="seed")

params = TwinParams(
    waveform=waveform, freq_hz=freq_hz, amp_v=amp_v, duration_s=duration_s, duty=duty,
    K_nm_per_v=K, tau_s=tau_us * 1e-6, theta_s=theta_us * 1e-6,
    hyst_r1_v=hyst_r1, hyst_r2_v=hyst_r2, hyst_r3_v=hyst_r3,
    hyst_w1=hyst_w1, hyst_w2=hyst_w2, hyst_w3=hyst_w3,
    sat_nm=sat_nm, noise_nm=noise_nm, seed=int(seed),
)

# ---------------------------------------------------------------- real-data matching --
# Frequency-matching: for anything except noise (whose "dominant frequency"
# is not a physically meaningful quantity -- see fft_features()'s own
# docstring), only offer real measurements whose ACTUAL measured
# dominant_freq_hz is close to the current command frequency. Comparing a
# 100Hz simulated sine against an 800Hz real sine would not be a physics
# comparison, just a coincidence of both being labeled "sine".
real_df = real_measurements(waveform)
real_df = real_df[real_df["path"].notna()]  # only files actually found on this machine

if waveform == "noise":
    matched = real_df.copy()
else:
    tol = max(10.0, 0.2 * freq_hz)
    matched = real_df[(real_df["dominant_freq_hz"] - freq_hz).abs() <= tol].copy()
    matched["diff"] = (matched["dominant_freq_hz"] - freq_hz).abs()
    matched = matched.sort_values("diff")

# ---------------------------------------------------------------- simulate --
result = simulate(params)
sim_window = window_slice(result["y_nm"], fs=FS_OUT)
sim_feat = extract_features(result["y_nm"], sim_window, waveform)

# ---------------------------------------------------------------- layout --
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Time-domain waveform")

    if waveform != "noise" and len(matched) == 0:
        nearest = real_df.reindex((real_df["dominant_freq_hz"] - freq_hz).abs().sort_values().index).head(5)
        st.warning(
            f"No real {waveform} measurements within {tol:.0f} Hz of {freq_hz:.0f} Hz. "
            f"Nearest available real frequencies: "
            + ", ".join(f"{f:.0f} Hz" for f in nearest["dominant_freq_hz"])
            + ". Move the Frequency slider to one of these to enable a real-data overlay."
        )
        overlay_choice = None
    else:
        options = ["(none)"] + [
            f"{row.filename}  ({row.dominant_freq_hz:.1f} Hz)" for row in matched.itertuples()
        ]
        overlay_choice = st.selectbox("Overlay a real measurement (frequency-matched)", options)

    real_signal = real_window = real_feat = None
    if overlay_choice and overlay_choice != "(none)":
        fname = overlay_choice.split("  (")[0]
        real_path = matched.loc[matched["filename"] == fname, "path"].iloc[0]
        real_signal, real_window = fe.bfe.load_signal(str(real_path))
        real_feat = extract_features(real_signal, real_window, waveform)

    corrector_model, corrector_yscale, t1v2_fitted = load_corrector()
    show_correction = False
    corrected_t = corrected_y = None
    if corrector_model is None:
        st.caption("Mamba correction overlay unavailable: run `stage2_calibrate_shared.py` "
                   "and `train_mamba_correction.py` first (see digital_twin/README.md).")
    elif waveform not in _CORRECTOR_WAVEFORMS:
        st.caption(f"Mamba correction overlay not available for '{waveform}' -- "
                   f"trained only for {_CORRECTOR_WAVEFORMS} (see build_pairs_residual.py).")
    else:
        show_correction = st.checkbox(
            "Show Mamba-corrected waveform (T1-v2 physics, not the sliders above)",
            help="The corrector was only ever trained on top of Stage 2's shared-calibrated "
                 "parameters (T1-v2). Applying it to a signal built from arbitrary slider "
                 "values would be out-of-distribution, so this line always uses T1-v2's fixed "
                 "parameters at the current Frequency/Duration -- it will not change as you "
                 "move the K/tau/hysteresis/saturation sliders."
        )
        if show_correction:
            import torch
            t1v2_params = TwinParams(waveform=waveform, freq_hz=freq_hz, duration_s=duration_s,
                                      duty=duty, seed=int(seed), **t1v2_fitted)
            t1v2_result = simulate(t1v2_params)
            with torch.no_grad():
                u_t = torch.from_numpy(t1v2_result["y_nm"] / corrector_yscale).float().unsqueeze(0)
                delta = corrector_model(u_t).numpy()[0] * corrector_yscale
            corrected_y = t1v2_result["y_nm"] + delta
            corrected_t = t1v2_result["t"] * 1000

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(result["t"] * 1000, result["y_nm"], label="simulated (sliders above)", lw=1.2)
    if show_correction and corrected_y is not None:
        ax.plot(corrected_t, corrected_y, label="T1-v2 + Mamba correction", lw=1.3, color="tab:red")
    if real_signal is not None:
        t_real = np.arange(len(real_signal)) / FS_OUT * 1000
        n_show = min(len(t_real), int(duration_s * FS_OUT))
        ax.plot(t_real[:n_show], (real_signal - np.mean(real_signal))[:n_show],
                label=f"real: {fname}", lw=1.0, alpha=0.75)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("displacement (nm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    with st.expander("Show what each physical stage does (command -> hysteresis -> FOPDT)"):
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        n_show = int(min(len(result["t_sim"]), 0.02 * 200_000))  # first 20ms
        ax2.plot(result["t_sim"][:n_show] * 1000, result["u_cmd_sim"][:n_show],
                 label="command u(t) [V]", lw=1)
        ax2.plot(result["t_sim"][:n_show] * 1000, result["u_hyst_sim"][:n_show],
                 label="after hysteresis [V]", lw=1)
        ax2b = ax2.twinx()
        ax2b.plot(result["t_sim"][:n_show] * 1000, result["y_lin_sim"][:n_show],
                  label="after FOPDT [nm]", lw=1, color="green")
        ax2.set_xlabel("time (ms)")
        ax2.set_ylabel("V")
        ax2b.set_ylabel("nm")
        fig2.legend(loc="upper right", fontsize=8, bbox_to_anchor=(0.9, 0.88))
        st.pyplot(fig2, use_container_width=True)

with col2:
    st.subheader("Live physical properties")
    stats = real_feature_stats()

    def std_lookup(k):
        try:
            return stats.loc[waveform, (k, "std")]
        except KeyError:
            return None

    def mean_lookup(k):
        try:
            return stats.loc[waveform, (k, "mean")]
        except KeyError:
            return None

    overall_gap = feature_gap(sim_feat, {k: mean_lookup(k) for k in sim_feat}, std_lookup)
    gap_col, glossary_col = st.columns([3, 1])
    gap_col.metric("Gap score vs. real-device mean (0 = identical, avg squared z-diff)",
                    f"{overall_gap:.2f}" if np.isfinite(overall_gap) else "n/a")

    # Same filter as the table rows below, computed once so the glossary
    # button lists exactly the rows the table shows -- no extra/missing
    # entries between the two views.
    table_feature_names = [
        k for k, v in sim_feat.items()
        if k not in ("waveform", "n_samples") and v is not None and not isinstance(v, str)
    ]

    @st.dialog("What do these physical properties mean?", width="large")
    def _show_glossary(feature_names):
        st.caption(
            "Plain-English explanation of every measured/simulated quantity in the "
            "properties table, for this waveform type."
        )
        for name in feature_names:
            desc = feature_docs.explain(name)
            if not desc:
                continue
            st.markdown(f"**{name}**")
            st.write(desc)
            st.divider()

    glossary_col.write("")  # vertical alignment with the metric above
    if glossary_col.button("📖 Glossary"):
        _show_glossary(table_feature_names)

    rows = []
    for k in table_feature_names:
        v = sim_feat[k]
        mean, std = mean_lookup(k), std_lookup(k)
        flag = ""
        if mean is not None and std is not None and np.isfinite(std) and std > 0:
            z = abs((v - mean) / std)
            flag = "ok" if z <= 2 else ("borderline" if z <= 4 else "far from real")
        real_v = real_feat.get(k) if real_feat else None
        rows.append({
            "feature": k,
            "simulated": v,
            "real device mean+/-std": f"{mean:.3g} +/- {std:.3g}" if mean is not None else "n/a",
            "this real file": real_v, "check": flag,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)

st.divider()

# ---------------------------------------------------------------- calibration (T1) --
st.subheader("Calibrate to real data -- T1 baseline")
st.caption(
    "Fits K, tau, hysteresis width and saturation (theta excluded -- not "
    "identifiable at 1kHz output rate) against a batch of frequency-matched "
    "real measurements, using a gradient-free optimizer (Nelder-Mead) on the "
    "same gap score shown above. This is the classical baseline the later "
    "Mamba learned twin (T2) needs to beat."
)

if waveform != "noise" and len(matched) > 0:
    batch_n = st.slider("Batch size (real measurements to calibrate against)",
                         1, min(8, len(matched)), min(4, len(matched)))
    if st.button("Run calibration"):
        real_batch = []
        for row in matched.head(batch_n).itertuples():
            sig, win = fe.bfe.load_signal(str(row.path))
            feat = extract_features(sig, win, waveform)
            real_batch.append((row.dominant_freq_hz, feat))
        with st.spinner(f"Fitting to {len(real_batch)} real measurement(s)..."):
            fitted, gap_before, gap_after, history = run_calibration(
                waveform, params, real_batch, std_lookup)
        st.session_state["last_calibration"] = {
            "waveform": waveform, "n": len(real_batch),
            "fitted": fitted, "gap_before": gap_before, "gap_after": gap_after,
        }

    calib = st.session_state.get("last_calibration")
    if calib and calib["waveform"] == waveform:
        c1, c2, c3 = st.columns(3)
        c1.metric("Gap before", f"{calib['gap_before']:.2f}")
        c2.metric("Gap after", f"{calib['gap_after']:.2f}",
                  delta=f"{calib['gap_after'] - calib['gap_before']:.2f}")
        c3.metric("Batch size", calib["n"])
        f = calib["fitted"]
        st.write(
            f"Fitted: K={f.K_nm_per_v:.1f} nm/V, tau={f.tau_s*1e6:.1f} us, "
            f"hysteresis thresholds=({f.hyst_r1_v:.3f}, {f.hyst_r2_v:.3f}, {f.hyst_r3_v:.3f}) V, "
            f"weights=({f.hyst_w1:.2f}, {f.hyst_w2:.2f}, {f.hyst_w3:.2f}), "
            f"saturation={f.sat_nm:.0f} nm"
        )
        def _apply_fitted_params():
            # Widget-bound session_state keys (K, tau_us, ...) can only be
            # written from an on_click callback, which Streamlit runs BEFORE
            # the sidebar widgets are re-instantiated on the next script
            # pass. Writing them directly in the button's `if st.button(...)`
            # body (as an earlier version of this file did) raises
            # StreamlitAPIException: "cannot be modified after the widget
            # ... is instantiated", because by the time that code runs, this
            # same script pass has already created the K/tau_us/... sliders
            # further up the page.
            calib = st.session_state["last_calibration"]
            fitted = calib["fitted"]
            st.session_state["K"] = float(fitted.K_nm_per_v)
            st.session_state["tau_us"] = float(fitted.tau_s * 1e6)
            st.session_state["hyst_r1"] = float(fitted.hyst_r1_v)
            st.session_state["hyst_r2"] = float(fitted.hyst_r2_v)
            st.session_state["hyst_r3"] = float(fitted.hyst_r3_v)
            st.session_state["hyst_w1"] = float(fitted.hyst_w1)
            st.session_state["hyst_w2"] = float(fitted.hyst_w2)
            st.session_state["hyst_w3"] = float(fitted.hyst_w3)
            st.session_state["sat_nm"] = float(fitted.sat_nm)

        st.button("Apply fitted parameters to the sliders above",
                  on_click=_apply_fitted_params)
else:
    st.info("Calibration needs at least one frequency-matched real measurement "
            "(or, for noise, at least one real measurement) to fit against.")

st.divider()
st.caption(
    "Roadmap: T0 = this analytical twin (with or without T1 calibration). "
    "T2 (not built yet) = a small Mamba SSM trained as a learned forward twin "
    "(command -> displacement) on real open-loop data, compared against T0/T1 "
    "on the same gap score."
)
