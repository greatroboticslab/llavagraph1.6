"""
compare_T0_T1_T2.py
=====================
The real head-to-head the roadmap has been building toward: T0
(analytical FOPDT+hysteresis, default/uncalibrated), T1 (same analytical
model, K/tau/hysteresis/saturation classically fit per waveform type),
and T2 (the trained MambaTwin), all scored on the SAME held-out test
set with the SAME feature-based gap score.

Train/test discipline matches T2's: T1 is fit using only TRAIN-split
real measurements (never the test set), exactly like T2 was only
trained on the train split. Comparing a test-fit T1 against T2 would be
an unfair advantage for T1.

Also answers the more specific question motivating T2 in the first
place: does the learned SSM's effective decay rate resemble a real
tau, i.e. is the draft_v2.tex FOPDT<->Mamba-SSM correspondence table
just a shape-of-equation analogy, or does a Mamba block trained on this
task actually learn something tau-like?
"""

import json
from pathlib import Path

import numpy as np
import torch

from mamba_twin_model import MambaTwin
from feature_extract import extract_features, window_slice, real_feature_stats
import feature_extract as fe
from physics_model import TwinParams, simulate, DEFAULT_TAU_S
from calibrate import calibrate as run_calibration, feature_gap

DATA_DIR = Path("pairs_data")
CKPT_PATH = Path("mamba_twin.pt")
WAVEFORMS = ["sine", "square", "ramp", "pulse"]
T1_BATCH_SIZE = 15  # real training measurements per waveform type used to fit T1


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = MambaTwin(d_model=ckpt["d_model"], d_state=ckpt["d_state"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["y_scale"]


def std_mean_lookup(stats, waveform):
    def std(k):
        try:
            return stats.loc[waveform, (k, "std")]
        except KeyError:
            return None

    def mean(k):
        try:
            return stats.loc[waveform, (k, "mean")]
        except KeyError:
            return None
    return std, mean


def fit_t1(waveform: str, meta_train: list, stats, rng: np.random.Generator):
    std, _ = std_mean_lookup(stats, waveform)
    files = [m for m in meta_train if m["waveform"] == waveform]
    chosen = rng.choice(len(files), size=min(T1_BATCH_SIZE, len(files)), replace=False)
    real_batch = []
    for i in chosen:
        m = files[i]
        real_df = fe.real_measurements(waveform)
        path = real_df.loc[real_df["filename"] == m["filename"], "path"].iloc[0]
        sig, win = fe.bfe.load_signal(str(path))
        feat = extract_features(sig, win, waveform)
        real_batch.append((m["freq_hz"], feat))
    base = TwinParams(waveform=waveform)
    fitted, gap_before, gap_after, _ = run_calibration(
        waveform, base, real_batch, std, calib_duration_s=0.15, maxfev=150)
    print(f"  T1 fit [{waveform}] on {len(real_batch)} train examples: "
          f"gap {gap_before:.2f} -> {gap_after:.2f}, "
          f"K={fitted.K_nm_per_v:.1f} tau={fitted.tau_s*1e6:.1f}us "
          f"hyst_r=({fitted.hyst_r1_v:.3f},{fitted.hyst_r2_v:.3f},{fitted.hyst_r3_v:.3f})V "
          f"sat={fitted.sat_nm:.0f}nm")
    return fitted


def main():
    model, y_scale = load_model()
    stats = real_feature_stats()
    meta = json.load(open(DATA_DIR / "meta.json"))
    d_test = np.load(DATA_DIR / "test.npz")
    u_test, y_test = d_test["u"], d_test["y"]

    rng = np.random.default_rng(0)

    print("Fitting T1 (per waveform type, train-split only)...")
    t1_params = {wf: fit_t1(wf, meta["train"], stats, rng) for wf in WAVEFORMS}

    with torch.no_grad():
        y_pred_all = model(torch.from_numpy(u_test).float()).numpy() * y_scale

    gaps = {wf: {"T0": [], "T1": [], "T2": []} for wf in WAVEFORMS}

    for i, m in enumerate(meta["test"]):
        waveform, freq_hz = m["waveform"], m["freq_hz"]
        std, mean = std_mean_lookup(stats, waveform)
        real_y = y_test[i]
        real_feat = extract_features(real_y, window_slice(real_y), waveform)
        real_dict = {k: mean(k) for k in real_feat}

        # T0: default params
        p0 = TwinParams(waveform=waveform, freq_hz=freq_hz, duration_s=len(real_y) / 1000.0)
        r0 = simulate(p0)
        f0 = extract_features(r0["y_nm"], window_slice(r0["y_nm"]), waveform)
        gaps[waveform]["T0"].append(feature_gap(f0, real_dict, std))

        # T1: calibrated params (fit on train split only)
        from dataclasses import replace
        p1 = replace(t1_params[waveform], freq_hz=freq_hz, duration_s=len(real_y) / 1000.0)
        r1 = simulate(p1)
        f1 = extract_features(r1["y_nm"], window_slice(r1["y_nm"]), waveform)
        gaps[waveform]["T1"].append(feature_gap(f1, real_dict, std))

        # T2: learned twin
        pred_y = y_pred_all[i] - y_pred_all[i].mean()
        f2 = extract_features(pred_y, window_slice(pred_y), waveform)
        gaps[waveform]["T2"].append(feature_gap(f2, real_dict, std))

    print()
    print(f"{'waveform':<10}{'n':>4}{'T0':>10}{'T1':>10}{'T2':>10}")
    all_gaps = {"T0": [], "T1": [], "T2": []}
    for wf in WAVEFORMS:
        n = len(gaps[wf]["T0"])
        row = [np.nanmean(gaps[wf][k]) for k in ("T0", "T1", "T2")]
        for k in ("T0", "T1", "T2"):
            all_gaps[k].extend(gaps[wf][k])
        print(f"{wf:<10}{n:>4}{row[0]:>10.3f}{row[1]:>10.3f}{row[2]:>10.3f}")
    overall = [np.nanmean(all_gaps[k]) for k in ("T0", "T1", "T2")]
    print(f"{'OVERALL':<10}{len(all_gaps['T0']):>4}{overall[0]:>10.3f}{overall[1]:>10.3f}{overall[2]:>10.3f}")

    # -- tau comparison: T2's learned effective decay vs T1's fitted tau vs literature default --
    print()
    print("Effective time constant comparison (real seconds, dt=1ms native sampling):")
    print(f"  literature-default tau (physics_model.DEFAULT_TAU_S): {DEFAULT_TAU_S*1e6:.1f} us")
    for wf in WAVEFORMS:
        print(f"  T1-fitted tau [{wf}]: {t1_params[wf].tau_s*1e6:.1f} us")

    dt_real = 0.001  # native sampling period, seconds
    with torch.no_grad():
        u_all = torch.from_numpy(u_test).float()
        x = model.in_proj(u_all.unsqueeze(-1))
        taus_per_layer = []
        for layer, norm in zip(model.layers, model.norms):
            xn = norm(x)
            A = -torch.exp(layer.A_log)                     # (D, N)
            delta = torch.nn.functional.softplus(layer.dt_proj(xn))  # (batch, T, D)
            Abar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (batch,T,D,N)
            Abar_mean = Abar.mean(dim=(0, 1))                # (D, N)
            tau_eff = -dt_real / torch.log(Abar_mean.clamp(min=1e-6))  # (D, N), seconds
            taus_per_layer.append(tau_eff)
            x = x + layer(xn)
    all_taus = torch.cat([t.flatten() for t in taus_per_layer])
    finite = all_taus[torch.isfinite(all_taus) & (all_taus > 0)]
    print(f"  T2 learned effective tau across all channels/states: "
          f"median={finite.median()*1e6:.1f}us, "
          f"IQR=[{finite.quantile(0.25)*1e6:.1f}, {finite.quantile(0.75)*1e6:.1f}]us "
          f"(n={len(finite)} channel-state units)")


if __name__ == "__main__":
    main()
