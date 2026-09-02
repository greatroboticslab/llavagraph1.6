"""
eval_mamba_twin.py
====================
Closes the loop the digital_twin/README.md roadmap promised: evaluates
T2 (the trained MambaTwin) on the held-out test set using the exact same
feature-based gap score used everywhere else in this project (calibrate.
feature_gap against feature_extract.real_feature_stats()), and reports
it side by side with T0 (the analytical FOPDT+hysteresis twin, default/
uncalibrated parameters) on the SAME test measurements -- the actual
answer to "does a learned SSM close the sim-to-real gap better than the
physics-only twin".
"""

import json
from pathlib import Path

import numpy as np
import torch

from mamba_twin_model import MambaTwin
from feature_extract import extract_features, window_slice, real_feature_stats
from physics_model import TwinParams, simulate
from calibrate import feature_gap

DATA_DIR = Path("pairs_data")
CKPT_PATH = Path("mamba_twin.pt")


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = MambaTwin(d_model=ckpt["d_model"], d_state=ckpt["d_state"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["y_scale"]


def main():
    model, y_scale = load_model()
    d = np.load(DATA_DIR / "test.npz")
    u_test, y_test = d["u"], d["y"]
    meta = json.load(open(DATA_DIR / "meta.json"))["test"]

    stats = real_feature_stats()

    def std_lookup(waveform):
        def f(k):
            try:
                return stats.loc[waveform, (k, "std")]
            except KeyError:
                return None
        return f

    def mean_lookup(waveform):
        def f(k):
            try:
                return stats.loc[waveform, (k, "mean")]
            except KeyError:
                return None
        return f

    with torch.no_grad():
        u_t = torch.from_numpy(u_test).float()
        y_pred = model(u_t).numpy() * y_scale

    t2_gaps, t0_gaps = [], []
    per_wf_t2, per_wf_t0 = {}, {}

    for i, m in enumerate(meta):
        waveform, freq_hz = m["waveform"], m["freq_hz"]
        real_y = y_test[i]
        real_feat = extract_features(real_y, window_slice(real_y), waveform)
        sl = std_lookup(waveform)
        ml = mean_lookup(waveform)
        real_dict = {k: ml(k) for k in real_feat}

        # T2: learned twin's prediction
        pred_y = y_pred[i] - y_pred[i].mean()
        pred_feat = extract_features(pred_y, window_slice(pred_y), waveform)
        g_t2 = feature_gap(pred_feat, real_dict, sl)

        # T0: analytical twin, default (uncalibrated) parameters, same freq/duration
        p = TwinParams(waveform=waveform, freq_hz=freq_hz, duration_s=len(real_y) / 1000.0)
        r = simulate(p)
        t0_feat = extract_features(r["y_nm"], window_slice(r["y_nm"]), waveform)
        g_t0 = feature_gap(t0_feat, real_dict, sl)

        t2_gaps.append(g_t2)
        t0_gaps.append(g_t0)
        per_wf_t2.setdefault(waveform, []).append(g_t2)
        per_wf_t0.setdefault(waveform, []).append(g_t0)

    print(f"{'waveform':<10}{'n':>4}{'T0 gap (analytical)':>22}{'T2 gap (learned)':>20}")
    for wf in per_wf_t0:
        n = len(per_wf_t0[wf])
        print(f"{wf:<10}{n:>4}{np.nanmean(per_wf_t0[wf]):>22.3f}{np.nanmean(per_wf_t2[wf]):>20.3f}")
    print(f"{'OVERALL':<10}{len(t0_gaps):>4}{np.nanmean(t0_gaps):>22.3f}{np.nanmean(t2_gaps):>20.3f}")


if __name__ == "__main__":
    main()
