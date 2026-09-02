"""
eval_mamba_correction.py
==========================
Evaluates the sim-to-real correction model on the held-out test set:
reports the feature-based gap score for "ideal (T0, uncorrected)" vs
"corrected (T0 + Mamba residual)" against real, and saves an overlay
plot of ideal / corrected / real for a handful of representative test
examples -- the direct visual answer to "does the corrected waveform
actually look more like the real one".
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from mamba_twin_model import MambaTwin
from feature_extract import extract_features, window_slice, real_feature_stats
from calibrate import feature_gap

DATA_DIR = Path("pairs_data_residual")
CKPT_PATH = Path("mamba_correction.pt")
WAVEFORMS = ["sine", "square", "ramp", "pulse"]


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = MambaTwin(d_model=ckpt["d_model"], d_state=ckpt["d_state"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["y_scale"]


def main():
    model, y_scale = load_model()
    stats = real_feature_stats()
    meta = json.load(open(DATA_DIR / "meta.json"))["test"]
    d = np.load(DATA_DIR / "test.npz")
    ideal_test, real_test = d["ideal"], d["real"]

    with torch.no_grad():
        ideal_t = torch.from_numpy(ideal_test).float() / y_scale
        corrected = (ideal_t + model(ideal_t)).numpy() * y_scale

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

    gaps = {wf: {"ideal": [], "corrected": []} for wf in WAVEFORMS}
    examples_for_plot = {}

    for i, m in enumerate(meta):
        wf, freq_hz = m["waveform"], m["freq_hz"]
        real_y = real_test[i]
        ideal_y = ideal_test[i]
        corr_y = corrected[i] - corrected[i].mean()

        real_feat = extract_features(real_y, window_slice(real_y), wf)
        real_dict = {k: mean_lookup(wf)(k) for k in real_feat}

        ideal_feat = extract_features(ideal_y, window_slice(ideal_y), wf)
        corr_feat = extract_features(corr_y, window_slice(corr_y), wf)

        g_ideal = feature_gap(ideal_feat, real_dict, std_lookup(wf))
        g_corr = feature_gap(corr_feat, real_dict, std_lookup(wf))
        gaps[wf]["ideal"].append(g_ideal)
        gaps[wf]["corrected"].append(g_corr)

        if wf not in examples_for_plot:
            examples_for_plot[wf] = (ideal_y, corr_y, real_y, freq_hz, m["filename"])

    print(f"{'waveform':<10}{'n':>4}{'ideal (T0)':>14}{'corrected':>14}")
    all_ideal, all_corr = [], []
    for wf in WAVEFORMS:
        n = len(gaps[wf]["ideal"])
        gi, gc = np.nanmean(gaps[wf]["ideal"]), np.nanmean(gaps[wf]["corrected"])
        all_ideal.extend(gaps[wf]["ideal"])
        all_corr.extend(gaps[wf]["corrected"])
        print(f"{wf:<10}{n:>4}{gi:>14.3f}{gc:>14.3f}")
    print(f"{'OVERALL':<10}{len(all_ideal):>4}{np.nanmean(all_ideal):>14.3f}{np.nanmean(all_corr):>14.3f}")

    fig, axes = plt.subplots(len(WAVEFORMS), 1, figsize=(9, 3 * len(WAVEFORMS)))
    t_ms = np.arange(ideal_test.shape[1]) / 1000.0 * 1000
    for ax, wf in zip(axes, WAVEFORMS):
        ideal_y, corr_y, real_y, freq_hz, fname = examples_for_plot[wf]
        ax.plot(t_ms, ideal_y - ideal_y.mean(), label="T0 ideal (uncorrected)", lw=1, alpha=0.7)
        ax.plot(t_ms, corr_y, label="Mamba-corrected", lw=1.3)
        ax.plot(t_ms, real_y - real_y.mean(), label="real", lw=1, alpha=0.8, linestyle="--")
        ax.set_title(f"{wf} @ {freq_hz:.1f} Hz ({fname})", fontsize=9)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("displacement (nm)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("correction_examples.png", dpi=130)
    print("saved correction_examples.png")


if __name__ == "__main__":
    main()
