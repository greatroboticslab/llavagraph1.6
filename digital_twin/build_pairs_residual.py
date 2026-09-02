"""
build_pairs_residual.py
=========================
Builds (ideal, real) pairs for the sim-to-real correction model: `ideal`
is the physics simulator's output, `real` is the actual measured
displacement. The task this dataset trains: take the physically
plausible but imperfect simulated waveform and learn what to change
about it so it looks like this real device's actual output.

`ideal` uses the Stage 2 shared-calibrated parameters
(stage2_fitted_params.json, from stage2_calibrate_shared.py) if present
-- i.e. T1-v2, not bare T0 defaults -- since the physics-deepening plan's
point is for the learned correction to sit on top of the best physics
can do, not to re-learn things a classical fit already explains. Falls
back to T0 defaults with a warning if Stage 2 hasn't been run yet.

This is a different (and more directly useful) framing than
build_pairs.py's (command, real) pairs: there, the model has to learn
the ENTIRE command->response mapping from a normalized command signal
that carries no information about hysteresis or saturation. Here, the
simulator already supplies a physically-informed starting point -- the
model only has to learn the residual on top of it, which is both an
easier learning problem and a more literal instance of "physics does
most of the work, the learned part corrects what physics gets wrong."

Uses the canonical split (stage_split.json, from make_split.py) so
results are directly comparable with stage2_calibrate_shared.py's
train/test membership -- Stage 2 calibrates on TRAIN only, so evaluating
the resulting T1-v2 (or a correction trained on top of it) on TEST is a
fair, non-leaked comparison.
"""

import json
from pathlib import Path

import numpy as np

from feature_extract import real_measurements
import feature_extract as fe
from physics_model import TwinParams, simulate
from signal_align import align_phase

FS_OUT = 1000.0
WINDOW_S = 0.26
N_SAMPLES = int(WINDOW_S * FS_OUT)
_INCLUDED_WAVEFORMS = ["sine", "square", "ramp", "pulse"]


def _load_base_params():
    """T1-v2 if Stage 2 has been run, else bare T0 defaults."""
    fitted_path = Path("stage2_fitted_params.json")
    if fitted_path.exists():
        fitted = json.load(open(fitted_path))["fitted"]
        print(f"build_pairs_residual: using Stage-2 shared-calibrated parameters "
              f"from {fitted_path} as the 'ideal' base (T1-v2).")
        return lambda waveform: TwinParams(waveform=waveform, seed=0, **fitted)
    print("build_pairs_residual: stage2_fitted_params.json not found -- "
          "falling back to bare T0 defaults. Run stage2_calibrate_shared.py first "
          "for the intended (T1-v2-based) dataset.")
    return lambda waveform: TwinParams(waveform=waveform, seed=0)


def build_dataset(split_file: str = "stage_split.json"):
    split = json.load(open(split_file))
    split_of_filename = {}
    for split_name, items in split.items():
        for it in items:
            split_of_filename[it["filename"]] = split_name

    base_params_fn = _load_base_params()
    splits = {"train": [], "val": [], "test": []}

    for waveform in _INCLUDED_WAVEFORMS:
        df = real_measurements(waveform)
        df = df[df["path"].notna()]
        for row in df.itertuples():
            split_name = split_of_filename.get(row.filename)
            if split_name is None:
                continue  # not part of the canonical split (shouldn't happen)

            sig, win = fe.bfe.load_signal(str(row.path))
            if len(win) < N_SAMPLES // 2:
                continue
            y_real = win[:N_SAMPLES]
            if len(y_real) < N_SAMPLES:
                y_real = np.pad(y_real, (0, N_SAMPLES - len(y_real)), mode="edge")

            freq_hz = float(row.dominant_freq_hz) if row.dominant_freq_hz and row.dominant_freq_hz > 0 else 1.0
            p = base_params_fn(waveform)
            p.freq_hz = freq_hz
            p.duration_s = WINDOW_S
            r = simulate(p)
            y_ideal = r["y_nm"][:N_SAMPLES]
            if len(y_ideal) < N_SAMPLES:
                y_ideal = np.pad(y_ideal, (0, N_SAMPLES - len(y_ideal)), mode="edge")

            y_ideal = align_phase(y_ideal, y_real, freq_hz, FS_OUT)

            splits[split_name].append({
                "waveform": waveform, "freq_hz": freq_hz, "filename": row.filename,
                "ideal": y_ideal.astype(np.float32), "real": y_real.astype(np.float32),
            })

    return splits


def save_dataset(splits: dict, out_dir: str = "pairs_data_residual"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    meta = {}
    for split_name, items in splits.items():
        ideal = np.stack([e["ideal"] for e in items])
        real = np.stack([e["real"] for e in items])
        np.savez(out / f"{split_name}.npz", ideal=ideal, real=real)
        meta[split_name] = [
            {"waveform": e["waveform"], "freq_hz": e["freq_hz"], "filename": e["filename"]}
            for e in items
        ]
        counts = {wf: sum(1 for e in items if e["waveform"] == wf) for wf in _INCLUDED_WAVEFORMS}
        print(f"{split_name}: {len(items)} examples ({', '.join(f'{k}={v}' for k, v in counts.items())})")
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    splits = build_dataset()
    save_dataset(splits)
