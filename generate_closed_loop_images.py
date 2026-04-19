#!/usr/bin/env python3
"""
generate_closed_loop_images.py
-------------------------------
Convert closed-loop piezo actuator raw .txt logs into PNG images for
MambaVision / ViT classification.

Image style matches the existing open-loop dataset exactly:
  • Fixed window indices 480–740 (260 samples) for every file
  • Wide landscape figure  figsize=(10, 5), dpi=150
  • Y-axis : "Relative Displacement (nm)"  (raw_to_nm, relative mode)
  • X-axis : "Time (ms)"  with actual time values (480 ms – 740 ms)
  • Title  : "{stem} - {label} (Indices 480-740)"
  • Light grid, blue line, linewidth=0.7

Noise class is filled with PID-physics synthetic recordings so all five
classes have equal sample counts.

Usage:
    python generate_closed_loop_images.py

Output:
    open_close_data/close_images/
        train / val / test  ×  noise / pulse / ramp / sine / square
    open_close_data/close_images/metadata.csv
"""

import csv
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent
CLOSE_RAW = ROOT / "open_close_data" / "close"
OUTPUT    = ROOT / "open_close_data" / "close_images"

CLASSES   = ["noise", "pulse", "ramp", "sine", "square"]

# Window positions — all 260 samples wide, matching open-loop style.
# Multiple windows per file give more training variety without changing style.
WIN_WIDTH    = 260
WIN_STARTS   = [480, 1000, 2000, 3500]   # 4 windows per file

# Figure style — matches open-loop images
FIG_W     = 10           # inches
FIG_H     = 5            # inches
DPI       = 150

# raw_to_nm: HeNe laser, relative mode (same as process_raw_8.py)
WAVELENGTH_NM = 632.991372
# nm = (D - D[0]) * (WAVELENGTH_NM / 8.0)

# Augmentation (train only)  — operate in nm space
N_AUG         = 4               # copies per original  (+1 original = 5 total)
AMP_JITTER    = (0.85, 1.15)    # amplitude scale range
DC_JITTER     = 0.05            # DC offset as fraction of signal range
NOISE_FRAC    = 0.02            # Gaussian noise as fraction of std

# Synthetic data — PID simulation for all 5 classes
N_SYNTHETIC      = 300          # synthetic files per class
SYNTH_DURATION_S = 5.0
SYNTH_FS         = 1000

# Dataset split
TRAIN_RATIO  = 0.70
RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_txt(filepath: Path):
    """
    Parse a closed-loop .txt log and return (fs, d_array).
    d_array contains raw integer D-values for the whole recording.
    """
    fs   = 1000.0
    data = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Sample Frequency"):
                parts = line.split()
                for i in range(len(parts) - 2):
                    if parts[i + 1] == "=" and parts[i] == "Frequency" and \
                       i > 0 and parts[i - 1] == "Sample":
                        try:
                            fs = float(parts[i + 2])
                        except ValueError:
                            pass
            elif line.startswith("D:"):
                try:
                    data.append(float(line.split()[0][2:]))
                except (IndexError, ValueError):
                    pass

    return fs, np.array(data, dtype=np.float64)


def raw_to_nm(d_array: np.ndarray) -> np.ndarray:
    """Relative displacement in nm.  Reference = first sample of the recording."""
    return (d_array - d_array[0]) * (WAVELENGTH_NM / 8.0)


def get_windows(nm_array: np.ndarray) -> list:
    """
    Extract all WIN_STARTS windows (each WIN_WIDTH samples wide).
    Skips any window whose start position exceeds the recording length.
    Returns at least one window (zero-padded if necessary).
    """
    windows = []
    for start in WIN_STARTS:
        end = start + WIN_WIDTH
        if start >= len(nm_array):
            continue
        if end <= len(nm_array):
            windows.append(nm_array[start:end].copy())
        else:
            # zero-pad tail
            w = np.zeros(WIN_WIDTH)
            avail = len(nm_array) - start
            w[:avail] = nm_array[start:]
            windows.append(w)
    if not windows:                          # recording too short — use start
        w = np.zeros(WIN_WIDTH)
        avail = min(WIN_WIDTH, len(nm_array))
        w[:avail] = nm_array[:avail]
        windows.append(w)
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation  (operates on nm window)
# ─────────────────────────────────────────────────────────────────────────────

def augment(window: np.ndarray, rng: np.random.Generator):
    """Return N_AUG augmented copies (does not include original)."""
    copies = []
    for _ in range(N_AUG):
        w = window.copy()
        w = w * rng.uniform(*AMP_JITTER)
        w = w + rng.uniform(-np.ptp(w) * DC_JITTER, np.ptp(w) * DC_JITTER)
        std = np.std(w)
        if std > 0:
            w = w + rng.normal(0, std * NOISE_FRAC, len(w))
        copies.append(w)
    return copies


# ─────────────────────────────────────────────────────────────────────────────
# Image saving  (matches open-loop style exactly)
# ─────────────────────────────────────────────────────────────────────────────

def save_image(window: np.ndarray, fs: float,
               stem: str, label: str, path: Path,
               win_start: int = WIN_STARTS[0]):
    """Save one waveform image in open-loop style."""
    n    = len(window)
    win_end = win_start + n
    t_ms = (np.arange(n) + win_start) / fs * 1000.0   # actual time in ms

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.plot(t_ms, window, linewidth=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Relative Displacement (nm)")
    ax.set_title(f"{stem} - {label} (Indices {win_start}-{win_end})")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic noise  (PID + second-order piezo plant model)
# ─────────────────────────────────────────────────────────────────────────────

def _pid_closed_loop_tf(Kp, Ki, Kd, wn, zeta, fs):
    G_num = [wn ** 2]
    G_den = [1.0, 2 * zeta * wn, wn ** 2]
    C_num = [Kd, Kp, Ki]
    C_den = [1.0, 0.0]
    CG_num = np.polymul(C_num, G_num)
    CG_den = np.polymul(C_den, G_den)
    L = max(len(CG_num), len(CG_den))
    T_num = np.pad(CG_num, (L - len(CG_num), 0))
    T_den = np.pad(CG_num, (L - len(CG_num), 0)) + np.pad(CG_den, (L - len(CG_den), 0))
    dt = scipy_signal.TransferFunction(T_num, T_den).to_discrete(1.0 / fs, method="bilinear")
    return dt


def _simulate_pid(reference: np.ndarray, rng: np.random.Generator,
                   d_range_target: tuple) -> np.ndarray:
    """
    Feed `reference` through a randomised PID closed-loop system.
    Returns raw D-values calibrated to d_range_target (min, max) counts.
    """
    n  = len(reference)
    Kp = rng.uniform(1, 4);  Ki = rng.uniform(20, 100);  Kd = rng.uniform(0.005, 0.02)
    wn = rng.uniform(150, 300);  zeta = rng.uniform(0.4, 0.8)
    try:
        sys_d = _pid_closed_loop_tf(Kp, Ki, Kd, wn, zeta, SYNTH_FS)
        _, y  = scipy_signal.dlsim(sys_d, reference)
        y     = y.flatten()
    except Exception:
        b, a = scipy_signal.butter(4, [5 / 500, 200 / 500], btype="band")
        y    = scipy_signal.filtfilt(b, a, reference)

    span = np.ptp(y)
    if span > 0:
        target = rng.uniform(*d_range_target)
        y = y / span * target
    y += rng.uniform(-40000.0, -35000.0)
    return y.astype(np.float64)


def make_synthetic(cls: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one synthetic closed-loop recording for `cls` as raw D-values.
    Each waveform type uses a physically appropriate PID reference signal.
    Calibrated to match experimental D-value ranges.
    """
    n  = int(SYNTH_DURATION_S * SYNTH_FS)
    t  = np.arange(n) / SYNTH_FS

    if cls == "noise":
        # White noise disturbance through closed loop
        ref = rng.standard_normal(n) * 1000.0
        return _simulate_pid(ref, rng, d_range_target=(2.0, 5.0))

    elif cls == "sine":
        freq = rng.uniform(1, 200)
        ref  = np.sin(2 * np.pi * freq * t) * rng.uniform(500, 2000)
        return _simulate_pid(ref, rng, d_range_target=(5.0, 22.0))

    elif cls == "square":
        freq = rng.uniform(1, 200)
        ref  = scipy_signal.square(2 * np.pi * freq * t) * rng.uniform(500, 2000)
        return _simulate_pid(ref, rng, d_range_target=(8.0, 18.0))

    elif cls == "ramp":
        freq = rng.uniform(1, 200)
        ref  = scipy_signal.sawtooth(2 * np.pi * freq * t) * rng.uniform(500, 2000)
        return _simulate_pid(ref, rng, d_range_target=(6.0, 20.0))

    elif cls == "pulse":
        freq  = rng.uniform(1, 200)
        duty  = rng.uniform(0.05, 0.15)   # narrow pulse
        ref   = scipy_signal.square(2 * np.pi * freq * t, duty=duty)
        ref   = ref * rng.uniform(200, 1000)
        return _simulate_pid(ref, rng, d_range_target=(2.0, 5.0))

    else:
        raise ValueError(f"Unknown class: {cls}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def collect_samples(rng: np.random.Generator):
    """
    Returns dict {class: [(fs, win_start, nm_window, stem), ...]}
    Each experimental file yields multiple windows; synthetics add more variety.
    """
    samples = {c: [] for c in CLASSES}

    # Experimental files → multiple windows each
    print("\nLoading experimental data ...")
    min_len = WIN_STARTS[0] + WIN_WIDTH   # must be at least this long
    for cls in CLASSES:
        cls_dir = CLOSE_RAW / cls
        if not cls_dir.exists():
            print(f"  [warn] {cls_dir} not found")
            continue
        txt_files = sorted(cls_dir.glob("*.txt"))
        print(f"  {cls:8s}: {len(txt_files)} experimental files")
        for fpath in txt_files:
            try:
                fs, d = load_txt(fpath)
                if len(d) < min_len:
                    continue
                nm   = raw_to_nm(d)
                wins = get_windows(nm)
                for w_idx, win in enumerate(wins):
                    start = WIN_STARTS[w_idx] if w_idx < len(WIN_STARTS) else WIN_STARTS[0]
                    samples[cls].append((fs, start, win, fpath.stem))
            except Exception as e:
                print(f"    [error] {fpath.name}: {e}")

    # PID synthetic for all 5 classes
    print(f"\nGenerating {N_SYNTHETIC} synthetic clips per class ...")
    for cls in CLASSES:
        for i in range(N_SYNTHETIC):
            d    = make_synthetic(cls, rng)
            nm   = raw_to_nm(d)
            wins = get_windows(nm)
            for w_idx, win in enumerate(wins):
                start = WIN_STARTS[w_idx] if w_idx < len(WIN_STARTS) else WIN_STARTS[0]
                samples[cls].append((float(SYNTH_FS), start, win, f"synth_{cls}_{i:04d}"))
        print(f"  {cls:8s}: {len(samples[cls])} total sources")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Write splits
# ─────────────────────────────────────────────────────────────────────────────

def write_split(records, split, meta_rows, rng):
    is_train = (split == "train")
    total = 0

    for cls, fs, win_start, win, stem in records:
        out_dir = OUTPUT / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        win_end = win_start + WIN_WIDTH
        tag     = f"w{win_start}"   # encode window position in filename

        # Original
        fname = f"{stem}_{tag}_absolute_orig.png"
        save_image(win, fs, stem, "Original", out_dir / fname, win_start)
        meta_rows.append([split, cls, fname, stem, win_start, "original"])
        total += 1

        # Augmented (train only)
        if is_train:
            for i, aug_win in enumerate(augment(win, rng), start=1):
                fname = f"{stem}_{tag}_absolute_aug_v{i}.png"
                save_image(aug_win, fs, stem, f"Aug v{i}", out_dir / fname, win_start)
                meta_rows.append([split, cls, fname, stem, win_start, f"aug_v{i}"])
                total += 1

    return total


def main():
    print("=" * 70)
    print("  Closed-Loop Image Generator")
    print("=" * 70)
    print(f"  Windows     : {[f'{s}-{s+WIN_WIDTH}' for s in WIN_STARTS]}  ({WIN_WIDTH} samples each)")
    print(f"  Figure      : {FIG_W}×{FIG_H} in @ {DPI} dpi")
    print(f"  Augmentations: {N_AUG} per original (train only)")
    print(f"  Output      : {OUTPUT}")

    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    samples = collect_samples(rng)

    # Flatten to records
    all_records = []
    for cls, srcs in samples.items():
        for fs, win_start, win, stem in srcs:
            all_records.append((cls, fs, win_start, win, stem))

    # Stratified split
    print("\nSplitting dataset ...")
    by_class = {c: [] for c in CLASSES}
    for rec in all_records:
        by_class[rec[0]].append(rec)   # rec[0] = cls

    train_r, val_r, test_r = [], [], []
    for cls, recs in by_class.items():
        rng.shuffle(recs)
        tr, tmp = train_test_split(recs, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED)
        vl, te  = train_test_split(tmp,  test_size=0.5,               random_state=RANDOM_SEED)
        train_r += tr; val_r += vl; test_r += te

    print(f"  Records → train:{len(train_r)}  val:{len(val_r)}  test:{len(test_r)}")

    meta_rows = [["split", "class", "filename", "source_stem", "win_start", "augmentation"]]

    print("\nGenerating images ...")
    n_tr = write_split(train_r, "train", meta_rows, rng)
    n_vl = write_split(val_r,   "val",   meta_rows, rng)
    n_te = write_split(test_r,  "test",  meta_rows, rng)

    meta_path = OUTPUT / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        csv.writer(f).writerows(meta_rows)

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)
    print(f"\n  Total images : {n_tr + n_vl + n_te}")
    print(f"    train : {n_tr}   val : {n_vl}   test : {n_te}")

    print("\n  Per-class image counts:")
    for split in ["train", "val", "test"]:
        counts = {cls: len(list((OUTPUT / split / cls).glob("*.png"))) for cls in CLASSES}
        print(f"  [{split}]  " + "  ".join(f"{c}:{counts[c]}" for c in CLASSES))

    print(f"\n  Metadata : {meta_path}")


if __name__ == "__main__":
    main()
