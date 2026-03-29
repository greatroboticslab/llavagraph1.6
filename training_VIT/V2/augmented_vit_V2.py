"""
================================================================================
  CSV → Augmented Waveform Images Pipeline
  For: Time-Domain Piezo Displacement Measurement → ViT Classification
  Classes : sine, square, noise, ramp, pulse

  Augmentation: Gaussian noise ONLY
  ──────────────────────────────────
  Chosen as the single safest augmentation for all five waveform classes.
  It adds vertical jitter without bending, stretching, or reordering any
  geometric feature:
    - Ramp  : linear slope stays linear
    - Square: sharp edges and flat plateaus stay intact
    - Sine  : smooth curves stay smooth
    - Pulse : narrow spike stays narrow
    - Noise : random character stays random

  Output per CSV (5 images):
      _orig.png    — detrended original
      _aug_v1.png  — noise σ = 0.5% of signal std  (barely visible)
      _aug_v2.png  — noise σ = 1.5% of signal std  (very light)
      _aug_v3.png  — noise σ = 3.0% of signal std  (light, realistic)
      _aug_v4.png  — noise σ = 5.0% of signal std  (moderate)

  Figure style: figsize=(10,6), lw=1.5, color='#1f77b4', dpi=300, grid 0.3
================================================================================

Folder structure expected:
    images_Timedomain/
        sine/   square/   noise/   ramp/   pulse/   <- each contains *.csv

Output:
    images_Timedomain_augmented/
        sine/   square/   noise/   ramp/   pulse/   <- each contains *.png

Install:
    pip install numpy pandas matplotlib
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
#  CONFIGURATION — edit these to match your setup
# ------------------------------------------------------------------

INPUT_ROOT  = "/Users/ilminurablikim/Desktop/csv_all"
OUTPUT_ROOT = "/Users/ilminurablikim/Desktop/images_Timedomain_augmented"
CLASSES     = ["sine", "square", "noise", "ramp", "pulse"]

# Index slice: only this portion of each CSV is plotted (~260 pts)
# Run diagnose_index_range.py first if you are unsure of a clean window
START_IDX = 480
END_IDX   = 740

TIME_COL         = "Time_ms"
DISPLACEMENT_COL = "Absolute_Displacement_nm"

# Figure style — identical to your original plotting script
FIG_SIZE   = (10, 6)
LINE_WIDTH = 1.5
LINE_COLOR = "#1f77b4"
GRID_ALPHA = 0.3
IMG_DPI    = 300

# 4 noise levels as fraction of signal std
# Kept gentle so waveform shape stays clearly visible in every image
NOISE_LEVELS = [0.005, 0.015, 0.030, 0.050]
NOISE_LABELS = ["Noise 0.5%", "Noise 1.5%", "Noise 3.0%", "Noise 5.0%"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------
#  CSV LOADER + DETREND
# ------------------------------------------------------------------

def load_signal_slice(filepath):
    """
    Load slice [START_IDX:END_IDX] then subtract mean to remove the
    large absolute DC offset (~1e10 nm), revealing the oscillation shape.
    """
    df = pd.read_csv(filepath)

    for col in [TIME_COL, DISPLACEMENT_COL]:
        if col not in df.columns:
            raise ValueError(
                "Column '{}' not found in {}. Available: {}".format(
                    col, os.path.basename(filepath), list(df.columns)))

    s, e = START_IDX, END_IDX
    if len(df) < e:
        print("    [WARN] {}: only {} rows, adjusting end {} -> {}".format(
              os.path.basename(filepath), len(df), e, len(df)))
        e = len(df)
    if s >= e:
        s = 0

    time_ms      = df[TIME_COL].values[s:e].astype(np.float64)
    displacement = df[DISPLACEMENT_COL].values[s:e].astype(np.float64)

    mask = ~(np.isnan(time_ms) | np.isnan(displacement))
    time_ms      = time_ms[mask]
    displacement = displacement[mask]

    if len(displacement) == 0:
        raise ValueError("No valid data in slice [{}:{}]".format(s, e))

    # Subtract mean: removes DC offset, waveform shape becomes visible
    displacement = displacement - displacement.mean()

    return displacement.astype(np.float32), time_ms.astype(np.float32), s, e


# ------------------------------------------------------------------
#  AUGMENTATION
# ------------------------------------------------------------------

def add_gaussian_noise(signal, noise_fraction):
    """
    Add Gaussian noise scaled to the signal's own std.
    noise_fraction=0.03 means sigma_noise = 0.03 x sigma_signal.
    """
    sigma = noise_fraction * signal.std()
    if sigma < 1e-12:
        sigma = 1e-3          # guard for near-flat signals
    noise = np.random.normal(0.0, sigma, size=signal.shape)
    return (signal + noise).astype(np.float32)


# ------------------------------------------------------------------
#  RENDERER
# ------------------------------------------------------------------

def render_to_image(displacement, time_ms, save_path, title, s, e):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(time_ms, displacement, linewidth=LINE_WIDTH, color=LINE_COLOR)
    plt.title("{} (Indices {}-{})".format(title, s, e))
    plt.xlabel("Time (ms)")
    plt.ylabel("Relative Displacement (nm)")
    plt.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(save_path, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
#  PER-CLASS PROCESSING
# ------------------------------------------------------------------

def process_class(class_name, csv_files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_success = 0

    for filepath in csv_files:
        stem = os.path.splitext(os.path.basename(filepath))[0]

        try:
            displacement, time_ms, s, e = load_signal_slice(filepath)
        except Exception as ex:
            print("    [SKIP] {} -- {}".format(os.path.basename(filepath), ex))
            continue

        # 1) Original (detrended, no augmentation)
        render_to_image(
            displacement, time_ms,
            save_path = os.path.join(out_dir, "{}_orig.png".format(stem)),
            title     = "{} - Original".format(stem),
            s=s, e=e
        )

        # 2-5) Gaussian noise at 4 increasing levels
        for vi, (level, label) in enumerate(
                zip(NOISE_LEVELS, NOISE_LABELS), start=1):
            aug = add_gaussian_noise(displacement, noise_fraction=level)
            render_to_image(
                aug, time_ms,
                save_path = os.path.join(
                    out_dir, "{}_aug_v{}.png".format(stem, vi)),
                title     = "{} - {}".format(stem, label),
                s=s, e=e
            )

        n_success += 1

    total_imgs = n_success * 5
    print("  [{:8s}]  {}/{} CSV(s) -> {} images  (1 orig + 4 noise)".format(
          class_name, n_success, len(csv_files), total_imgs))
    return total_imgs


# ------------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Piezo Waveform  CSV -> Augmented Images")
    print("  Method        : Gaussian noise only")
    print("  Noise levels  : 0.5% / 1.5% / 3.0% / 5.0% of signal std")
    print("  Index slice   : [{} : {}]  (~{} pts)".format(
          START_IDX, END_IDX, END_IDX - START_IDX))
    print("  Detrend       : subtract slice mean  (removes DC offset)")
    print("  Output per CSV: 5 images  (1 orig + 4 aug)")
    print("=" * 64)

    grand_csv = 0
    grand_img = 0

    for class_name in CLASSES:
        in_dir  = os.path.join(INPUT_ROOT,  class_name)
        out_dir = os.path.join(OUTPUT_ROOT, class_name)

        csv_files = sorted(
            glob.glob(os.path.join(in_dir, "*.csv")) +
            glob.glob(os.path.join(in_dir, "*.CSV"))
        )

        if not csv_files:
            print("  [{:8s}]  No CSVs in '{}' -- skipped".format(
                  class_name, in_dir))
            continue

        n_imgs     = process_class(class_name, csv_files, out_dir)
        grand_csv += len(csv_files)
        grand_img += n_imgs

    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for class_name in CLASSES:
        n_csv = len(glob.glob(os.path.join(INPUT_ROOT,  class_name, "*.csv")))
        n_img = len(glob.glob(os.path.join(OUTPUT_ROOT, class_name, "*.png")))
        print("  {:8s} : {:4d} CSV  ->  {:6d} images".format(
              class_name, n_csv, n_img))
    total_csv = sum(
        len(glob.glob(os.path.join(INPUT_ROOT, c, "*.csv"))) for c in CLASSES)
    total_img = sum(
        len(glob.glob(os.path.join(OUTPUT_ROOT, c, "*.png"))) for c in CLASSES)
    print("-" * 64)
    print("  {:8s} : {:4d} CSV  ->  {:6d} images".format(
          "TOTAL", total_csv, total_img))
    print("=" * 64)
    print("\n  Saved to: {}/".format(os.path.abspath(OUTPUT_ROOT)))
    print("  Ready for ViT ImageFolder training.\n")


if __name__ == "__main__":
    main()
