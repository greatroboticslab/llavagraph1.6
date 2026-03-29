"""
================================================================================
  CSV → Augmented Waveform Images Pipeline
  For: Time-Domain Piezo Displacement Measurement → ViT Classification
  Classes : sine, square, noise, ramp, pulse

  Augmentation Strategy: Gaussian noise × 4 levels
  Output per CSV:
      _orig.png       — clean original render
      _aug_v1.png     — very light noise  (σ = 1% of signal std)
      _aug_v2.png     — light noise       (σ = 3% of signal std)
      _aug_v3.png     — medium noise      (σ = 5% of signal std)
      _aug_v4.png     — stronger noise    (σ = 8% of signal std)
  Total: 5 images per CSV file

  Figure style matches your original plotting script exactly:
      figsize=(10, 6), linewidth=1.5, color='#1f77b4',
      dpi=300, grid alpha=0.3, Time_ms x-axis,
      Absolute_Displacement_nm y-axis, index slice [START:END]
================================================================================

Expected folder structure:
    images_Timedomain/
        sine/     *.csv
        square/   *.csv
        noise/    *.csv
        ramp/     *.csv
        pulse/    *.csv

Output folder structure:
    images_Timedomain_augmented/
        sine/     *.png
        square/   *.png
        noise/    *.png
        ramp/     *.png
        pulse/    *.png

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

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  — edit these to match your setup
# ─────────────────────────────────────────────────────────────

INPUT_ROOT  = "/Users/ilminurablikim/Desktop/csv2.20"           # Root folder with 5 class sub-folders
OUTPUT_ROOT = "/Users/ilminurablikim/Desktop/images_Timedomain_augmented" # Output root folder
CLASSES     = ["sine", "square", "noise", "ramp", "pulse"]

# ── Index slice (same as your original script) ──────────────
# Only this portion of each CSV is plotted, ~200 points
START_IDX = 480
END_IDX   = 740

# ── Column names (must match your CSV headers) ───────────────
TIME_COL        = "Time_ms"
DISPLACEMENT_COL = "Absolute_Displacement_nm"

# ── Figure style (identical to your original script) ─────────
FIG_SIZE    = (10, 6)       # inches
LINE_WIDTH  = 1.5
LINE_COLOR  = "#1f77b4"     # matplotlib default blue
GRID_ALPHA  = 0.3
IMG_DPI     = 300

# ── Noise levels for augmentation ────────────────────────────
NOISE_LEVELS = [0.01, 0.03, 0.05, 0.08]   # fraction of signal std

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
#  CSV LOADER  — extracts slice [START_IDX : END_IDX]
# ─────────────────────────────────────────────────────────────

def load_signal_slice(filepath: str):
    """
    Load Time_ms and Absolute_Displacement_nm columns,
    then slice to [START_IDX : END_IDX].
    Returns (displacement, time_ms) as float32 arrays.
    """
    df = pd.read_csv(filepath)

    # Validate columns
    for col in [TIME_COL, DISPLACEMENT_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {os.path.basename(filepath)}.\n"
                f"  Available columns: {list(df.columns)}"
            )

    total_pts = len(df)
    s = START_IDX
    e = END_IDX

    # Adjust if file is shorter than expected
    if total_pts < e:
        print(f"    [WARN] {os.path.basename(filepath)} has only {total_pts} pts "
              f"(expected {e}). Adjusting end to {total_pts}.")
        e = total_pts
    if s >= e:
        s = 0

    time_ms      = df[TIME_COL].values[s:e].astype(np.float32)
    displacement = df[DISPLACEMENT_COL].values[s:e].astype(np.float32)

    # Drop NaNs (keep paired)
    mask = ~(np.isnan(time_ms) | np.isnan(displacement))
    time_ms      = time_ms[mask]
    displacement = displacement[mask]

    if len(displacement) == 0:
        raise ValueError(f"No valid data in slice [{s}:{e}] for {filepath}")

    return displacement, time_ms, s, e


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION  — Gaussian noise, 4 levels
# ─────────────────────────────────────────────────────────────

def add_gaussian_noise(signal: np.ndarray, noise_fraction: float) -> np.ndarray:
    """
    Add Gaussian noise scaled to signal std.
    noise_fraction = 0.03 → sigma_noise = 0.03 × sigma_signal.
    Only the displacement values are augmented; Time_ms stays unchanged.
    """
    sigma = noise_fraction * signal.std()
    noise = np.random.normal(0.0, sigma, size=signal.shape)
    return (signal + noise).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  RENDERER  — matches your original plot_original_data() exactly
# ─────────────────────────────────────────────────────────────

def render_to_image(displacement: np.ndarray,
                    time_ms: np.ndarray,
                    save_path: str,
                    title: str,
                    slice_start: int,
                    slice_end: int):
    """
    Render waveform image with the exact same style as your original script:
      - figsize=(10, 6)
      - linewidth=1.5, color='#1f77b4'
      - title, xlabel 'Time (ms)', ylabel 'Displacement (nm)'
      - grid alpha=0.3
      - dpi=300, bbox_inches='tight'
    """
    plt.figure(figsize=FIG_SIZE)
    plt.plot(time_ms, displacement,
             linewidth=LINE_WIDTH,
             color=LINE_COLOR)

    plt.title(f"{title} (Indices {slice_start}-{slice_end})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Displacement (nm)")
    plt.grid(True, alpha=GRID_ALPHA)
    plt.tight_layout()

    plt.savefig(save_path, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def process_class(class_name: str, csv_files: list, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n_success = 0

    for filepath in csv_files:
        stem = os.path.splitext(os.path.basename(filepath))[0]

        try:
            displacement, time_ms, s, e = load_signal_slice(filepath)
        except Exception as ex:
            print(f"    [SKIP] {os.path.basename(filepath)} — {ex}")
            continue

        # ── 1) Original ───────────────────────────────────────
        render_to_image(
            displacement, time_ms,
            save_path = os.path.join(out_dir, f"{stem}_orig.png"),
            title     = f"{stem} - Original",
            slice_start=s, slice_end=e
        )

        # ── 2-5) Augmented: noise at 4 different levels ───────
        noise_labels = ["VeryLight(1%)", "Light(3%)", "Medium(5%)", "Strong(8%)"]
        for vi, (level, label) in enumerate(zip(NOISE_LEVELS, noise_labels), start=1):
            aug_displacement = add_gaussian_noise(displacement, noise_fraction=level)
            render_to_image(
                aug_displacement, time_ms,
                save_path  = os.path.join(out_dir, f"{stem}_aug_v{vi}.png"),
                title      = f"{stem} - Noise {label}",
                slice_start=s, slice_end=e
            )

        n_success += 1

    total_imgs = n_success * 5
    print(f"  [{class_name:8s}]  {n_success}/{len(csv_files)} CSV(s) -> "
          f"{total_imgs} images  (1 orig + 4 aug each)")
    return total_imgs


def main():
    print("=" * 62)
    print("  Piezo Waveform  CSV -> Augmented Images")
    print(f"  Index slice   : [{START_IDX} : {END_IDX}]  (~{END_IDX-START_IDX} pts)")
    print(f"  Figure style  : figsize={FIG_SIZE}, dpi={IMG_DPI}, "
          f"lw={LINE_WIDTH}, color='{LINE_COLOR}'")
    print("  Augmentation  : Gaussian noise x 4 levels")
    print("  Output per CSV: 5 images  (1 orig + 4 aug)")
    print("=" * 62)

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
            print(f"  [{class_name:8s}]  No CSV files found in '{in_dir}' — skipped")
            continue

        n_imgs    = process_class(class_name, csv_files, out_dir)
        grand_csv += len(csv_files)
        grand_img += n_imgs

    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    for class_name in CLASSES:
        n_csv = len(glob.glob(os.path.join(INPUT_ROOT,  class_name, "*.csv")))
        n_img = len(glob.glob(os.path.join(OUTPUT_ROOT, class_name, "*.png")))
        print(f"  {class_name:8s} : {n_csv:4d} CSV  ->  {n_img:6d} images")
    print("-" * 62)
    print(f"  {'TOTAL':8s} : {grand_csv:4d} CSV  ->  {grand_img:6d} images")
    print("=" * 62)
    print(f"\n  Saved to: {os.path.abspath(OUTPUT_ROOT)}/")
    print("  Ready for ViT ImageFolder training.\n")


if __name__ == "__main__":
    main()
