"""
generate_images.py
------------------
Step 1 of the pipeline: Convert raw piezo waveform CSV files into PNG images.

Two representations are supported:
  • Time-domain  – amplitude vs. sample index
  • FFT          – magnitude spectrum (log scale) vs. frequency bin

Expected input directory structure:
    RAW_DATA_DIR/
        noise/    *.csv
        sine/     *.csv
        square/   *.csv
        pulse/    *.csv
        ramp/     *.csv

Output directory structure (mirrors input):
    IMAGE_DATA_DIR/
        time/
            noise/  *.png
            sine/   *.png
            ...
        fft/
            noise/  *.png
            sine/   *.png
            ...

Usage:
    python generate_images.py [--raw_dir PATH] [--img_dir PATH] [--mode time|fft|both]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> np.ndarray:
    """
    Load amplitude values from a CSV file.

    Handles:
      • Single-column CSVs with or without a header.
      • Multi-column CSVs where AMPLITUDE_COLUMN is specified in config.
      • Automatic detection of numeric columns when config columns are None.

    Returns a 1-D float32 numpy array of amplitude values.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception:
        # Re-try without a header row
        df = pd.read_csv(filepath, header=None)

    # Select amplitude column
    if config.AMPLITUDE_COLUMN and config.AMPLITUDE_COLUMN in df.columns:
        amplitude = df[config.AMPLITUDE_COLUMN].values.astype(np.float32)
    else:
        # Pick the first fully numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError(f"No numeric columns found in {filepath}")
        # Prefer a column whose name contains 'amp' or 'value'
        preferred = [c for c in numeric_cols
                     if any(kw in str(c).lower() for kw in ("amp", "value", "signal", "ch"))]
        col = preferred[0] if preferred else numeric_cols[-1]
        amplitude = df[col].values.astype(np.float32)

    # Remove NaNs
    amplitude = amplitude[~np.isnan(amplitude)]

    if len(amplitude) == 0:
        raise ValueError(f"Empty signal after cleaning in {filepath}")

    return amplitude


def maybe_window(signal: np.ndarray, window_size) -> list:
    """
    If WINDOW_SIZE is set, split signal into non-overlapping windows.
    Otherwise return the full signal as a single-element list.
    """
    if window_size is None or window_size >= len(signal):
        return [signal]
    return [signal[i:i + window_size]
            for i in range(0, len(signal) - window_size + 1, window_size)]


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Min-max normalize to [-1, 1]."""
    mn, mx = signal.min(), signal.max()
    if mx - mn < 1e-9:
        return np.zeros_like(signal)
    return 2 * (signal - mn) / (mx - mn) - 1


def save_time_domain_image(signal: np.ndarray, out_path: str):
    """Render amplitude vs. sample-index as a clean PNG (no axes clutter)."""
    fig, ax = plt.subplots(figsize=(config.IMG_WIDTH / config.IMG_DPI,
                                    config.IMG_HEIGHT / config.IMG_DPI),
                           dpi=config.IMG_DPI)
    ax.plot(signal, color="steelblue", linewidth=0.8)
    ax.set_xlim(0, len(signal) - 1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=config.IMG_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_fft_image(signal: np.ndarray, out_path: str):
    """
    Compute the FFT magnitude spectrum and save as a PNG.
    Uses log scale and only the positive (one-sided) spectrum.
    """
    n = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal)) / n    # normalise
    fft_mag[fft_mag < 1e-10] = 1e-10             # avoid log(0)
    log_mag = 20 * np.log10(fft_mag)             # convert to dB

    fig, ax = plt.subplots(figsize=(config.IMG_WIDTH / config.IMG_DPI,
                                    config.IMG_HEIGHT / config.IMG_DPI),
                           dpi=config.IMG_DPI)
    ax.fill_between(np.arange(len(log_mag)), log_mag,
                    log_mag.min(), color="crimson", alpha=0.85)
    ax.set_xlim(0, len(log_mag) - 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=config.IMG_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_combined_image(signal: np.ndarray, out_path: str):
    """
    2-row image: top = time-domain, bottom = FFT spectrum.
    Useful alternative to separate directories.
    """
    n = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal)) / n
    fft_mag[fft_mag < 1e-10] = 1e-10
    log_mag = 20 * np.log10(fft_mag)

    fig, axes = plt.subplots(2, 1,
                             figsize=(config.IMG_WIDTH / config.IMG_DPI,
                                      config.IMG_HEIGHT / config.IMG_DPI),
                             dpi=config.IMG_DPI)
    axes[0].plot(signal, color="steelblue", linewidth=0.8)
    axes[0].set_xlim(0, n - 1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].axis("off")

    axes[1].fill_between(np.arange(len(log_mag)), log_mag,
                         log_mag.min(), color="crimson", alpha=0.85)
    axes[1].set_xlim(0, len(log_mag) - 1)
    axes[1].axis("off")

    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0, h_pad=0.1)
    plt.savefig(out_path, dpi=config.IMG_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion logic
# ──────────────────────────────────────────────────────────────────────────────

def process_class(class_name: str, raw_dir: str, img_dir: str, mode: str):
    """
    Process all CSV files for one class label.

    Args:
        class_name: e.g. "sine"
        raw_dir:    root of raw CSV data
        img_dir:    root of output image data
        mode:       "time" | "fft" | "both" | "combined"
    """
    csv_dir = os.path.join(raw_dir, class_name)
    if not os.path.isdir(csv_dir):
        print(f"  [SKIP] {csv_dir} not found.")
        return

    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    if len(csv_files) == 0:
        print(f"  [SKIP] No CSVs found in {csv_dir}.")
        return

    # Build output directories
    if mode in ("time", "both"):
        time_out = os.path.join(img_dir, "time", class_name)
        os.makedirs(time_out, exist_ok=True)
    if mode in ("fft", "both"):
        fft_out = os.path.join(img_dir, "fft", class_name)
        os.makedirs(fft_out, exist_ok=True)
    if mode == "combined":
        combined_out = os.path.join(img_dir, "combined", class_name)
        os.makedirs(combined_out, exist_ok=True)

    ok = skipped = 0
    for csv_path in tqdm(csv_files, desc=f"  {class_name}", unit="file", leave=False):
        stem = csv_path.stem
        try:
            raw_signal = load_csv(str(csv_path))
        except Exception as e:
            print(f"\n  [WARN] Could not load {csv_path.name}: {e}")
            skipped += 1
            continue

        windows = maybe_window(raw_signal, config.WINDOW_SIZE)

        for w_idx, window in enumerate(windows):
            signal = normalize_signal(window)
            suffix = f"_w{w_idx:04d}" if len(windows) > 1 else ""

            try:
                if mode in ("time", "both"):
                    out = os.path.join(time_out, f"{stem}{suffix}.png")
                    if not os.path.exists(out):
                        save_time_domain_image(signal, out)

                if mode in ("fft", "both"):
                    out = os.path.join(fft_out, f"{stem}{suffix}.png")
                    if not os.path.exists(out):
                        save_fft_image(signal, out)

                if mode == "combined":
                    out = os.path.join(combined_out, f"{stem}{suffix}.png")
                    if not os.path.exists(out):
                        save_combined_image(signal, out)

                ok += 1
            except Exception as e:
                print(f"\n  [WARN] Failed to save image for {csv_path.name}: {e}")
                skipped += 1

    print(f"  {class_name}: {ok} images generated, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert piezo waveform CSVs to PNG images for MambaVision."
    )
    parser.add_argument("--raw_dir", default=config.RAW_DATA_DIR,
                        help="Root directory of raw CSV data.")
    parser.add_argument("--img_dir", default=config.IMAGE_DATA_DIR,
                        help="Root directory for output images.")
    parser.add_argument("--mode", default="both",
                        choices=["time", "fft", "both", "combined"],
                        help=(
                            "time     – time-domain images only\n"
                            "fft      – FFT spectrum images only\n"
                            "both     – both time and fft (separate folders)\n"
                            "combined – single image with time + fft stacked"
                        ))
    parser.add_argument("--classes", nargs="+", default=config.CLASSES,
                        help="Subset of classes to process.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Piezo Waveform → Image Converter")
    print(f"{'='*60}")
    print(f"  Raw data  : {args.raw_dir}")
    print(f"  Output    : {args.img_dir}")
    print(f"  Mode      : {args.mode}")
    print(f"  Classes   : {args.classes}")
    print(f"{'='*60}\n")

    for cls in args.classes:
        print(f"Processing class: [{cls}]")
        process_class(cls, args.raw_dir, args.img_dir, args.mode)

    print(f"\n✓ Done. Images saved to: {args.img_dir}")


if __name__ == "__main__":
    main()
