"""
inference.py
------------
Run inference on new piezo waveform data (CSV files or image files).

Two modes:
  • Single file  – predict one CSV / PNG and print the result
  • Batch folder – predict all CSVs in a folder and save a results CSV

Usage:
    # Single CSV prediction
    python inference.py --input path/to/signal.csv --checkpoint checkpoints/best.pth

    # Single image prediction
    python inference.py --input path/to/image.png --checkpoint checkpoints/best.pth --input_type image

    # Batch folder prediction
    python inference.py --input path/to/folder/ --checkpoint checkpoints/best.pth --batch

    # Visualise prediction
    python inference.py --input path/to/signal.csv --checkpoint checkpoints/best.pth --visualize
"""

import os
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

import config
from generate_images import (load_csv, normalize_signal,
                              save_time_domain_image, save_fft_image)
from model import build_model, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Transform for inference (no augmentation)
# ──────────────────────────────────────────────────────────────────────────────

INFER_TRANSFORM = T.Compose([
    T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH),
             interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop((config.IMG_HEIGHT, config.IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def csv_to_tensor(csv_path: str,
                  representation: str = "time") -> torch.Tensor:
    """
    Load a CSV waveform, render it as an image, and return a model-ready tensor.

    Args:
        csv_path:       Path to the CSV file.
        representation: "time" | "fft"

    Returns:
        Tensor of shape (1, C, H, W) ready for the model.
    """
    signal = load_csv(csv_path)
    signal = normalize_signal(signal)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if representation == "fft":
            save_fft_image(signal, tmp_path)
        else:
            save_time_domain_image(signal, tmp_path)
        image = Image.open(tmp_path).convert("RGB")
        tensor = INFER_TRANSFORM(image).unsqueeze(0)   # (1, C, H, W)
    finally:
        os.unlink(tmp_path)

    return tensor, signal


def image_to_tensor(img_path: str) -> torch.Tensor:
    """Load an already-rendered PNG and return a model-ready tensor."""
    image = Image.open(img_path).convert("RGB")
    return INFER_TRANSFORM(image).unsqueeze(0)


@torch.no_grad()
def predict_tensor(model: nn.Module,
                   tensor: torch.Tensor,
                   device: str = config.DEVICE) -> Tuple[str, float, np.ndarray]:
    """
    Run a single tensor through the model.

    Returns:
        predicted_class : class name string
        confidence      : softmax probability for predicted class
        all_probs       : array of shape (NUM_CLASSES,) – all class probabilities
    """
    model.eval()
    tensor = tensor.to(device)
    logits = model(tensor)                              # (1, C)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = probs.argmax()
    return config.IDX_TO_CLASS[pred_idx], float(probs[pred_idx]), probs


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def visualize_prediction(signal: np.ndarray,
                          class_name: str,
                          confidence: float,
                          all_probs: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Plot:
      • Top-left : time-domain waveform
      • Top-right: FFT spectrum
      • Bottom   : bar chart of class probabilities
    """
    n = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal)) / n
    fft_mag[fft_mag < 1e-10] = 1e-10
    log_mag = 20 * np.log10(fft_mag)

    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

    # Time domain
    ax_time = fig.add_subplot(gs[0, 0])
    ax_time.plot(signal, color="steelblue", linewidth=0.8)
    ax_time.set_title("Time Domain")
    ax_time.set_xlabel("Sample"); ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, alpha=0.3)

    # FFT
    ax_fft = fig.add_subplot(gs[0, 1])
    ax_fft.fill_between(np.arange(len(log_mag)), log_mag,
                         log_mag.min(), color="crimson", alpha=0.8)
    ax_fft.set_title("FFT Spectrum")
    ax_fft.set_xlabel("Frequency bin"); ax_fft.set_ylabel("Magnitude (dB)")
    ax_fft.grid(True, alpha=0.3)

    # Probability bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    colors = ["#2196F3" if c != class_name else "#4CAF50"
              for c in config.CLASSES]
    bars = ax_bar.bar(config.CLASSES, all_probs * 100, color=colors, edgecolor="white")
    ax_bar.set_ylabel("Probability (%)")
    ax_bar.set_title(
        f"Prediction: {class_name.upper()}  "
        f"(confidence = {confidence*100:.1f}%)"
    )
    ax_bar.set_ylim(0, 110)
    for bar, prob in zip(bars, all_probs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{prob*100:.1f}%",
                    ha="center", va="bottom", fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Piezo Waveform Classification – Inference Result",
                 fontsize=13, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Visualisation saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────────────────────

def batch_predict(folder: str,
                  model: nn.Module,
                  representation: str = "time",
                  device: str = config.DEVICE,
                  save_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Predict all CSV files in `folder`.

    Returns a DataFrame with columns:
        file, predicted_class, confidence, prob_<class> × N_CLASSES
    """
    csv_files = sorted(Path(folder).rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    rows = []
    for csv_path in tqdm(csv_files, desc="  Batch predict", unit="file"):
        try:
            tensor, _ = csv_to_tensor(str(csv_path), representation)
            cls_name, conf, probs = predict_tensor(model, tensor, device)
            row = {
                "file":            str(csv_path),
                "predicted_class": cls_name,
                "confidence":      round(conf, 4),
                **{f"prob_{c}": round(float(p), 4)
                   for c, p in zip(config.CLASSES, probs)},
            }
            rows.append(row)
        except Exception as e:
            print(f"\n  [WARN] Failed on {csv_path.name}: {e}")

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\n  Results saved to {save_csv}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with trained MambaVision piezo classifier."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a CSV file, PNG file, or folder of CSVs.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth",
                        help="Path to the .pth checkpoint file.")
    parser.add_argument("--representation", default="time",
                        choices=["time", "fft"],
                        help="Which waveform representation to use for CSVs.")
    parser.add_argument("--input_type", default="csv",
                        choices=["csv", "image"],
                        help="Whether the input is a CSV or an already-rendered PNG.")
    parser.add_argument("--batch", action="store_true",
                        help="Batch-predict all CSVs in --input folder.")
    parser.add_argument("--visualize", action="store_true",
                        help="Show / save a visualisation for single-file mode.")
    parser.add_argument("--save_dir", default=config.RESULTS_DIR,
                        help="Directory to save outputs.")
    parser.add_argument("--device", default=config.DEVICE)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint} …")
    model = build_model().to(args.device)
    load_checkpoint(args.checkpoint, model, device=args.device)

    # ── Batch mode ─────────────────────────────────────────────────────────────
    if args.batch:
        out_csv = os.path.join(args.save_dir, "batch_predictions.csv")
        df = batch_predict(
            args.input, model,
            representation=args.representation,
            device=args.device,
            save_csv=out_csv
        )
        print(f"\n  Batch prediction summary:\n")
        print(df["predicted_class"].value_counts().to_string())
        return

    # ── Single-file mode ───────────────────────────────────────────────────────
    input_path = args.input
    signal = None

    if args.input_type == "csv":
        tensor, signal = csv_to_tensor(input_path, args.representation)
    else:
        tensor = image_to_tensor(input_path)

    cls_name, confidence, all_probs = predict_tensor(model, tensor, args.device)

    # Print results
    print(f"\n{'─'*40}")
    print(f"  File       : {input_path}")
    print(f"  Prediction : {cls_name.upper()}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print(f"\n  Class probabilities:")
    for cls, prob in zip(config.CLASSES, all_probs):
        bar = "█" * int(prob * 30)
        print(f"    {cls:8s} {bar:<30s} {prob*100:5.1f}%")
    print(f"{'─'*40}")

    # Visualise
    if args.visualize and signal is not None:
        vis_path = os.path.join(
            args.save_dir,
            f"inference_{Path(input_path).stem}.png"
        )
        visualize_prediction(signal, cls_name, confidence, all_probs,
                             save_path=vis_path)
    elif args.visualize:
        print("  [NOTE] Visualisation is only available for CSV input.")


if __name__ == "__main__":
    main()
