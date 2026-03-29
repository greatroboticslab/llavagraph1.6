"""
utils.py
--------
Utility scripts for the Piezo Waveform pipeline:

  1. generate_synthetic_data()   – Create dummy CSV data for smoke-testing
                                   the full pipeline without real sensor data.
  2. grad_cam()                  – Gradient-weighted Class Activation Mapping
                                   to visualise what the model focuses on.
  3. export_onnx()               – Export the trained model to ONNX format
                                   for deployment / edge inference.

Usage:
    # Generate synthetic data (for testing)
    python utils.py --action synthetic --out_dir data/raw --n_samples 200

    # GradCAM on a single image
    python utils.py --action gradcam --checkpoint checkpoints/best.pth
                    --input path/to/image.png --save_dir results/

    # Export to ONNX
    python utils.py --action onnx --checkpoint checkpoints/best.pth
                    --out_path checkpoints/model.onnx
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetic waveform data generator
# ──────────────────────────────────────────────────────────────────────────────

def _generate_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n).astype(np.float32)


def _generate_sine(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.02, 0.15)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(n, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t + phase)


def _generate_square(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.02, 0.15)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(n, dtype=np.float32)
    return np.sign(np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def _generate_pulse(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = np.zeros(n, dtype=np.float32)
    n_pulses = rng.integers(3, 12)
    for _ in range(n_pulses):
        start  = rng.integers(0, n - 10)
        width  = rng.integers(2, max(3, n // 20))
        height = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
        signal[start:start + width] = height
    return signal


def _generate_ramp(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.02, 0.10)
    t = np.arange(n, dtype=np.float32)
    period = 1.0 / freq
    return (2 * ((t / period) % 1.0) - 1).astype(np.float32)


GENERATORS = {
    "noise":  _generate_noise,
    "sine":   _generate_sine,
    "square": _generate_square,
    "pulse":  _generate_pulse,
    "ramp":   _generate_ramp,
}


def generate_synthetic_data(out_dir: str = "data/raw",
                             n_samples: int = 200,
                             signal_length: int = 1024,
                             add_noise_snr_db: float = 20.0):
    """
    Generate synthetic CSV waveform files for all five classes.

    Args:
        out_dir:          Root directory to write CSVs (one sub-folder per class).
        n_samples:        Number of CSV files per class.
        signal_length:    Number of samples per waveform.
        add_noise_snr_db: SNR in dB of additive Gaussian noise (set very high
                          to generate clean waveforms).
    """
    noise_std = 10 ** (-add_noise_snr_db / 20.0)

    for cls_name, generator in GENERATORS.items():
        cls_dir = os.path.join(out_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(n_samples):
            seed   = hash((cls_name, i)) % (2 ** 31)
            signal = generator(signal_length, seed)
            # Add realistic noise
            rng    = np.random.default_rng(seed + 99999)
            signal = signal + rng.normal(0, noise_std, signal_length).astype(np.float32)

            df = pd.DataFrame({"amplitude": signal})
            fpath = os.path.join(cls_dir, f"{cls_name}_{i:04d}.csv")
            df.to_csv(fpath, index=False)

        print(f"  {cls_name:8s}: {n_samples} CSV files → {cls_dir}")

    print(f"\n✓ Synthetic data generated at: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Attaches hooks to the target layer, runs a forward + backward pass,
    and computes the spatial heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._hooks       = []

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self,
                 input_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Args:
            input_tensor: (1, C, H, W) tensor on the correct device.
            class_idx:    Target class (None = use predicted class).

        Returns:
            heatmap: 2-D numpy array (H, W), normalised to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1).squeeze(0)  # (H, W)
        cam     = torch.relu(cam).cpu().numpy()

        # Normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-9:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def _find_last_conv(model: nn.Module) -> nn.Module:
    """Walk the model and return the last Conv2d layer."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def run_grad_cam(checkpoint: str,
                 img_path: str,
                 save_path: str,
                 device: str = config.DEVICE):
    """
    Compute and save a Grad-CAM visualisation for a single image.
    """
    from model import build_model, load_checkpoint
    from inference import INFER_TRANSFORM

    model = build_model().to(device)
    load_checkpoint(checkpoint, model, device=device)
    model.eval()

    # Find target layer
    target_layer = _find_last_conv(model.backbone)
    if target_layer is None:
        print("  [WARN] No Conv2d found; Grad-CAM may not work for pure attention models.")
        return

    cam_extractor = GradCAM(model, target_layer)

    # Load image
    image = Image.open(img_path).convert("RGB")
    tensor = INFER_TRANSFORM(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    with torch.enable_grad():
        heatmap = cam_extractor(tensor)
    cam_extractor.remove_hooks()

    # Overlay on original image
    img_np = np.array(image.resize((config.IMG_WIDTH, config.IMG_HEIGHT)))
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(
        heatmap_resized.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(img_np)
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay"); axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. ONNX export
# ──────────────────────────────────────────────────────────────────────────────

def export_onnx(checkpoint: str,
                out_path: str = "checkpoints/model.onnx",
                device: str = "cpu",
                opset: int = 17):
    """
    Export the trained model to ONNX format.

    The exported model accepts input of shape (1, 3, H, W) and outputs
    logits of shape (1, NUM_CLASSES).
    """
    from model import build_model, load_checkpoint

    print(f"\nExporting to ONNX …")
    model = build_model().to(device)
    load_checkpoint(checkpoint, model, device=device)
    model.eval()

    dummy = torch.randn(1, 3, config.IMG_HEIGHT, config.IMG_WIDTH).to(device)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"},
                      "logits":       {0: "batch_size"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  ✓ ONNX model saved to {out_path}")
    print(f"    Input  : (batch, 3, {config.IMG_HEIGHT}, {config.IMG_WIDTH})")
    print(f"    Output : (batch, {config.NUM_CLASSES})  classes: {config.CLASSES}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Piezo pipeline utilities.")
    parser.add_argument("--action", required=True,
                        choices=["synthetic", "gradcam", "onnx"])
    # Synthetic
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--signal_length", type=int, default=1024)
    # GradCAM / ONNX
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--input", default=None, help="Input image path for GradCAM.")
    parser.add_argument("--save_dir", default=config.RESULTS_DIR)
    parser.add_argument("--out_path", default="checkpoints/model.onnx")
    parser.add_argument("--device", default=config.DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.action == "synthetic":
        generate_synthetic_data(
            out_dir=args.out_dir,
            n_samples=args.n_samples,
            signal_length=args.signal_length,
        )

    elif args.action == "gradcam":
        assert args.input, "--input is required for gradcam"
        stem = Path(args.input).stem
        save_path = os.path.join(args.save_dir, f"gradcam_{stem}.png")
        run_grad_cam(args.checkpoint, args.input, save_path, args.device)

    elif args.action == "onnx":
        export_onnx(args.checkpoint, args.out_path, device="cpu")
