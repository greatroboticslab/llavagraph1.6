"""
evaluate.py
-----------
Evaluate a trained MambaVision checkpoint on the test split.

Outputs:
  • Overall accuracy, top-3 accuracy
  • Per-class precision / recall / F1
  • Confusion matrix PNG
  • Misclassified samples list (CSV)

Usage:
    python evaluate.py --checkpoint checkpoints/best.pth
                       [--split test|val|train]
                       [--save_dir results/]
"""

import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             top_k_accuracy_score)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import config
from dataset import PiezoWaveformDataset, get_transforms, discover_files
from model import build_model, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Inference pass
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module,
                  loader: DataLoader,
                  device: str = config.DEVICE):
    """
    Run the model over the loader and collect predictions.

    Returns:
        all_labels   : (N,) int array – ground-truth class indices
        all_preds    : (N,) int array – predicted class indices
        all_probs    : (N, C) float array – softmax probabilities
        all_paths    : list of file paths (if loader.dataset has .paths)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Inference", unit="batch"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)

    all_paths = getattr(loader.dataset, "paths", [])
    return all_labels, all_preds, all_probs, all_paths


# ──────────────────────────────────────────────────────────────────────────────
# Metrics & visualisation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray,
                    preds: np.ndarray,
                    probs: np.ndarray) -> dict:
    """Compute and print classification metrics."""
    top1 = (labels == preds).mean()
    top3 = top_k_accuracy_score(labels, probs,
                                k=min(3, config.NUM_CLASSES),
                                labels=list(range(config.NUM_CLASSES)))

    report = classification_report(
        labels, preds,
        target_names=config.CLASSES,
        output_dict=True
    )

    print(f"\n{'─'*50}")
    print(f"  Overall Top-1 Accuracy : {top1 * 100:.2f}%")
    print(f"  Overall Top-3 Accuracy : {top3 * 100:.2f}%")
    print(f"{'─'*50}")
    print(f"\n  Per-class report:\n")
    print(classification_report(labels, preds, target_names=config.CLASSES))

    return {"top1": top1, "top3": top3, "report": report}


def plot_confusion_matrix(labels: np.ndarray,
                           preds: np.ndarray,
                           save_path: str):
    """Save a normalised confusion matrix PNG."""
    cm = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm,
                annot=cm,           # show raw counts inside cells
                fmt="d",
                cmap="Blues",
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES,
                linewidths=0.5,
                vmin=0.0, vmax=1.0,
                ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix (cell shade = row-normalised, "
                 "number = raw count)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def save_misclassified(paths,
                       labels: np.ndarray,
                       preds: np.ndarray,
                       probs: np.ndarray,
                       save_path: str):
    """Save a CSV listing every misclassified sample with confidence scores."""
    if len(paths) == 0:
        print("  [SKIP] No file paths available for misclassification analysis.")
        return

    rows = []
    for i, (path, true, pred) in enumerate(zip(paths, labels, preds)):
        if true != pred:
            rows.append({
                "path":            path,
                "true_label":      config.IDX_TO_CLASS[true],
                "pred_label":      config.IDX_TO_CLASS[pred],
                "confidence":      float(probs[i, pred]),
                **{f"prob_{c}": float(probs[i, j])
                   for j, c in enumerate(config.CLASSES)},
            })

    df = pd.DataFrame(rows)
    df.sort_values("confidence", ascending=False, inplace=True)
    df.to_csv(save_path, index=False)
    print(f"  Misclassified samples ({len(df)}) saved to {save_path}")


def plot_per_class_confidence(labels: np.ndarray,
                               preds: np.ndarray,
                               probs: np.ndarray,
                               save_path: str):
    """Box-plot of correct-class confidence per ground-truth class."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_class = []
    for cls_idx, cls_name in enumerate(config.CLASSES):
        mask = labels == cls_idx
        if mask.sum() == 0:
            data_by_class.append([])
            continue
        data_by_class.append(probs[mask, cls_idx].tolist())

    ax.boxplot(data_by_class, labels=config.CLASSES, patch_artist=True)
    ax.set_ylabel("Confidence (softmax prob for true class)")
    ax.set_title("Per-class prediction confidence")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confidence plot saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Piezo Waveform Classifier – Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")
    print(f"{'='*60}\n")

    # ── Load dataset split ────────────────────────────────────────────────────
    # Load from pre-split directories
    split_dir = {"train": config.TRAIN_DIR, "val": config.VAL_DIR, "test": config.TEST_DIR}[args.split]
    paths, labels = discover_files(split_dir)
    dataset = PiezoWaveformDataset(paths, labels, get_transforms(args.split))

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=True)
    print(f"  Dataset size ({args.split}): {len(dataset)} samples")

    # ── Build & load model ────────────────────────────────────────────────────
    model = build_model().to(args.device)
    load_checkpoint(args.checkpoint, model, device=args.device)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_labels, all_preds, all_probs, all_paths = run_inference(
        model, loader, args.device
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    # Save metrics JSON
    metrics_path = os.path.join(args.save_dir, f"metrics_{args.split}.json")
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "report"}, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        all_labels, all_preds,
        os.path.join(args.save_dir, f"confusion_matrix_{args.split}.png")
    )
    plot_per_class_confidence(
        all_labels, all_preds, all_probs,
        os.path.join(args.save_dir, f"confidence_boxplot_{args.split}.png")
    )
    save_misclassified(
        all_paths, all_labels, all_preds, all_probs,
        os.path.join(args.save_dir, f"misclassified_{args.split}.csv")
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MambaVision checkpoint."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pth",
                        help="Path to the .pth checkpoint file.")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--save_dir", default=config.RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
