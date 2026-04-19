#!/usr/bin/env python3
"""
evaluate_both.py  —  Evaluate MambaVision and ViT on closed-loop test set.

Outputs (all saved to closed_loop/results/comparison/):
  • confusion matrices (per model)
  • per-class accuracy bar chart
  • training curve overlay (MambaVision vs ViT)
  • inference speed comparison bar chart
  • summary table (CSV + printed)
  • metrics JSON for each model

Usage:
    cd closed_loop
    python evaluate_both.py
    python evaluate_both.py --split val   # evaluate on val instead
"""

import sys
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from sklearn.metrics import (classification_report, confusion_matrix,
                              top_k_accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
HERE  = os.path.dirname(os.path.abspath(__file__))
MAMBA = os.path.join(os.path.dirname(HERE), "mambavision")
sys.path.insert(0, HERE)
sys.path.insert(1, MAMBA)

import config as cfg
import importlib
mamba_cfg = importlib.import_module("config")
for attr in dir(cfg):
    setattr(mamba_cfg, attr, getattr(cfg, attr))

from model import build_model, load_checkpoint
from transformers import ViTForImageClassification

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

OUT_DIR = os.path.join(HERE, "results", "comparison")


# ── Dataset ───────────────────────────────────────────────────────────────────

class WaveformDataset(Dataset):
    def __init__(self, root, split):
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.transform = tf
        self.paths, self.labels = [], []
        for i, cls in enumerate(cfg.CLASSES):
            cls_dir = os.path.join(root, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in sorted(os.listdir(cls_dir)):
                if f.endswith(".png"):
                    self.paths.append(os.path.join(cls_dir, f))
                    self.labels.append(i)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, forward_fn):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for imgs, labels in tqdm(loader, desc="  inference", leave=False):
        imgs = imgs.to(device)
        logits = forward_fn(model, imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_preds.append(probs.argmax(axis=1))
        all_probs.append(probs)
    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))


def measure_speed(model, loader, device, forward_fn, n_batches=30):
    """Returns avg ms per image (after 3-batch warmup)."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            if i < 3:                   # warmup
                forward_fn(model, imgs)
                continue
            if i >= n_batches + 3:
                break
            t0 = time.perf_counter()
            forward_fn(model, imgs)
            torch.mps.synchronize() if device == "mps" else None
            times.append((time.perf_counter() - t0) * 1000 / imgs.size(0))
    return float(np.mean(times)), float(np.std(times))


# ── Metrics & plots ───────────────────────────────────────────────────────────

def metrics_dict(labels, preds, probs):
    top1 = float((labels == preds).mean())
    top3 = float(top_k_accuracy_score(labels, probs,
                                      k=min(3, cfg.NUM_CLASSES),
                                      labels=list(range(cfg.NUM_CLASSES))))
    report = classification_report(labels, preds,
                                   target_names=cfg.CLASSES,
                                   output_dict=True)
    return {"top1": top1, "top3": top3, "report": report}


def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds, labels=list(range(cfg.NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES,
                linewidths=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_accuracy(results, save_path):
    """Grouped bar chart: per-class accuracy for each model."""
    models  = list(results.keys())
    x       = np.arange(cfg.NUM_CLASSES)
    width   = 0.35
    colors  = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (name, color) in enumerate(zip(models, colors)):
        report = results[name]["metrics"]["report"]
        accs   = [report[cls]["recall"] for cls in cfg.CLASSES]
        ax.bar(x + k * width - width / 2, accs, width,
               label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg.CLASSES, fontsize=11)
    ax.set_ylabel("Recall (per-class accuracy)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Accuracy: MambaVision vs ViT")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(save_path):
    """Overlay training curves from both models' history.json files."""
    paths = {
        "MambaVision": os.path.join(HERE, "results", "mambavision", "history.json"),
        "ViT":         os.path.join(HERE, "results", "vit", "history.json"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"MambaVision": "#4C72B0", "ViT": "#DD8452"}
    ls     = {"train": "-",  "val": "--"}

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"  [warn] history not found: {path}")
            continue
        with open(path) as f:
            h = json.load(f)
        e = range(1, len(h["train_loss"]) + 1)
        c = colors[name]
        axes[0].plot(e, h["train_loss"], c=c, ls=ls["train"], label=f"{name} train")
        axes[0].plot(e, h["val_loss"],   c=c, ls=ls["val"],   label=f"{name} val")
        axes[1].plot(e, h["train_acc"],  c=c, ls=ls["train"], label=f"{name} train")
        axes[1].plot(e, h["val_acc"],    c=c, ls=ls["val"],   label=f"{name} val")

    for ax, title, ylabel in zip(axes,
                                  ["Loss Curves", "Accuracy Curves"],
                                  ["Cross-Entropy Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_speed_comparison(speed_results, save_path):
    """Bar chart comparing inference speed (ms/image)."""
    names  = list(speed_results.keys())
    means  = [speed_results[n]["mean_ms"] for n in names]
    stds   = [speed_results[n]["std_ms"]  for n in names]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.4)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{mean:.2f} ms", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Inference time per image (ms)")
    ax.set_title("Inference Speed Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary_table(results, speed_results):
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        report = m["report"]
        row = {
            "Model":       name,
            "Top-1 (%)":   f"{m['top1']*100:.2f}",
            "Top-3 (%)":   f"{m['top3']*100:.2f}",
            "ms/image":    f"{speed_results.get(name, {}).get('mean_ms', float('nan')):.2f}",
        }
        for cls in cfg.CLASSES:
            row[f"{cls} F1"] = f"{report[cls]['f1-score']:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = cfg.DEVICE
    print(f"\n{'='*60}")
    print(f"  Closed-Loop Evaluation: MambaVision vs ViT")
    print(f"  Split  : {args.split}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds     = WaveformDataset(cfg.DATA_DIR, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    print(f"  Test samples: {len(ds)}")

    results       = {}
    speed_results = {}

    # ── MambaVision ───────────────────────────────────────────────────────────
    mamba_ckpt = os.path.join(HERE, "results", "mambavision", "checkpoints", "best.pth")
    if os.path.exists(mamba_ckpt):
        print("\n[MambaVision]")
        mamba_model = build_model().to(device)
        load_checkpoint(mamba_ckpt, mamba_model, device=device)

        def mamba_forward(m, x): return m(x)

        labels, preds, probs = run_inference(
            mamba_model, loader, device, mamba_forward)
        m = metrics_dict(labels, preds, probs)
        print(f"  Top-1: {m['top1']*100:.2f}%   Top-3: {m['top3']*100:.2f}%")
        print(classification_report(labels, preds, target_names=cfg.CLASSES))
        results["MambaVision"] = {"labels": labels, "preds": preds,
                                  "probs": probs, "metrics": m}

        with open(os.path.join(OUT_DIR, "metrics_mamba.json"), "w") as f:
            json.dump({"top1": m["top1"], "top3": m["top3"],
                       "report": m["report"]}, f, indent=2)

        plot_confusion_matrix(
            labels, preds,
            f"MambaVision — Confusion Matrix ({args.split})",
            os.path.join(OUT_DIR, f"cm_mamba_{args.split}.png"))

        mean_ms, std_ms = measure_speed(mamba_model, loader, device, mamba_forward)
        speed_results["MambaVision"] = {"mean_ms": mean_ms, "std_ms": std_ms}
        print(f"  Speed: {mean_ms:.2f} ± {std_ms:.2f} ms/image")
    else:
        print(f"  [skip MambaVision] checkpoint not found: {mamba_ckpt}")

    # ── ViT ───────────────────────────────────────────────────────────────────
    vit_ckpt = os.path.join(HERE, "results", "vit", "checkpoints", "best")
    if os.path.exists(vit_ckpt):
        print("\n[ViT]")
        vit_model = ViTForImageClassification.from_pretrained(
            vit_ckpt,
            num_labels=cfg.NUM_CLASSES,
            id2label=cfg.IDX_TO_CLASS,
            label2id=cfg.CLASS_TO_IDX,
            ignore_mismatched_sizes=True,
        ).to(device)

        def vit_forward(m, x): return m(pixel_values=x).logits

        labels, preds, probs = run_inference(
            vit_model, loader, device, vit_forward)
        m = metrics_dict(labels, preds, probs)
        print(f"  Top-1: {m['top1']*100:.2f}%   Top-3: {m['top3']*100:.2f}%")
        print(classification_report(labels, preds, target_names=cfg.CLASSES))
        results["ViT"] = {"labels": labels, "preds": preds,
                          "probs": probs, "metrics": m}

        with open(os.path.join(OUT_DIR, "metrics_vit.json"), "w") as f:
            json.dump({"top1": m["top1"], "top3": m["top3"],
                       "report": m["report"]}, f, indent=2)

        plot_confusion_matrix(
            labels, preds,
            f"ViT — Confusion Matrix ({args.split})",
            os.path.join(OUT_DIR, f"cm_vit_{args.split}.png"))

        mean_ms, std_ms = measure_speed(vit_model, loader, device, vit_forward)
        speed_results["ViT"] = {"mean_ms": mean_ms, "std_ms": std_ms}
        print(f"  Speed: {mean_ms:.2f} ± {std_ms:.2f} ms/image")
    else:
        print(f"  [skip ViT] checkpoint not found: {vit_ckpt}")

    if not results:
        print("\nNo checkpoints found. Train at least one model first.")
        return

    # ── Comparison figures ────────────────────────────────────────────────────
    print("\n[Generating comparison figures]")

    if len(results) == 2:
        plot_per_class_accuracy(
            results, os.path.join(OUT_DIR, "per_class_accuracy.png"))
        plot_speed_comparison(
            speed_results, os.path.join(OUT_DIR, "speed_comparison.png"))

    plot_training_curves(os.path.join(OUT_DIR, "training_curves_overlay.png"))

    # ── Summary table ─────────────────────────────────────────────────────────
    df = print_summary_table(results, speed_results)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(f"\n  Summary CSV saved: {OUT_DIR}/summary.csv")
    print(f"  All outputs in  : {OUT_DIR}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
