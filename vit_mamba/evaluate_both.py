#!/usr/bin/env python3
"""
evaluate_both.py  —  Evaluate MambaVision and ViT on the test set.

Outputs (saved to results/comparison/):
  • confusion matrices (per model)
  • per-class accuracy bar chart
  • training curve overlay
  • inference speed comparison
  • summary CSV + metrics JSON

Usage:
    python evaluate_both.py
    python evaluate_both.py --split val
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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as cfg
from model import build_model, load_checkpoint
from train_vit import ViTClassifier, build_vit_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
OUT_DIR       = os.path.join(HERE, "results", "comparison")


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
        imgs   = imgs.to(device)
        logits = forward_fn(model, imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_preds.append(probs.argmax(axis=1))
        all_probs.append(probs)
    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))


def measure_speed(model, loader, device, forward_fn, n_batches=30):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            if i < 3:
                forward_fn(model, imgs)
                continue
            if i >= n_batches + 3:
                break
            t0 = time.perf_counter()
            forward_fn(model, imgs)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000 / imgs.size(0))
    return float(np.mean(times)), float(np.std(times))


# ── Metrics & plots ───────────────────────────────────────────────────────────

def metrics_dict(labels, preds):
    top1   = float((labels == preds).mean())
    report = classification_report(labels, preds,
                                   target_names=cfg.CLASSES,
                                   output_dict=True)
    return {"top1": top1, "report": report}


def plot_confusion_matrix(labels, preds, title, save_path):
    cm      = confusion_matrix(labels, preds, labels=list(range(cfg.NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES,
                linewidths=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(title,        fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_accuracy(results, save_path):
    models = list(results.keys())
    x      = np.arange(cfg.NUM_CLASSES)
    width  = 0.35
    colors = ["#4C72B0", "#DD8452"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (name, color) in enumerate(zip(models, colors)):
        report = results[name]["metrics"]["report"]
        accs   = [report[cls]["recall"] for cls in cfg.CLASSES]
        ax.bar(x + k * width - width / 2, accs, width,
               label=name, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cfg.CLASSES, fontsize=11)
    ax.set_ylabel("Recall (per-class accuracy)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Accuracy: MambaVision vs ViT")
    ax.legend(fontsize=11); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(save_path):
    paths = {
        "MambaVision": os.path.join(HERE, "results", "mambavision", "history.json"),
        "ViT":         os.path.join(HERE, "results", "vit",         "history.json"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"MambaVision": "#4C72B0", "ViT": "#DD8452"}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"  [warn] history not found: {path}")
            continue
        with open(path) as f:
            h = json.load(f)
        e = range(1, len(h["train_loss"]) + 1)
        c = colors[name]
        axes[0].plot(e, h["train_loss"], c=c, ls="-",  label=f"{name} train")
        axes[0].plot(e, h["val_loss"],   c=c, ls="--", label=f"{name} val")
        axes[1].plot(e, h["train_acc"],  c=c, ls="-",  label=f"{name} train")
        axes[1].plot(e, h["val_acc"],    c=c, ls="--", label=f"{name} val")
    for ax, title, ylabel in zip(axes,
                                 ["Loss Curves", "Accuracy Curves"],
                                 ["Cross-Entropy Loss", "Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary_table(results, speed_results):
    rows = []
    for name, r in results.items():
        m      = r["metrics"]
        report = m["report"]
        row    = {
            "Model":          name,
            "Accuracy (%)":   f"{m['top1']*100:.2f}",
            "ms/image":       f"{speed_results.get(name, {}).get('mean_ms', float('nan')):.2f}",
        }
        for cls in cfg.CLASSES:
            row[f"{cls} F1"] = f"{report[cls]['f1-score']:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = cfg.DEVICE
    print(f"\n{'='*60}")
    print(f"  Evaluation: MambaVision vs ViT  |  split={args.split}  device={device}")
    print(f"{'='*60}\n")

    ds     = WaveformDataset(cfg.DATA_DIR, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())
    print(f"  {args.split} samples: {len(ds)}")

    results       = {}
    speed_results = {}

    # ── MambaVision ───────────────────────────────────────────────────────────
    mamba_ckpt = os.path.join(HERE, "results", "mambavision", "checkpoints", "best.pth")
    if os.path.exists(mamba_ckpt):
        print("\n[MambaVision]")
        mamba_model = build_model().to(device)
        load_checkpoint(mamba_ckpt, mamba_model, device=device)

        def mamba_fwd(m, x): return m(x)

        labels, preds, probs = run_inference(mamba_model, loader, device, mamba_fwd)
        m = metrics_dict(labels, preds)
        print(f"  Accuracy: {m['top1']*100:.2f}%")
        print(classification_report(labels, preds, target_names=cfg.CLASSES))
        results["MambaVision"] = {"labels": labels, "preds": preds,
                                  "probs": probs, "metrics": m}
        with open(os.path.join(OUT_DIR, "metrics_mamba.json"), "w") as f:
            json.dump({"accuracy": m["top1"], "report": m["report"]}, f, indent=2)
        plot_confusion_matrix(labels, preds,
                              f"MambaVision — Confusion Matrix ({args.split})",
                              os.path.join(OUT_DIR, f"cm_mamba_{args.split}.png"))
        mean_ms, std_ms = measure_speed(mamba_model, loader, device, mamba_fwd)
        speed_results["MambaVision"] = {"mean_ms": mean_ms, "std_ms": std_ms}
        print(f"  Speed: {mean_ms:.2f} ± {std_ms:.2f} ms/image")
    else:
        print(f"  [skip MambaVision] checkpoint not found: {mamba_ckpt}")

    # ── ViT ───────────────────────────────────────────────────────────────────
    vit_ckpt = os.path.join(HERE, "results", "vit", "checkpoints", "best.pth")
    if os.path.exists(vit_ckpt):
        print("\n[ViT]")
        # model.py adds all_tied_weights_keys as a read-only property to
        # PreTrainedModel for MambaVision compatibility. ViT's post_init()
        # tries to SET this attribute, which fails against a read-only property.
        # Remove it temporarily so ViT can load cleanly.
        from transformers import PreTrainedModel as _PTM
        _saved = _PTM.__dict__.get('all_tied_weights_keys')
        if isinstance(_saved, property):
            delattr(_PTM, 'all_tied_weights_keys')
        vit_model = build_vit_model(strategy="full").to(device)
        if isinstance(_saved, property):
            _PTM.all_tied_weights_keys = _saved
        ckpt = torch.load(vit_ckpt, map_location=device)
        vit_model.load_state_dict(ckpt["model"])
        print(f"  Checkpoint loaded (epoch {ckpt.get('epoch','?')}, "
              f"val_acc={ckpt.get('val_acc',0):.4f})")

        def vit_fwd(m, x): return m(x)

        labels, preds, probs = run_inference(vit_model, loader, device, vit_fwd)
        m = metrics_dict(labels, preds)
        print(f"  Accuracy: {m['top1']*100:.2f}%")
        print(classification_report(labels, preds, target_names=cfg.CLASSES))
        results["ViT"] = {"labels": labels, "preds": preds,
                          "probs": probs, "metrics": m}
        with open(os.path.join(OUT_DIR, "metrics_vit.json"), "w") as f:
            json.dump({"accuracy": m["top1"], "report": m["report"]}, f, indent=2)
        plot_confusion_matrix(labels, preds,
                              f"ViT — Confusion Matrix ({args.split})",
                              os.path.join(OUT_DIR, f"cm_vit_{args.split}.png"))
        mean_ms, std_ms = measure_speed(vit_model, loader, device, vit_fwd)
        speed_results["ViT"] = {"mean_ms": mean_ms, "std_ms": std_ms}
        print(f"  Speed: {mean_ms:.2f} ± {std_ms:.2f} ms/image")
    else:
        print(f"  [skip ViT] checkpoint not found: {vit_ckpt}")

    if not results:
        print("\nNo checkpoints found. Train at least one model first.")
        return

    print("\n[Generating comparison figures]")
    if len(results) == 2:
        plot_per_class_accuracy(results, os.path.join(OUT_DIR, "per_class_accuracy.png"))
    plot_training_curves(os.path.join(OUT_DIR, "training_curves_overlay.png"))

    df = print_summary_table(results, speed_results)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(f"\n  All outputs saved to: {OUT_DIR}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
