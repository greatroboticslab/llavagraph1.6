#!/usr/bin/env python3
"""
train_vit.py  —  ViT (google/vit-base-patch16-224) training on closed-loop data.

Usage:
    python train_vit.py
    python train_vit.py --epochs 30 --batch_size 16
"""

import sys
import os
import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from transformers import ViTForImageClassification

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(split):
    base = [T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if split == "train":
        base = ([T.Resize((224, 224)),
                 T.RandomHorizontalFlip(0.5),
                 T.ColorJitter(brightness=0.2, contrast=0.2)] +
                base[1:])
    return T.Compose(base)


class WaveformDataset(Dataset):
    def __init__(self, root, split):
        self.transform = get_transform(split)
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


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, split):
    is_train = split == "train"
    model.train() if is_train else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=f"  {split:5s}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out    = model(pixel_values=imgs)
            logits = out.logits
            loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += bs

    n = max(total, 1)
    return {"loss": total_loss / n, "acc": correct / n}


def train(args):
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)

    print("=" * 60)
    print(f"  ViT — Closed-Loop Training")
    print(f"  Data    : {cfg.DATA_DIR}")
    print(f"  Device  : {args.device}")
    print(f"  Epochs  : {args.epochs}  BS: {args.batch_size}  LR: {args.lr}")
    print("=" * 60)

    train_ds = WaveformDataset(cfg.DATA_DIR, "train")
    val_ds   = WaveformDataset(cfg.DATA_DIR, "val")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=cfg.NUM_CLASSES,
        id2label=cfg.IDX_TO_CLASS,
        label2id=cfg.CLASS_TO_IDX,
        ignore_mismatched_sizes=True,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    # Freeze all layers except last 2 encoder blocks + classifier head
    for param in model.parameters():
        param.requires_grad = False
    n_blocks = len(model.vit.encoder.layer)
    for block in model.vit.encoder.layer[n_blocks - 2:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.vit.layernorm.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  ViT params: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    head_params     = list(model.classifier.parameters())
    optimizer = AdamW([{"params": backbone_params, "lr": args.lr * 0.1},
                       {"params": head_params,     "lr": args.lr}],
                      weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc, patience = 0.0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer, args.device, "train")
        vl = run_epoch(model, val_loader,   criterion, None,      args.device, "val")
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{args.epochs}  "
              f"train_loss={tr['loss']:.4f} acc={tr['acc']:.4f}  |  "
              f"val_loss={vl['loss']:.4f} acc={vl['acc']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        for k, v in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                     ("val_loss",   vl["loss"]), ("val_acc",   vl["acc"])]:
            history[k].append(v)

        if vl["acc"] > best_acc:
            best_acc, patience = vl["acc"], 0
            model.save_pretrained(os.path.join(args.ckpt_dir, "best"))
            print(f"  ✓ best val_acc={best_acc:.4f} saved")
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                print("  Early stopping.")
                break

    print(f"\n  Best val_acc = {best_acc:.4f}")
    with open(os.path.join(args.results_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    _save_curves(history, args.results_dir)
    return best_acc


def _save_curves(history, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(e, history["train_loss"], label="train")
        axes[0].plot(e, history["val_loss"],   label="val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(e, history["train_acc"], label="train")
        axes[1].plot(e, history["val_acc"],   label="val")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
        plt.close()
    except Exception as ex:
        print(f"[warn] curves: {ex}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=cfg.EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--device",      default=cfg.DEVICE)
    p.add_argument("--results_dir", default=cfg.RESULTS_DIR_VIT)
    p.add_argument("--ckpt_dir",    default=cfg.CKPT_DIR_VIT)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
