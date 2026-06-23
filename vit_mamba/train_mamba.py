#!/usr/bin/env python3
"""
train_mamba.py  —  MambaVision (nvidia/MambaVision-T-1K) fine-tuning
                   for 5-class piezo waveform classification.

Usage:
    python train_mamba.py
    python train_mamba.py --epochs 50 --batch_size 32 --strategy partial
"""

import sys
import os
import argparse
import time
import json
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as cfg
from dataset import build_dataloaders, compute_class_weights
from model   import build_model, save_checkpoint, load_checkpoint


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log = logging.getLogger("mamba_train")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(logging.FileHandler(path, mode="w"))
    for h in log.handlers:
        h.setFormatter(fmt)
    return log


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, split, use_amp=False):
    is_train = split == "train"
    model.train() if is_train else model.eval()
    total_loss = correct = total = 0

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=f"  {split:5s}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct    += logits.argmax(1).eq(labels).sum().item()
            total      += bs

    n = max(total, 1)
    return {"loss": total_loss / n, "acc": correct / n}


def train(args):
    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    log = setup_logger(os.path.join(args.results_dir, "training.log"))
    log.info("=" * 60)
    log.info(f"  MambaVision — Piezo Waveform Training")
    log.info(f"  Model    : {args.model_id}")
    log.info(f"  Data     : {cfg.DATA_DIR}")
    log.info(f"  Device   : {args.device}")
    log.info(f"  Epochs   : {args.epochs}  BS: {args.batch_size}  LR: {args.lr}")
    log.info("=" * 60)

    cfg.BATCH_SIZE = args.batch_size

    loaders      = build_dataloaders(cfg.DATA_DIR)
    class_weights = compute_class_weights(
        loaders["train"].dataset.labels).to(args.device)

    model = build_model(model_id=args.model_id,
                        finetune_strategy=args.strategy).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=cfg.LABEL_SMOOTHING)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    optimizer = AdamW([{"params": backbone_params, "lr": args.lr * 0.1},
                       {"params": head_params,     "lr": args.lr}],
                      weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=args.epochs - cfg.WARMUP_EPOCHS,
                                  eta_min=1e-6)

    use_amp = torch.cuda.is_available()
    if use_amp:
        log.info("  AMP: enabled (bfloat16)")

    best_acc, patience = 0.0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], criterion, optimizer,
                       args.device, "train", use_amp)
        vl = run_epoch(model, loaders["val"],   criterion, None,
                       args.device, "val",   use_amp)
        if epoch >= cfg.WARMUP_EPOCHS:
            scheduler.step()

        log.info(f"Epoch {epoch+1:03d}/{args.epochs}  "
                 f"train_loss={tr['loss']:.4f} acc={tr['acc']:.4f}  |  "
                 f"val_loss={vl['loss']:.4f} acc={vl['acc']:.4f}  "
                 f"({time.time()-t0:.1f}s)")

        for k, v in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                     ("val_loss",   vl["loss"]), ("val_acc",   vl["acc"])]:
            history[k].append(v)

        if vl["acc"] > best_acc:
            best_acc, patience = vl["acc"], 0
            save_checkpoint(model, optimizer, epoch, vl["acc"],
                            os.path.join(args.ckpt_dir, "best.pth"))
            log.info(f"  ✓ best val_acc={best_acc:.4f} saved")
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                log.info("  Early stopping.")
                break

    log.info(f"\n  Best val_acc = {best_acc:.4f}")
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
    p.add_argument("--model_id",    default=cfg.MODEL_ID)
    p.add_argument("--epochs",      type=int,   default=cfg.EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--strategy",    default=cfg.FINETUNE_STRATEGY,
                   choices=["full", "head_only", "partial"])
    p.add_argument("--device",      default=cfg.DEVICE)
    p.add_argument("--results_dir", default=cfg.RESULTS_DIR_MAMBA)
    p.add_argument("--ckpt_dir",    default=cfg.CKPT_DIR_MAMBA)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
