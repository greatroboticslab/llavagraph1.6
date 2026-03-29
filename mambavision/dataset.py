"""
dataset.py
----------
PyTorch Dataset and DataLoader utilities for the Piezo Waveform pipeline.

Supports:
  • Loading from pre-split train/val/test directories
  • Class-weight computation for imbalanced datasets
  • Standard ImageNet-style transforms + configurable augmentation
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> T.Compose:
    """
    Return a torchvision transform pipeline.

    Args:
        split: "train" | "val" | "test"
    """
    resize = T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH),
                      interpolation=T.InterpolationMode.BICUBIC)
    normalize = T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

    if split == "train" and config.USE_AUGMENTATION:
        aug_list = [resize, T.CenterCrop((config.IMG_HEIGHT, config.IMG_WIDTH))]

        if config.RANDOM_HORIZONTAL_FLIP:
            aug_list.append(T.RandomHorizontalFlip(p=0.5))
        if config.RANDOM_VERTICAL_FLIP:
            aug_list.append(T.RandomVerticalFlip(p=0.5))
        if config.COLOR_JITTER_STRENGTH > 0:
            s = config.COLOR_JITTER_STRENGTH
            aug_list.append(T.ColorJitter(brightness=s, contrast=s,
                                          saturation=s * 0.5, hue=0.0))
        aug_list.append(T.ToTensor())
        aug_list.append(normalize)
        if config.RANDOM_ERASING_PROB > 0:
            aug_list.append(T.RandomErasing(p=config.RANDOM_ERASING_PROB,
                                            scale=(0.02, 0.20)))
        return T.Compose(aug_list)

    else:   # val / test
        return T.Compose([
            resize,
            T.CenterCrop((config.IMG_HEIGHT, config.IMG_WIDTH)),
            T.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_files(split_dir: str,
                   classes: List[str] = None) -> Tuple[List[str], List[int]]:
    """
    Walk split_dir/<class>/ directories and collect (filepath, label) pairs.

    Args:
        split_dir: e.g. "data/train" or "data/val" or "data/test"
        classes:   subset of class names to load (default: config.CLASSES)

    Returns:
        paths  – list of absolute PNG file paths
        labels – list of integer class indices
    """
    classes = classes or config.CLASSES
    paths, labels = [], []

    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Class directory not found: {cls_dir}")
            continue
        label_idx = config.CLASS_TO_IDX[cls]
        for fpath in sorted(Path(cls_dir).glob("*.png")):
            paths.append(str(fpath))
            labels.append(label_idx)

    if len(paths) == 0:
        raise RuntimeError(
            f"No PNG files found under {split_dir}. "
            "Ensure your data folder contains PNG images in class subdirectories."
        )
    return paths, labels


def split_files(paths: List[str],
                labels: List[int],
                train_ratio: float = config.TRAIN_RATIO,
                val_ratio:   float = config.VAL_RATIO,
                seed: int = config.RANDOM_SEED,
                save_split_json: Optional[str] = None
                ) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Stratified train / val / test split.

    Returns:
        (train_paths, train_labels),
        (val_paths,   val_labels),
        (test_paths,  test_labels)
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    # First split: train vs (val + test)
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )
    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l,
        test_size=relative_test,
        stratify=tmp_l,
        random_state=seed
    )

    if save_split_json:
        split_info = {
            "train": list(zip(tr_p, tr_l)),
            "val":   list(zip(va_p, va_l)),
            "test":  list(zip(te_p, te_l)),
        }
        os.makedirs(os.path.dirname(save_split_json) or ".", exist_ok=True)
        with open(save_split_json, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"  Split saved to {save_split_json}")

    print(f"  Split → train: {len(tr_p)}  val: {len(va_p)}  test: {len(te_p)}")
    return (tr_p, tr_l), (va_p, va_l), (te_p, te_l)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class PiezoWaveformDataset(Dataset):
    """
    PyTorch Dataset for piezo waveform PNG images.

    Args:
        paths:     list of image file paths
        labels:    list of integer class indices
        transform: torchvision transform pipeline
        dual_mode: if True, expect paths to hold (time_path, fft_path) tuples
                   and return a 2-image pair for dual-stream models (future use)
    """

    def __init__(self,
                 paths: List[str],
                 labels: List[int],
                 transform: Optional[T.Compose] = None,
                 dual_mode: bool = False):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
        self.dual_mode = dual_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        label    = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    # ── Convenience class-methods ─────────────────────────────────────────────

    @classmethod
    def from_directory(cls,
                       image_root: str,
                       split: str,
                       classes: List[str] = None,
                       save_split_json: Optional[str] = None):
        """
        Build train / val / test datasets from an image root directory.

        Usage:
            train_ds, val_ds, test_ds = PiezoWaveformDataset.from_directory(
                "data/images/time", split="all"
            )
        Returns the requested split dataset, or a tuple of (train, val, test)
        when split == "all".
        """
        paths, labels = discover_files(image_root, classes)
        (tr_p, tr_l), (va_p, va_l), (te_p, te_l) = split_files(
            paths, labels, save_split_json=save_split_json
        )

        datasets = {
            "train": cls(tr_p, tr_l, get_transforms("train")),
            "val":   cls(va_p, va_l, get_transforms("val")),
            "test":  cls(te_p, te_l, get_transforms("test")),
        }

        if split == "all":
            return datasets["train"], datasets["val"], datasets["test"]
        return datasets[split]

    @classmethod
    def from_split_json(cls, json_path: str, split: str):
        """Recreate a dataset from a previously saved split JSON file."""
        with open(json_path) as f:
            info = json.load(f)
        entries = info[split]
        paths  = [e[0] for e in entries]
        labels = [e[1] for e in entries]
        return cls(paths, labels, get_transforms(split))


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(data_dir: str = config.DATA_DIR,
                      save_split_json: Optional[str] = None
                      ) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from pre-split directories.

    Args:
        data_dir:        Root of data (e.g. "data").
                         Expects train/, val/, test/ subdirectories.
        save_split_json: Optional path to persist the split for reproducibility.

    Returns:
        dict with keys "train", "val", "test"
    """
    print(f"\nBuilding DataLoaders from: {data_dir}")

    # Load from pre-split directories
    train_paths, train_labels = discover_files(config.TRAIN_DIR)
    val_paths, val_labels = discover_files(config.VAL_DIR)
    test_paths, test_labels = discover_files(config.TEST_DIR)

    print(f"  Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")

    train_ds = PiezoWaveformDataset(train_paths, train_labels, get_transforms("train"))
    val_ds = PiezoWaveformDataset(val_paths, val_labels, get_transforms("val"))
    test_ds = PiezoWaveformDataset(test_paths, test_labels, get_transforms("test"))

    # Check if pin_memory is supported (not on MPS)
    import torch
    pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(train_ds,
                            batch_size=config.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=pin_memory,
                            drop_last=True),
        "val":   DataLoader(val_ds,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=pin_memory),
        "test":  DataLoader(test_ds,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.NUM_WORKERS,
                            pin_memory=pin_memory),
    }
    return loaders


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Inverse-frequency class weights for weighted cross-entropy.
    Useful when classes have different sample counts.
    """
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * config.NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaders = build_dataloaders(
        config.IMAGE_DATA_DIR,
        representation="time",
        save_split_json=os.path.join(config.RESULTS_DIR, "split.json")
    )
    for split_name, loader in loaders.items():
        imgs, lbls = next(iter(loader))
        print(f"{split_name:5s} → images: {imgs.shape}  labels: {lbls.shape}")
