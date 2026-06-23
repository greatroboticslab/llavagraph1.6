# Piezo Waveform Classifier — MambaVision vs ViT

5-class image classification of piezoelectric waveform signals.
Two models are trained and compared on the same dataset under identical conditions.

**Classes:** `noise` `pulse` `ramp` `sine` `square`

---

## Results (Test Set)

| Model | Params | Test Accuracy | Inference Speed |
|---|---|---|---|
| MambaVision-T (nvidia/MambaVision-T-1K) | 31.8M | **99.59%** | **0.79 ms/image** |
| ViT-Base (google/vit-base-patch16-224-in21k) | 85.8M | 98.77% | 2.28 ms/image |

Both models trained with `--strategy full` (all parameters), same dataset split, same optimizer (AdamW), same augmentation.
MambaVision is 2.7x smaller, 2.9x faster, and 0.82% more accurate.

**Per-class F1 scores:**

| Class | MambaVision | ViT |
|---|---|---|
| noise | 1.000 | 0.995 |
| pulse | 0.993 | 0.993 |
| ramp | 0.994 | 0.976 |
| sine | 0.992 | 0.979 |
| square | 1.000 | 0.996 |

---

## Training Curves

### MambaVision

![MambaVision training curves](results_final/training_curves_mamba.png)

Train and validation accuracy track closely throughout training (final gap < 0.3%), indicating the model generalizes well with no meaningful overfitting.

### ViT

![ViT training curves](results_final/training_curves_vit.png)

Training accuracy reaches 100% from epoch 34 onward while validation accuracy plateaus around 97–98%. The train/val gap of approximately 2% is considered mild overfitting and is expected given the combination of a large model (85.8M parameters) and a relatively small training set (1,468 images). Two factors make this acceptable:

1. **Early stopping saves the right checkpoint.** The model weights saved to disk are from epoch 44 (best val_acc = 97.96%), not from the final epoch. The overfitting visible in epochs 45–50 is in discarded weights.
2. **Test accuracy confirms generalization.** The saved checkpoint achieves 98.77% on the held-out test set, which is close to its best val_acc. If the model had truly overfit, test accuracy would be significantly lower.

### Side-by-Side Overlay

![Training curves overlay](results_final/training_curves_overlay.png)

---

## Confusion Matrices (Test Set)

| MambaVision | ViT |
|---|---|
| ![MambaVision confusion matrix](results_final/cm_mamba_test.png) | ![ViT confusion matrix](results_final/cm_vit_test.png) |

### Per-Class Accuracy Comparison

![Per-class accuracy](results_final/per_class_accuracy.png)

---

## File Descriptions

### Core Scripts

**`config.py`**
Central settings file. All paths, hyperparameters, and class definitions live here. Every other script imports this. Change training settings here rather than in individual scripts.

Key values: `EPOCHS=50`, `BATCH_SIZE=32`, `EARLY_STOP_PATIENCE=15`, `LABEL_SMOOTHING=0.05`, `WEIGHT_DECAY=1e-4`

**`dataset.py`**
Loads images from disk, applies transforms, and returns PyTorch DataLoaders for train/val/test splits. Training split uses augmentation (random flip, color jitter); val/test use resize + normalize only. Also computes per-class weights to handle class imbalance.

**`model.py`**
Defines `MambaVisionClassifier`: loads `nvidia/MambaVision-T-1K` from HuggingFace and attaches a custom head (`LayerNorm → Dropout → Linear`). Supports three fine-tuning strategies: `full` (all parameters), `partial` (last N backbone stages + head), `head_only` (head only). Includes `save_checkpoint` and `load_checkpoint` helpers.

**`train_mamba.py`**
Training loop for MambaVision. Runs up to 50 epochs with cosine LR decay, AMP (bfloat16), gradient clipping, and early stopping. Saves best checkpoint to `results/mambavision/checkpoints/best.pth`.

Usage: `python train_mamba.py --model_id nvidia/MambaVision-T-1K --epochs 50 --batch_size 32 --lr 3e-4 --strategy full`

**`train_vit.py`**
Training loop for ViT. Uses `ViTModel` (backbone only) + custom `ViTClassifier` head to avoid key-naming mismatches in transformers 5.x. Saves checkpoint as `results/vit/checkpoints/best.pth` in the same `.pth` format as MambaVision.

Usage: `python train_vit.py --epochs 50 --batch_size 32 --lr 1e-4 --strategy full`

**`evaluate_both.py`**
Loads both saved checkpoints and runs inference on the test set. Produces per-class precision/recall/F1, confusion matrices, a training curve overlay, and a summary table. Also measures inference speed (ms/image). Run this after both training jobs complete.

**`patch_mambavision.py`**
One-time patch script to run on the cluster after first downloading the MambaVision model. Fixes a bug in the HuggingFace cached code where `.item()` is called on a meta tensor during model initialization.

Run once on the cluster: `python patch_mambavision.py`

### SLURM Job Scripts

**`run_mamba.sbatch`** — submits MambaVision training (1x A100, 8 CPU, 32GB, 6h)

**`run_vit.sbatch`** — submits ViT training (1x A100, 8 CPU, 32GB, 6h)

**`run_eval.sbatch`** — submits evaluation (1x A100, 4 CPU, 16GB, 1h). Run after both training jobs finish.

All scripts activate the `mamba_env` conda environment, which must have `mamba-ssm` installed.

---

## Folder Structure

```
vit_mamba/
├── config.py
├── dataset.py
├── model.py
├── train_mamba.py
├── train_vit.py
├── evaluate_both.py
├── patch_mambavision.py
├── run_mamba.sbatch
├── run_vit.sbatch
├── run_eval.sbatch
├── requirements.txt
├── data/
│   ├── train/          1468 images
│   ├── val/             490 images
│   └── test/            487 images
├── results/
│   ├── mambavision/
│   │   ├── checkpoints/best.pth
│   │   ├── history.json
│   │   └── training_curves.png
│   ├── vit/
│   │   ├── checkpoints/best.pth
│   │   ├── history.json
│   │   └── training_curves.png
│   └── comparison/
│       ├── cm_mamba_test.png
│       ├── cm_vit_test.png
│       ├── per_class_accuracy.png
│       ├── training_curves_overlay.png
│       ├── metrics_mamba.json
│       ├── metrics_vit.json
│       └── summary.csv
└── logs/
    └── <job_id>.log
```

---

## How to Run

```bash
# 1. First-time setup: patch the MambaVision cached model (run once on cluster)
python patch_mambavision.py

# 2. Train both models (can be submitted simultaneously)
sbatch run_mamba.sbatch
sbatch run_vit.sbatch

# 3. After both jobs complete, run evaluation
sbatch run_eval.sbatch
```

Monitor a running job:
```bash
tail -f logs/mamba_<job_id>.log
squeue -u <username>
```

---

## Environment

- Cluster: MTSU hamilton.cs.mtsu.edu (SLURM, A100 GPUs)
- Conda env: `mamba_env` (includes `mamba-ssm`, `transformers>=5.12`, `torch`, `timm`)
- Python: 3.10

---

## Quick Reference

| | Value |
|---|---|
| Dataset size | 2445 images total (1468 / 490 / 487) |
| Input size | 224 x 224 RGB PNG |
| Training time | ~50 min/model on A100 |
| Fine-tuning strategy used | `full` (all parameters trainable) |
| MambaVision LR | backbone 3e-5, head 3e-4 |
| ViT LR | backbone 1e-5, head 1e-4 |
| Early stopping patience | 15 epochs |
