# Closed-Loop Piezo Actuator Waveform Classification

Classifying piezo actuator waveforms recorded under **closed-loop (PID) control** using two vision models: **MambaVision-T** and **ViT-Base**. Part of a journal paper comparing open-loop vs closed-loop classification performance.

**5 classes:** `noise` · `pulse` · `ramp` · `sine` · `square`

**Test set results:**
| Model | Accuracy | Inference Speed |
|-------|----------|-----------------|
| MambaVision-T | **95.60%** | **5.40 ms/image** |
| ViT-Base | 94.88% | 16.78 ms/image |

---

## Repository Contents

```
closed-loop-classification branch
│
├── generate_closed_loop_images.py   ← Step 1: raw data → PNG images
│
├── Closed_loop/
│   ├── README.md                    ← this file
│   ├── config.py                    ← all hyperparameters & paths
│   ├── train_mamba.py               ← Step 2: train MambaVision
│   ├── train_vit.py                 ← Step 3: train ViT
│   ├── evaluate_both.py             ← Step 4: evaluate & compare
│   ├── .gitignore                   ← excludes checkpoints & large logs
│   └── results/
│       ├── mambavision/
│       │   ├── training.log         ← full epoch log (28 epochs)
│       │   └── history.json         ← loss & accuracy per epoch (parsed)
│       ├── vit/
│       │   └── history.json         ← loss & accuracy per epoch (parsed)
│       └── comparison/
│           ├── summary.csv                  ← accuracy + speed table
│           ├── cm_mamba_test.png            ← MambaVision confusion matrix
│           ├── cm_vit_test.png              ← ViT confusion matrix
│           ├── per_class_accuracy.png       ← per-class recall bar chart
│           ├── training_curves_overlay.png  ← both models on same axes
│           ├── learning_curve_mamba.png     ← MambaVision overfitting plot
│           ├── learning_curve_vit.png       ← ViT overfitting plot
│           ├── speed_comparison.png         ← inference speed bar chart
│           ├── metrics_mamba.json           ← full classification report
│           └── metrics_vit.json             ← full classification report
│
└── open_close_data/close/           ← Step 0: raw experimental data
    ├── noise/   (2 .txt files)
    ├── pulse/   (200 .txt files)
    ├── ramp/    (200 .txt files)
    ├── sine/    (200 .txt files)
    └── square/  (200 .txt files)
```

> **Not in this repo:** the generated PNG image dataset (`~4.8 GB`, `~35k images`) and model checkpoints (`~580 MB`). Run Step 1 to regenerate images locally. Checkpoints must be recreated by training.

---

## Prerequisites

```bash
pip install torch torchvision transformers timm scipy numpy matplotlib seaborn scikit-learn pandas tqdm Pillow
```

This code was developed on **Apple Silicon (MPS)**. It auto-detects `mps → cuda → cpu` in that order.

The training scripts also require `mambavision/` (the open-loop module directory) to be present in the repo root because `train_mamba.py` reuses `mambavision/model.py` and `mambavision/dataset.py`.

---

## Raw Data Format

Each `.txt` file in `open_close_data/close/{class}/` is a Moku:Go interferometer log with this structure:

```
Sample Frequency = 1000
D:-38421
D:-38422
D:-38420
...
```

- `Sample Frequency`: sampling rate in Hz (default 1000 Hz if not found)
- `D:` prefix: raw integer D-count from the interferometer

The noise class has only **2 experimental files** (10s and 30s recordings). All other classes have **200 files** each, covering frequencies from ~1 Hz to ~400 Hz.

---

## Step 1 — Generate Images

**Script:** `generate_closed_loop_images.py` (repo root)

Converts all raw `.txt` files into time-domain PNG images ready for training.

### What it does, in order

**1. Load raw data**
Reads every `.txt` file in `open_close_data/close/{class}/`. Parses the sample frequency from the header and collects all `D:` lines into a float64 numpy array.

**2. Convert raw D-counts to nanometres**
Uses the HeNe laser relative displacement formula — identical to `process_raw_8.py`:
```
displacement (nm) = (D - D[0]) × (632.991372 / 8.0)
```
`D[0]` is the first sample, so all displacements are relative to the start of the recording.

**3. Extract windows**
4 windows are extracted from each recording at fixed sample positions:
```
WIN_STARTS = [480, 1000, 2000, 3500]   # each 260 samples wide
```
Windows that fall outside the recording length are zero-padded. This gives 4× more training samples per file without changing the visual style.

**4. Generate synthetic data (PID simulation)**
Because noise has only 2 experimental files, and to increase dataset variety for all classes, **300 synthetic recordings are generated per class** using `scipy.signal`.

Each synthetic recording simulates a discrete closed-loop PID system with a second-order piezo plant model `G(s) = wn² / (s² + 2ζwn·s + wn²)`. PID gains and plant parameters are randomised per sample:
- `Kp ∈ [1, 4]`, `Ki ∈ [20, 100]`, `Kd ∈ [0.005, 0.02]`
- `wn ∈ [150, 300] rad/s`, `ζ ∈ [0.4, 0.8]`

The reference signal driving the PID differs per class to produce physically appropriate waveforms:
| Class | Reference Signal |
|-------|-----------------|
| `noise` | White noise `N(0, 1000²)` |
| `sine` | `A·sin(2πft)`, f ∈ [1, 200] Hz |
| `square` | `A·square(2πft)`, f ∈ [1, 200] Hz |
| `ramp` | `A·sawtooth(2πft)`, f ∈ [1, 200] Hz |
| `pulse` | `A·square(2πft, duty∈[0.05,0.15])`, f ∈ [1, 200] Hz |

Synthetic amplitude is calibrated to match experimental D-count ranges per class.

**5. Dataset split (stratified)**
All windows (experimental + synthetic) are split per class:
- Train: 70% · Val: 15% · Test: 15%
- Stratified so each class has the same proportions across splits

**6. Augmentation (train split only)**
Each training window produces **4 additional augmented copies** (N_AUG = 4):
- Amplitude scale: random ×[0.85, 1.15]
- DC offset: ±5% of signal range
- Gaussian noise: σ = 2% of signal std

**7. Save PNG images**
Each window (original + augmented) is saved as a PNG matching the open-loop image style exactly:
```
figsize = (10, 5) inches    dpi = 150
xlabel  = "Time (ms)"       ylabel = "Relative Displacement (nm)"
title   = "{stem} - {label} (Indices {start}-{end})"
linewidth = 0.7             grid = True
```

**Final dataset size:**
| Split | Images |
|-------|--------|
| Train | 32,205 |
| Val   | 1,381  |
| Test  | 1,386  |
| **Total** | **34,972** |

**Output:**
```
open_close_data/close_images/
├── train/  noise/  pulse/  ramp/  sine/  square/
├── val/    noise/  pulse/  ramp/  sine/  square/
├── test/   noise/  pulse/  ramp/  sine/  square/
└── metadata.csv   (split, class, filename, source_stem, win_start, augmentation)
```

### Run
```bash
# from repo root
python generate_closed_loop_images.py
```

---

## Step 2 — Train MambaVision

**Script:** `Closed_loop/train_mamba.py`

Fine-tunes `nvidia/MambaVision-T-1K` (~21.7M parameters) on the generated dataset.

### What it does, in order

1. **Imports config** from `Closed_loop/config.py` and monkey-patches the `mambavision` module so `mambavision/dataset.py` and `mambavision/model.py` pick up the closed-loop paths automatically.

2. **Builds dataloaders** via `mambavision/dataset.py` → `build_dataloaders()`. Train transform: Resize(224,224) + RandomHorizontalFlip + ColorJitter + ToTensor + ImageNet normalise. Val/test: Resize + ToTensor + normalise.

3. **Computes class weights** from training label distribution (via `compute_class_weights`) and passes them to `CrossEntropyLoss` to handle any remaining class imbalance.

4. **Builds model** via `mambavision/model.py` → `build_model(model_id="nvidia/MambaVision-T-1K", finetune_strategy="partial")`. Partial strategy: freezes early stages, unfreezes last 2 stages + classification head.

5. **Optimizer:** AdamW with differential learning rates:
   - Backbone params: `lr × 0.1 = 3×10⁻⁵`
   - Head params: `lr = 3×10⁻⁴`
   - Weight decay: `1×10⁻⁴`

6. **Scheduler:** `CosineAnnealingLR(T_max = epochs − warmup_epochs, eta_min = 1×10⁻⁶)`. Warmup: scheduler does not step for the first 3 epochs.

7. **Training loop:** For each epoch — forward pass, `CrossEntropyLoss` (label smoothing = 0.05), backward, gradient clip (max norm 5.0), step. Logs train and val loss/accuracy per epoch.

8. **Early stopping:** patience = 15. If val accuracy does not improve for 15 consecutive epochs, training stops.

9. **Saves** best checkpoint to `results/mambavision/checkpoints/best.pth` (state dict + epoch + val_acc). Saves `history.json` and `training_curves.png` on completion.

### Our training run
- Stopped manually at epoch 28 after plateau
- **Best val accuracy: 96.38%** (epoch 27, saved to `best.pth`)
- Train accuracy at stop: ~99.84%
- Train/val gap: ~3.4% — mild overfitting, not severe

### Run
```bash
cd Closed_loop
python train_mamba.py

# with custom settings:
python train_mamba.py --epochs 50 --batch_size 16 --lr 3e-4 --strategy partial --device mps
```

**CLI flags:**
| Flag | Default | Options |
|------|---------|---------|
| `--epochs` | 50 | any int |
| `--batch_size` | 16 | any int |
| `--lr` | 3e-4 | float |
| `--strategy` | `partial` | `full` / `head_only` / `partial` |
| `--device` | auto-detected | `mps` / `cuda` / `cpu` |
| `--results_dir` | `results/mambavision` | path |
| `--ckpt_dir` | `results/mambavision/checkpoints` | path |

---

## Step 3 — Train ViT

**Script:** `Closed_loop/train_vit.py`

Fine-tunes `google/vit-base-patch16-224-in21k` (85.8M parameters total, **14.2M trainable**) on the same dataset.

### What it does, in order

1. **Builds dataset** using its own `WaveformDataset` class (does not depend on `mambavision/`). Scans `open_close_data/close_images/{split}/{class}/*.png` and assigns integer labels by class index.

2. **Loads ViT** from HuggingFace:
   ```python
   ViTForImageClassification.from_pretrained(
       "google/vit-base-patch16-224-in21k",
       num_labels=5,
       ignore_mismatched_sizes=True
   )
   ```

3. **Freezes all parameters**, then selectively unfreezes:
   ```python
   # Freeze everything
   for param in model.parameters():
       param.requires_grad = False

   # Unfreeze last 2 encoder blocks (blocks 10 and 11 of 12)
   for block in model.vit.encoder.layer[-2:]:
       for param in block.parameters():
           param.requires_grad = True

   # Unfreeze final LayerNorm
   for param in model.vit.layernorm.parameters():
       param.requires_grad = True

   # Unfreeze classifier head
   for param in model.classifier.parameters():
       param.requires_grad = True
   ```
   Result: **14.2M trainable / 85.8M total**

4. **Optimizer:** AdamW with differential learning rates:
   - Backbone params (unfrozen blocks + LayerNorm): `lr × 0.1 = 3×10⁻⁵`
   - Classifier head: `lr = 3×10⁻⁴`
   - Weight decay: `1×10⁻⁴`

5. **Scheduler:** `CosineAnnealingLR(T_max = epochs, eta_min = 1×10⁻⁶)`

6. **Loss:** `CrossEntropyLoss(label_smoothing=0.05)` — no class weights (dataset is balanced at this point).

7. **Training loop:** Forward pass using `model(pixel_values=imgs).logits`, backward, gradient clip (max norm 5.0). Early stopping patience = 15.

8. **Saves** best checkpoint to `results/vit/checkpoints/best/` in HuggingFace `save_pretrained()` format (reloadable with `from_pretrained()`). Saves `history.json` and `training_curves.png` on completion.

### Our training run
- Stopped manually at epoch 30 after plateau
- **Best val accuracy: 95.58%** (epoch 29, saved to `checkpoints/best/`)
- Train accuracy at stop: ~98.46%
- Train/val gap: ~2.9% — mild overfitting, not severe

### Run
```bash
cd Closed_loop
python train_vit.py

# with custom settings:
python train_vit.py --epochs 50 --batch_size 16 --lr 3e-4 --device mps
```

**CLI flags:**
| Flag | Default | Options |
|------|---------|---------|
| `--epochs` | 50 | any int |
| `--batch_size` | 16 | any int |
| `--lr` | 3e-4 | float |
| `--device` | auto-detected | `mps` / `cuda` / `cpu` |
| `--results_dir` | `results/vit` | path |
| `--ckpt_dir` | `results/vit/checkpoints` | path |

---

## Step 4 — Evaluate Both Models

**Script:** `Closed_loop/evaluate_both.py`

Loads both saved checkpoints and evaluates them on the test set. Generates all comparison figures.

### What it does, in order

1. **Loads test dataset** from `open_close_data/close_images/test/` (1,386 images, 5 classes).

2. **Loads MambaVision checkpoint** from `results/mambavision/checkpoints/best.pth` using `mambavision/model.py → load_checkpoint()`.

3. **Loads ViT checkpoint** from `results/vit/checkpoints/best/` using `ViTForImageClassification.from_pretrained()`.

4. **Runs inference** for each model — full test set forward pass with `torch.no_grad()`, collects logits → softmax probabilities → argmax predictions.

5. **Measures inference speed** — 30 batches of size 32, with 3-batch warmup, averaged. On MPS: includes `torch.mps.synchronize()` for accurate timing.

6. **Computes metrics** per model:
   - Top-1 accuracy
   - Per-class precision, recall, F1 (via `sklearn.metrics.classification_report`)

7. **Saves all outputs** to `results/comparison/`:

| Output file | What it shows |
|-------------|---------------|
| `summary.csv` | Accuracy (%), ms/image, per-class F1 for both models |
| `cm_mamba_test.png` | Row-normalised confusion matrix, raw counts annotated |
| `cm_vit_test.png` | Same for ViT |
| `per_class_accuracy.png` | Grouped bar chart: per-class recall side by side |
| `training_curves_overlay.png` | Loss and accuracy curves for both models on one plot |
| `learning_curve_mamba.png` | MambaVision train vs val with best-epoch marker |
| `learning_curve_vit.png` | ViT train vs val with best-epoch marker |
| `speed_comparison.png` | Bar chart: ms/image ± std |
| `metrics_mamba.json` | Full sklearn classification report (JSON) |
| `metrics_vit.json` | Full sklearn classification report (JSON) |

### Run
```bash
cd Closed_loop
python evaluate_both.py              # test set (default)
python evaluate_both.py --split val  # val set instead
python evaluate_both.py --batch_size 64  # larger batch for speed measurement
```

---

## Configuration Reference (`Closed_loop/config.py`)

All hyperparameters are centralised here. Both training scripts import this file.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DATA_DIR` | `../open_close_data/close_images` | Root of generated image dataset |
| `CLASSES` | `["noise","pulse","ramp","sine","square"]` | Class names (order = label index) |
| `NUM_CLASSES` | 5 | Number of output classes |
| `MODEL_ID` | `nvidia/MambaVision-T-1K` | HuggingFace model ID for MambaVision |
| `FINETUNE_STRATEGY` | `partial` | `partial` / `full` / `head_only` |
| `UNFREEZE_LAST_N_STAGES` | 2 | How many MambaVision stages to unfreeze |
| `IMG_HEIGHT / IMG_WIDTH` | 224 | Input resolution |
| `IMAGENET_MEAN/STD` | [0.485,0.456,0.406] / [0.229,0.224,0.225] | Normalisation |
| `EPOCHS` | 50 | Max epochs (early stopping may cut short) |
| `BATCH_SIZE` | 16 | Batch size |
| `LEARNING_RATE` | 3e-4 | Base LR (backbone gets ×0.1) |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `WARMUP_EPOCHS` | 3 | MambaVision LR warmup (no scheduler step) |
| `LABEL_SMOOTHING` | 0.05 | CrossEntropyLoss label smoothing |
| `EARLY_STOP_PATIENCE` | 15 | Epochs without val improvement before stop |
| `RANDOM_SEED` | 42 | For reproducibility |
| `DEVICE` | `mps` → `cuda` → `cpu` | Auto-detected at import |

---

## Detailed Results

### Test Set Accuracy & Speed

| Model | Accuracy (%) | Inference (ms/image) |
|-------|-------------|----------------------|
| MambaVision-T | **95.60** | **5.40 ± σ** |
| ViT-Base | 94.88 | 16.78 ± σ |

MambaVision is **3.1× faster** at inference while achieving higher accuracy.

### Per-Class F1 Score (Test Set)

| Class | MambaVision | ViT | Notes |
|-------|-------------|-----|-------|
| noise | 1.000 | 1.000 | Perfectly classified by both |
| pulse | 0.997 | 0.998 | Near-perfect |
| square | 0.972 | 0.973 | Strong for both |
| ramp | 0.919 | 0.897 | Hardest class |
| sine | 0.910 | 0.897 | Hardest class |

`ramp` and `sine` are hardest — short time windows of these signals can look visually similar when frequency content overlaps.

### Overfitting Analysis

Neither model shows severe overfitting (val accuracy never collapses):

| Model | Best Val Acc | Epoch | Train Acc at Stop | Gap |
|-------|-------------|-------|-------------------|-----|
| MambaVision | 96.38% | 27/28 | ~99.84% | ~3.5% |
| ViT | 95.58% | 29/30 | ~98.46% | ~2.9% |

A ~3% train/val gap is normal for models of this size (~22M and ~86M params) fine-tuned on ~32k images. See `results/comparison/learning_curve_*.png` for the full curves.
