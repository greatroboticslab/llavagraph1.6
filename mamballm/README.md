# MambaLLM — Piezoelectric Actuator Waveform Diagnostics

Fine-tuning **Nemotron-H-8B-Base** (NVIDIA's hybrid Mamba+Transformer LLM) to diagnose
closed-loop piezoelectric actuator waveform deviations and prescribe quantitative corrections.

---

## What This Does

Given a measured waveform signal (sine, square, ramp, pulse, or noise) and its extracted
features, the model outputs a structured diagnosis:

1. **DEVIATION SUMMARY** — actual vs. ideal value with signed error for each metric
2. **TEMPORAL DEVIATION** — which time quarter (Q1–Q4, window 480–740 ms) deviates most and by how much
3. **VIBRATION ANALYSIS** — dominant distortion mechanism (hysteresis, creep, resonance), drift vs. steady-state
4. **CORRECTION VECTOR** — specific magnitudes for phase compensation (°), amplitude rescaling (%),
   harmonic suppression, hysteresis feedforward (nm), creep compensation (nm/ms), and per-quarter feedforward

### Example Output (model-generated)

```
DEVIATION SUMMARY: Peak displacement = 495.78 nm vs ideal 495.78 nm → 0 nm deviation.
RMS displacement = 206.67 nm vs ideal 350.57 nm → -143.90 nm RMS deficit.
THD = 3.63% vs ideal 0% → +3.63% excess distortion.
Measured phase = 76.10° vs FOPDT predicted 9.11° → +66.99° nonlinear phase excess.
Hysteresis (pos/neg asymmetry) = 18.62 nm vs ideal 0 nm → +18.62 nm asymmetry.

TEMPORAL DEVIATION (Q1–Q4): Q4 deviates most (-102.30 nm from window average).
Amplitude drift = 74.86 nm across the 260 ms window → significant creep.

VIBRATION ANALYSIS: Primary distortion from hysteresis and nonlinear phase excess.
Creep causes drifting amplitude. 3rd harmonic dominant at 0.0187 × fundamental.

CORRECTION VECTOR: Phase compensation = -66.99°. 3rd harmonic suppression > 1.87%.
Hysteresis feedforward = -18.62 nm/half-cycle. Creep compensation = -0.288 nm/ms.
```

---

## Pipeline Overview

```
Raw piezo measurements (CSV)
        │
        ▼
batch_feature_extraction.py     ← extract 50+ signal features per waveform
        │
        ▼  features_v3.csv (250 waveforms × 59 features)
        │
        ▼
generate_training_data.py       ← Gemini-2.5-flash generates deviation analysis
        │
        ▼  training_data_v3.csv (2,465 labeled examples)
        │
        ▼
cluster_scripts/prepare_data.py ← format as instruction-following JSONL
        │
        ▼  train.jsonl (1,479) / val.jsonl (493) / test.jsonl (493)
        │
        ▼
cluster_scripts/finetune.py     ← QLoRA fine-tune Nemotron-H-8B on SLURM
        │
        ▼  checkpoints/final_adapter/ (LoRA weights)
        │
        ▼
cluster_scripts/inference.py    ← run fine-tuned model on test set
```

---

## Model Details

| Item | Value |
|---|---|
| Base model | `nvidia/Nemotron-H-8B-Base-8K` |
| Architecture | Hybrid Mamba (SSM) + Transformer |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Fine-tuning method | QLoRA — r=16, α=32, dropout=0.05 |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Effective batch size | 16 (batch=1, grad_accum=16) |
| Optimizer | Paged AdamW 32-bit |
| LR schedule | Cosine, warmup 5%, peak 5e-5 |
| Max sequence length | 1024 tokens |
| Training epochs | 6 |
| Hardware | 1× NVIDIA RTX 5000 Ada (56 GB RAM, SLURM) |

### Mamba CUDA Kernel Patch

Nemotron-H's Mamba SSM layers use CUDA kernels that are incompatible with 4-bit quantized
weights. Both `finetune.py` and `inference.py` apply a forward-method patch at runtime:

```python
def _torch_forward_only(self, hidden_states, cache_params=None, ...):
    return self.torch_forward(hidden_states, cache_params, ...)

for module in model.modules():
    if hasattr(module, "torch_forward") and hasattr(module, "cuda_kernels_forward"):
        module.forward = types.MethodType(_torch_forward_only, module)
```

This forces the pure-PyTorch path (`torch_forward`) which works with quantized weights.

---

## Feature Set

Features are extracted by `batch_feature_extraction.py` and filtered per waveform type in
`prepare_data.py` so the model never sees features irrelevant to the current waveform.

| Category | Features |
|---|---|
| FFT | dominant frequency/amplitude, THD, spectral entropy/flatness/centroid, band energy ratio, harmonic ratios (2nd–5th), odd/even ratio |
| Time-domain | RMS, peak, crest factor, skewness, kurtosis, zero-crossing rate, peak density |
| Waveform-specific | sine: residual %, phase lag °; square: duty cycle, edge sharpness; noise: kurtosis, Gaussianity error, autocorr lag-1; ramp: rise/fall linearity, asymmetry; pulse: crest factor, duty cycle, ringing ratio |
| FOPDT model | predicted phase lag (°), amplitude attenuation — using K=380 nm/V, τ=250 µs, θ=5 µs |
| Hysteresis proxy | pos/neg half-cycle asymmetry (nm), half-cycle asymmetry fraction |
| Per-quarter (Q1–Q4) | peak, mean, RMS displacement (nm), sine-fit residual (%) for each 65 ms quarter in the 480–740 ms window |
| Drift | amplitude drift Q1→Q4 (nm), worst quarter index |

---

## File Structure

```
mamballm/
├── batch_feature_extraction.py     # Extract all 59 features from raw CSV waveform data
├── generate_training_data.py       # Generate training descriptions with Gemini-2.5-flash
├── clean_training_data.py          # Remove corrupted or duplicate training examples
├── features_v3.csv                 # Extracted features (250 waveforms)
├── training_data_v3.csv            # Labeled training data (2,465 examples)
├── train_result.png                # Training loss curve (job 53514, 6 epochs)
├── 2nd_train.png                   # Second training run metrics
├── logs/
│   ├── 53514_train.out             # Full training log (6-epoch run, RTX 5000 Ada)
│   └── 53516_infer.out             # Inference output on 20 test samples
└── cluster_scripts/
    ├── prepare_data.py             # CSV → train/val/test JSONL conversion
    ├── finetune.py                 # QLoRA fine-tuning script
    ├── inference.py                # Load adapter, run on test set, print comparisons
    ├── submit_finetune.sh          # SLURM submission script for training
    ├── submit_inference.sh         # SLURM submission script for inference
    ├── install_deps.sh             # Install Python dependencies in cluster env
    └── data/
        ├── train.jsonl             # 1,479 training examples
        ├── val.jsonl               # 493 validation examples
        └── test.jsonl              # 493 test examples
```

---

## Training Results

Training ran on a single NVIDIA RTX 5000 Ada for ~20 hours (job 53514, 6 epochs):

- Final eval loss: **~0.67**
- Token accuracy: **~88–92%**
- The model correctly identifies dominant distortion mechanisms (hysteresis, creep, resonance)
  and outputs quantitative correction vectors (phase °, amplitude %, harmonic suppression, nm feedforward)

See `logs/53514_train.out` for the full training log and `logs/53516_infer.out` for sample
REFERENCE vs. GENERATED comparisons on 20 held-out test examples.

---

## Reproduction

### 1. Install dependencies (on cluster)

```bash
bash cluster_scripts/install_deps.sh
```

### 2. Prepare data

```bash
# Run locally before uploading to cluster
python cluster_scripts/prepare_data.py \
    --input training_data_v3.csv \
    --output_dir cluster_scripts/data
```

### 3. Fine-tune (SLURM)

```bash
sbatch cluster_scripts/submit_finetune.sh
```

Edit `submit_finetune.sh` to set the correct cluster path (`cd /projects/...`) and
partition/GPU type for your environment.

### 4. Run inference (SLURM)

```bash
sbatch cluster_scripts/submit_inference.sh
```

### 5. Generate new training data (optional)

Requires a `GEMINI_API_KEY` environment variable:

```bash
export GEMINI_API_KEY=your_key_here
python generate_training_data.py --input features_v3.csv --output training_data_v3.csv
```

---

## SLURM Configuration

| Setting | Value |
|---|---|
| Partition | `research-gpu` |
| GPU | `RTX5000Ada:1` |
| Memory | 56 GB |
| Training time | 20 h |
| Inference time | 8 h |
| CUDA | 12.4 |

---

## Related Work

This project is part of the `llavagraph1.6` repository, which also includes:
- MambaVision-T waveform classification (image-based)
- Closed-loop piezoelectric actuator control experiments
- Training curve comparisons across architectures
