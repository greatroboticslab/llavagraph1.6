# Mamballm2 — Nemotron-H-8B Fine-Tuning for Piezoelectric Waveform Analysis

LoRA fine-tuning of NVIDIA Nemotron-H-8B-Base-8K on piezoelectric actuator waveform data.
The model learns to analyze waveform features and generate structured diagnostic reports with correction vectors.

---

## Model

| Property | Value |
|---|---|
| Base model | nvidia/Nemotron-H-8B-Base-8K |
| Architecture | Hybrid Mamba-Transformer (8B parameters) |
| Fine-tuning method | LoRA (r=16, alpha=32) |
| Trainable parameters | 34.8M (0.43% of total) |
| Precision | bf16 (no quantization) |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj, out_proj |

---

## Dataset

| Split | Samples |
|---|---|
| Train | 1,468 |
| Validation | 490 |
| Test | 487 |
| Total | 2,445 |

Waveform types: sine, square, ramp, noise, pulse.
Each sample contains a system prompt, waveform feature description (input), and a structured diagnostic report (output) with DEVIATION SUMMARY, TEMPORAL DEVIATION, VIBRATION ANALYSIS, and CORRECTION VECTOR sections.

Raw data: `training_data.csv` (7 MB). Processed splits: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

---

## Training

| Setting | Value |
|---|---|
| Hardware | 1x NVIDIA A100 80GB (SLURM, partition a100) |
| Cluster | Hamilton HPC, MTSU |
| Epochs | 5 |
| Effective batch size | 16 (batch=1, grad_accum=16) |
| Learning rate | 2e-4 (cosine schedule, warmup 5%) |
| Max sequence length | 1024 tokens |
| Optimizer | adamw_torch_fused |
| Gradient checkpointing | enabled (use_reentrant=False) |
| Wall time | ~10.5 hours |

Training loss progression (response tokens only, prompt masked):

| Epoch | Train Loss | Eval Loss | Eval Accuracy |
|---|---|---|---|
| 1 | 0.97 | 0.9372 | 74.4% |
| 2 | 0.58 | 0.6694 | 81.0% |
| 3 | 0.46 | 0.5763 | 83.5% |
| 4 | 0.40 | 0.5422 | 84.6% |
| 5 | 0.36 | 0.5368 | 84.8% |

No overfitting observed — eval loss declined consistently across all 5 epochs.

---

## Files

| File | Purpose |
|---|---|
| `finetune.py` | Main training script |
| `prepare_data.py` | Converts training_data.csv to JSONL splits |
| `generate_training_data.py` | Generates training examples from waveform features |
| `inference.py` | Runs the fine-tuned adapter on a data split |
| `evaluate.py` | Computes ROUGE-1/2/L metrics on inference output |
| `run_finetune.sbatch` | SLURM job script for training |
| `run_inference.sbatch` | SLURM job script for inference |
| `precompile_triton.sh` | Pre-compiles Triton CUDA kernels on a compute node |
| `visualize_training_quality.py` | Side-by-side waveform image vs training text |
| `visualize_principle.py` | Visualizes the diagnostic approach |
| `visualize_strategy.py` | Visualizes the correction strategy |

---

## Usage

**Prepare data:**
```bash
python prepare_data.py --input training_data.csv --output_dir ./data
```

**Fine-tune (SLURM):**
```bash
sbatch run_finetune.sbatch
```

**Run inference:**
```bash
python inference.py --adapter_dir ./checkpoints/final_adapter --split test
```

**Evaluate:**
```bash
python evaluate.py --predictions ./results/predictions_test.jsonl
```

---

## Dependencies

Tested on Python 3.10, CUDA 12.4.

```
transformers>=4.47
peft>=0.10
trl==1.7.0
accelerate>=0.27
datasets>=2.18
torch>=2.1 (bf16 + bfloat16 Mamba kernels)
mamba_ssm
rouge-score
```

The cluster environment requires pre-compiling Triton kernels before the first SLURM job.
Run `precompile_triton.sh` on a compute node (via `srun --pty bash`) once before submitting.
