#!/bin/bash
#SBATCH --job-name=nemotron_piezo
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:RTX5000Ada:1
#SBATCH --mem=56G
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

# ── Environment ────────────────────────────────────────────────
module purge
module load cuda/12.4
source $HOME/envs/nemotron_env/bin/activate

# Bypass interactive trust_remote_code prompt for Nemotron-H
export HF_HUB_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/ya4v/llavagraph1.6/piezo_finetune
mkdir -p logs checkpoints

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# ── Data already prepared locally (train/val/test.jsonl uploaded) ──
# python prepare_data.py --input training_data_v2.csv --output_dir ./data

# ── Fine-tune ──────────────────────────────────────────────────
yes | python finetune.py \
    --data_dir    ./data \
    --output_dir  ./checkpoints \
    --epochs      6 \
    --batch_size  1 \
    --grad_accum  16 \
    --lr          5e-5

echo "Done: $(date)"
