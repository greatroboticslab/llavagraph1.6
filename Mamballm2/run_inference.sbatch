#!/bin/bash
#SBATCH --job-name=mamba_llm_infer
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_infer.out
#SBATCH --error=logs/%j_infer.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba_env

export HF_HUB_TRUST_REMOTE_CODE=1

cd /projects/ya4v/llavagraph1.6/mamba_llm
mkdir -p logs results

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "-----------------------------------------------"

python inference.py \
    --adapter_dir ./checkpoints/final_adapter \
    --data_dir    ./data \
    --output_dir  ./results \
    --split       test

echo "-----------------------------------------------"
echo "Done: $(date)"
