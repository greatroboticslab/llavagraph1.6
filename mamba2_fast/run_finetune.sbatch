#!/bin/bash
#SBATCH --job-name=mamba2_finetune
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:RTX5000Ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%j_finetune.out
#SBATCH --error=logs/%j_finetune.out

module load cuda/12.4

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba_env

pip install --quiet accelerate 'accelerate>=0.26.0' 2>&1 | tail -3
# Fast SSM kernels — training will run without these (falls back to the
# naive scan) but will be much slower. Installing them here too so the
# fine-tuning wall-clock time is representative, not an artifact of a
# missing kernel.
pip install --quiet mamba-ssm causal-conv1d 2>&1 | tail -5

cd /projects/ya4v/llavagraph1.6/mamba2_fast
mkdir -p logs

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "-----------------------------------------------"

python finetune_mamba2.py

echo "-----------------------------------------------"
echo "Done: $(date)"
