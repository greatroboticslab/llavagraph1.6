#!/bin/bash
#SBATCH --job-name=mamba2_eval_cls
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:RTX5000Ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_eval_cls.out
#SBATCH --error=logs/%j_eval_cls.out

module load cuda/12.4

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/ya4v/.conda/envs/mamba_env

cd /projects/ya4v/llavagraph1.6/mamba2_fast
mkdir -p logs

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "-----------------------------------------------"

python eval_classifier.py

echo "-----------------------------------------------"
echo "Done: $(date)"
