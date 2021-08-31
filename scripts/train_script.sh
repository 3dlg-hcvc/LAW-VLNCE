#!/bin/bash
#SBATCH --job-name=_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1   # Request GPU "generic resources"
#SBATCH --cpus-per-task=24   # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --ntasks-per-node=1  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=1-23:59     # DD-HH:MM:SS

# Activate virtual environment
source ~/vlnce_law/bin/activate

# Copy data, as necessary, for faster access
tar -xvf ~/data/scene_datasets/mp3d.tar.gz -C $SLURM_TMPDIR/

printenv | grep SLURM
set -x
srun -u \
python -u run.py \
    --exp-config vlnce_baselines/config/paper_configs/goal_config/cma_pm_aug.yaml \
    --run-type train
