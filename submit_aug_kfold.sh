#!/bin/bash
#SBATCH --job-name=aug_kfold
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module load cuda/12.9.0-cinr
source ~/env/adp_thesis/bin/activate

ARCHS=("unet" "manet")
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

echo "Running augmentation experiments for arch: $ARCH"
python -u augment_kfold.py --arch "$ARCH"
