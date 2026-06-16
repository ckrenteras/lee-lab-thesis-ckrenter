#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module load cuda/12.9.0-cinr

source ~/env/adp_thesis/bin/activate

ARGS="--arch $ARCH"
[ -n "$DEPTH" ]   && ARGS="$ARGS --depth $DEPTH"
[ -n "$TAG" ]     && ARGS="$ARGS --tag $TAG"
[ -n "$NTRIALS" ] && ARGS="$ARGS --n-trials $NTRIALS"

python optuna_hyperparams.py $ARGS