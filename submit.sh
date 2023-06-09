#!/bin/bash

#SBATCH --job-name=neuralucb
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm_log/S-%x.%j.out

ml purge
ml load cuda/11.1
eval "$(conda shell.bash hook)"
conda activate py38

COMMAND="python main.py --num_pca 8 --T 100 --topk 500"
srun $COMMAND
