#!/bin/bash

#SBATCH --job-name=multi_task_wsd_linear_1batch
#SBATCH --output=run_all10_wsd_linear_1batch.out
#SBATCH --error=run_all10_wsd_linear_1batch.err
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-4:30
#SBATCH --account=def-kevinlb

python model.py
