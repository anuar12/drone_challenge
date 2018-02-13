#!/bin/bash

#SBATCH --job-name=multi_task
#SBATCH --output=run_all10.out
#SBATCH --error=run_all10.err
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-5:00
#SBATCH --account=def-kevinlb

python model.py
