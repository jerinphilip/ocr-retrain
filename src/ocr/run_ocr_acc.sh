#!/bin/sh
#SBATCH --job-name=hi-ocr    # Job name
#SBATCH --time=INFINITE               # Time limit hrs:min:sec
#SBATCH --partition=atom-cpu
#SBATCH --output=output/out/%j.out
#SBATCH --array=0-31
#SBATCH --mem=4G
#SBATCH -n 1
#SBATCH --error=output/err/%j.err
#SBATCH --overcommit
python3 retrain.py configs/Hindi.json $SLURM_ARRAY_TASK_ID Hindi
