#!/bin/sh
#SBATCH --job-name=ml-ocr    # Job name
#SBATCH --time=INFINITE               # Time limit hrs:min:sec
#SBATCH --partition=atom-cpu
#SBATCH --output=outputs/out/%j.out
#SBATCH --array=0-31
#SBATCH --mem=4G
#SBATCH -n 1
#SBATCH --overcommit
#python3 retrain.py configs/Hindi.json $SLURM_ARRAY_TASK_ID Hindi
python3 tests/test.py configs/malayalam.json  $SLURM_ARRAY_TASK_ID ml
