#!/bin/sh
#SBATCH --job-name=ml-ocr    # Job name
#SBATCH --time=INFINITE               # Time limit hrs:min:sec
#SBATCH --partition=atom-cpu
#SBATCH --output=log/%j.out
#SBATCH --mem=8G
#SBATCH -n 1
#SBATCH --overcommit
#python3 retrain.py configs/Hindi.json $SLURM_ARRAY_TASK_ID Hindi
#python3 tests/test.py configs/malayalam.json  $SLURM_ARRAY_TASK_ID ml
python3 -m doctools.scripts.praveen_prepare -c doctools/configs/Hindi.json -l hi -b 1 -o output/debug 
