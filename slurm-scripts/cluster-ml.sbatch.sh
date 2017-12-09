#!/bin/bash
#SBATCH --job-name=cluster   # Job name
#SBATCH --time=INFINITE               # Time limit hrs:min:sec
#SBATCH --partition=atom-cpu
#SBATCH --output=log/%j.out
#SBATCH --array=0-31
#SBATCH --mem=8G
#SBATCH -n 1
#SBATCH --overcommit
set -x
#python3 retrain.py configs/Hindi.json $SLURM_ARRAY_TASK_ID Hindi $SLURM_ARRAY_TASK_ID
#python3 -m doctools.scripts.cost_new doctools/configs/Hindi.json  $SLURM_ARRAY_TASK_ID 

python3 -m doctools.scripts.cluster -c doctools/configs/malayalam.json -l ml -b $SLURM_ARRAY_TASK_ID -o /dev/null

#python -m  doctools.postproc.correction.test -c doctools/configs/malayalam.json -l hi -b $SLURM_ARRAY_TASK_ID -o doctools/outdir
