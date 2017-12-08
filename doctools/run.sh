#!/bin/sh
#SBATCH --job-name=hi-ocr    # Job name
#SBATCH --time=INFINITE               # Time limit hrs:min:sec
#SBATCH --partition=atom-cpu
#SBATCH --output=doctools/outputs/%j.out
#SBATCH --array=0-31
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --error=doctools/outputs/%j.err
#SBATCH --overcommit
#python3 retrain.py configs/Hindi.json $SLURM_ARRAY_TASK_ID Hindi $SLURM_ARRAY_TASK_ID
#python3 -m doctools.scripts.cost_new doctools/configs/Hindi.json  $SLURM_ARRAY_TASK_ID 

# python -m doctools.scripts.cluster -o doctools/outdir/ -c doctools/configs/Hindi.json -l hindi -b $SLURM_ARRAY_TASK_ID

python -m  doctools.postproc.correction.test -c doctools/configs/malayalam.json -l hi -b $SLURM_ARRAY_TASK_ID -o doctools/outdir
