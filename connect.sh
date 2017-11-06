#!/bin/sh
#SBATCH --job-name=ml-ocr    # Job name
#SBATCH --time=3:00:00               # Time limit hrs:min:sec
#SBATCH --partition=cosmos-debug
#SBATCH --output=log/%j.out
#SBATCH --array=0-31
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --qos=bogan
#SBATCH -n 1
#SBATCH --overcommit
export LANG='en_US.UTF-8'
python doctools/hwnet/codes/deploy/hw.py \
    -c doctools/configs/Hindi.json -l hi -b $SLURM_ARRAY_TASK_ID -o /OCRData2/praveen-intermediate/
    #--img_folder /OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0006/0006_101SanjayanPhalithangal_Img_600_Original \
    #--test_vocab_file sample.txt \
    #--pretrained_path /users/jerin/ocr-retrain/doctools/hwnet/pretrained_models/ckpt_best.t7 \
    #--save_dir /tmp/praveen-ocr
#python3 -m doctools.scripts.praveen_prepare\
