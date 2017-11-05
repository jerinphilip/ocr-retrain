export LANG='en_US.UTF-8'
python doctools/hwnet/codes/deploy/hw.py \
    --img_folder /OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0006/0006_101SanjayanPhalithangal_Img_600_Original \
    --test_vocab_file sample.txt \
    --pretrained_path /users/jerin/ocr-retrain/doctools/hwnet/pretrained_models/ckpt_best.t7 \
    --save_dir /tmp/praveen-ocr
