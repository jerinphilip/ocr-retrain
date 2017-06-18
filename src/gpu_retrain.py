from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import torch
from parser.webtotrain import read_book
from gpu_ocr.train import train
import numpy as np


if __name__ == '__main__':
    pagewise = read_book("/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0006/", 'line')
    xs = []
    ys = []
    for (imgs, truths) in pagewise:
        for img, truth in zip(imgs, truths):
            x = torch.Tensor(np.array(img, dtype=np.float32).T)
            t, h = x.size()
            x = x.contiguous().view(1, t, h)
            xs.append(x)
            ys.append(truth)
    print(len(xs), len(ys))

    #trim = len(xs)
    files = train(xs[:trim], ys[:trim], "/OCRData2/ocr/retrain/src/parameters/lookups/Malayalam.txt")

