
import sys
import torch

root = '/OCRData2/ocr/gpu/'
sys.path.insert(0, root)


# Test 1 - Parser
from parser import read_book
book = '/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0006/'
pagewise = read_book(book)

# Test 2 - DataLoader
from ocr.preproc import DataLoader 
loader = DataLoader(pagewise=pagewise)

print("Data split:",)
print(list(map(len, [loader.split.train, loader.split.val, loader.split.test])))


# Test-3 - Analyze dimensions
seq, target = loader.split.train[0]
print(seq.shape)

# Test-4 - Get codebook
from parser.lookup import codebook
lookup_filename = "/OCRData2/ocr/retrain/src/parameters/lookups/Malayalam.txt"
lmap, ilmap = codebook(lookup_filename)

from ocr.pytorch.util import gpu_format, bucket

# Test-5 - Convert to GPU
convert_gpu = lambda x: list(map(gpu_format(lmap), x))
train_set = convert_gpu(loader.split.train)
validation_set = convert_gpu(loader.split.val)

# Test-6 - Bucketing
max_size = 2**15
train_batches = bucket(train_set, max_size=max_size)
val_batches = bucket(validation_set, max_size=max_size)
x, z, lx, lz = train_batches[0]
print(x, z, lx, lz)

# Test-7 - To OCR
from ocr.pytorch.engine import Engine
satisfactory = False
savepath = "file.tar"
kwargs = {}
try:
    with open(savepath, "rb") as savefile:
        kwargs = torch.load(savefile)
except FileNotFoundError:
    kwargs = {
            "input_size": 32,
            "output_classes": len(lmap.keys()),
            "lr": 1e-3,
            "momentum": 0.5
    }


while not satisfactory:
    engine = Engine(**kwargs)
    val_err, train_err = engine.train(train_batches, 
            val_batches, debug=True)
    kwargs['save'] = engine.export()
    torch.save(kwargs, open(savepath, "wb+"))
    print("Val, Train:", "(%.2lf, %.2lf)"%(val_err, train_err))
