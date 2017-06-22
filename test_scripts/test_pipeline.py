import sys
import torch

root = '/OCRData2/ocr/gpu/'
sys.path.insert(0, root)


# Test 1 - Parser
from parser import read_book
book = '/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0006/'
pagewise = read_book(book, 'line')

# Test 2 - DataLoader
from ocr.preproc import DataLoader 
loader = DataLoader(pagewise=pagewise)

print("Data split:",)
print(list(map(len, [loader.split.train, loader.split.val, loader.split.test])))


# Test-3 - Analyze dimensions
seq, target = loader.split.train[0]
print(seq.shape)


# Test-4 - Test concat function
from ocr.util import concat
seq_targ_1, seq_targ_2 = loader.split.train[:2]
print(concat(seq_targ_1, seq_targ_2))

# Test-5 - Reduce using concat, test out
width = 2**9
from ocr.util import knot
#train = knot(loader.split.train, max_width=width)
train = loader.split.train
train = loader.split.train[:1]
#validation = knot(loader.split.val, max_width=width)
validation = loader.split.val
validation = []
print(train[0][1])

# Test-6 - Convert to gpu ready
from ocr.pytorch.util import gpu_format
from parser.lookup import codebook
lookup_filename = "/OCRData2/ocr/retrain/src/parameters/lookups/Malayalam.txt"
lmap, ilmap = codebook(lookup_filename)
print(gpu_format(lmap)(train[0]))


# Test-8 - Convert to GPU
convert_gpu = lambda x: list(map(gpu_format(lmap), x))
train_set = convert_gpu(train)
#print("Training:")
#for seq, targ in train_set:
#    print("Seq Size, Targ size:", seq.size(), targ.size())
validation_set = convert_gpu(validation)
#print("Validation:")
#for seq, targ in validation_set:
#    print("Seq Size, Targ size:", seq.size(), targ.size())

# Test-7 - Test training engine
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
            "lr": 3e-4,
            "momentum": 0.8
    }


while not satisfactory:
    engine = Engine(**kwargs)
    val_err, train_err = engine.train(train_set, validation_set, debug=True)
    kwargs['save'] = engine.export()
    torch.save(kwargs, open(savepath, "wb+"))
    #print("Val, Train:", "(%.2lf, %.2lf)"%(val_err, train_err))
    if val_err < 15.00:
        width = min(width*2, 2**15)
        train_set = knot(loader.split.train, max_width=width)
        validation_set = knot(loader.split.val, max_width=width)
        train_set = convert_gpu(train)
        validation_set = convert_gpu(validation)
