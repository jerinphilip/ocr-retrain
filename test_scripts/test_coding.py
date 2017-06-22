
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



# Test-3 - Analyze dimensions
seq, target = loader.split.train[0]


# Test-4 - Test concat function
from ocr.util import concat
seq_targ_1, seq_targ_2 = loader.split.train[:2]

# Test-5 - Reduce using concat, test out
width = 2**9
from ocr.util import knot
#train = knot(loader.split.train, max_width=width)
train = loader.split.train
#validation = knot(loader.split.val, max_width=width)
validation = loader.split.val

# Test-6 - Convert to gpu ready
from ocr.pytorch.util import gpu_format
from parser.lookup import codebook
lookup_filename = "/OCRData2/ocr/retrain/src/parameters/lookups/Malayalam.txt"
lmap, ilmap = codebook(lookup_filename)


# Test-8 - Convert to GPU
convert_gpu = lambda x: list(map(gpu_format(lmap), x))
train_set = convert_gpu(train)
validation_set = convert_gpu(validation)


from ocr.pytorch.coding import Decoder
decoder = Decoder(lmap, ilmap)


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

engine = Engine(**kwargs)
for seq, targ in validation_set:
    probs = engine.recognize(seq)
    string = decoder.decode(probs)
    print(decoder.to_string(targ)," = ", string)
