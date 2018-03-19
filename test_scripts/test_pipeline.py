import sys
import torch
import pdb

# root = '/OCRData2/ocr/gpu/'
# sys.path.insert(0, root)


# Test 1 - Parser
from parser import read_book
book = '/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0002/'
# book = 'data/Advaita_Deepika/'
pagewise = read_book(book_path=book, unit='line')
print('finished reading data')
# Test 2 - DataLoader
from ocr.preproc import DataLoader 
print('Reading Data')
loader = DataLoader(pagewise=pagewise[2:4])

print("Data split:")
# print(list(map(len, [loader.split.train, loader.split.val, loader.split.test])))


# Test-3 - Analyze dimensions
seq, target = loader.split.train[0]


# Test-4 - Test concat function
# from ocr.util import concat
# seq_targ_1, seq_targ_2 = loader.split.train[:2]
# print(concat(seq_targ_1, seq_targ_2))

# Test-5 - Reduce using concat, test out
# width = 2**15
# from ocr.util import knot
# train = knot(loader.split.train, max_width=width)
# validation = knot(loader.split.test, max_width=width)
train = loader.split.train
validation = loader.split.test
# train = [(train[i][0], train[i][1]) for i in range(len(train)) if len(train[i][1])!=0]
# validation = [(validation[i][0], validation[i][1]) for i in range(len(validation)) if len(validation[i][1])!=0]
print(train[0][1])

# Test-6 - Convert to gpu ready
from ocr.pytorch.util import gpu_format
from parser.lookup import codebook
lookup_filename = "lookups/Malayalam.txt"
# lookup_filename = 'lookups/Hindi.txt'
lmap, ilmap = codebook(lookup_filename)

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
# fpath = 'lookups/Sanskrit.pkl'
# with open(fpath, "wb+") as f:
#     pickle.dump(lmap, f)
#     print("Saving:mapping")
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


#train_set = train_set[:300]
#validation_set = validation_set[:30]
from ocr.pytorch.coding import Decoder
engine = Engine(**kwargs)
# while not satisfactory:
print('Training')
val_err, train_err = engine.train(train_set, debug=True)
engine.test(validation_set)
decoder  = Decoder(lmap, ilmap)
for pair in (validation_set):
	sequence, target = pair
	prediction = engine.recognize(sequence, decoder)
	print('Prediction: %s'%prediction)
	
kwargs = engine.export()
# torch.save(kwargs, open(savepath, "wb+"))
